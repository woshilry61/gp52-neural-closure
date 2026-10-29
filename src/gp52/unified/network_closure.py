"""Train graph closures on physical flux/reaction moments, keeping the one-dimensional states."""
from pathlib import Path
import json,time,argparse,subprocess
import numpy as np
import torch
from .protocol import ROOT,COUNTS as ALL_COUNTS,LENGTH as ALL_LENGTH
from .topology import junctions,NETWORKS
COUNTS=np.array(ALL_COUNTS["seven"]);LENGTH=np.array(ALL_LENGTH["seven"]);D=2e-4;RATE=2.

torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest')
def graph(n,edge=False):
 counts=n-1 if edge else n
 N=sum(counts);off=np.r_[0,np.cumsum(counts)];a=np.zeros((N,N))
 for v,nv in enumerate(counts):
  for j in range(nv-1):i=off[v]+j;a[i,i+1]=a[i+1,i]=1
 for node in junctions('seven'):
  incoming=np.where(np.array(NETWORKS['seven']['outlet_nodes'])==node)[0]
  outgoing=np.where(np.array(NETWORKS['seven']['inlet_nodes'])==node)[0]
  ends=[off[v+1]-1 for v in incoming]+[off[v] for v in outgoing]
  for i in ends:
   for j in ends:
    if i!=j:a[i,j]=1
 return a/np.maximum(a.sum(1,keepdims=True),1)
class Head(torch.nn.Module):
 def __init__(self,inputs,outputs,adj):
  super().__init__();self.input=torch.nn.Linear(inputs,32)
  self.center=torch.nn.ModuleList(torch.nn.Linear(32,32) for _ in range(2))
  self.neighbor=torch.nn.ModuleList(torch.nn.Linear(32,32,bias=False) for _ in range(2))
  self.head=torch.nn.Linear(32,outputs)
  torch.nn.init.zeros_(self.head.weight);torch.nn.init.zeros_(self.head.bias)
  self.register_buffer('adj',torch.tensor(adj,dtype=torch.float32))
 def forward(self,x):
  z=torch.tanh(self.input(x))
  for local,message in zip(self.center,self.neighbor):z=torch.tanh(z+.25*(local(z)+message(self.adj@z)))
  return self.head(z)
def history_numpy(x,dt,tau):
 h=x.copy()
 for j in range(1,len(h)):h[j]=np.exp(-dt/tau)*h[j-1]+(-np.expm1(-dt/tau))*x[j]
 return h
def arrays_from_blood(blood,t):
 U=np.concatenate([d['q'][:,1:-1]/d['af'][:,1:-1] for d in blood],1)
 A=np.concatenate([d['af'][:,1:-1] for d in blood],1);R=np.sqrt(A/np.pi)
 dx=np.concatenate([np.full(n-1,LENGTH[v]/n) for v,n in enumerate(COUNTS)])
 xpos=np.concatenate([d['xf'][1:-1]/LENGTH[v] for v,d in enumerate(blood)])
 tau0=float(np.mean(R*R/D));dt=float(t[1]-t[0])
 hist=[history_numpy(U,dt,k*tau0) for k in [.01,.05,.2]]
 features=np.stack([np.broadcast_to(xpos,U.shape),np.log(A/.001),U/.2,*[h/.2 for h in hist],np.broadcast_to(dx/.002,U.shape),R/.02],-1)
 area=np.concatenate([d['area'] for d in blood],1)
 vel=np.concatenate([.5*(d['q'][:,1:]/d['af'][:,1:]+d['q'][:,:-1]/d['af'][:,:-1]) for d in blood],1)
 return dict(U=U,A=A,R=R,dx=dx,features=features,area=area,velocity=vel,t=t)
def reaction_features(y,ar,dt):
 h=y[:,0].transpose(0,2,1);g=y[:,1].transpose(0,2,1);off=np.r_[0,np.cumsum(COUNTS)]
 def gradient(z):
  return np.concatenate([np.gradient(z[:,:,off[v]:off[v+1]],LENGTH[v]/COUNTS[v],axis=2,edge_order=1) for v in range(len(COUNTS))],2)
 radius=np.sqrt(ar['area']/np.pi)[None]
 gh=gradient(h)*radius;gg=gradient(g)*radius;den=h*g+1e-4
 features=[h,g,h*g,gh,gg,np.broadcast_to(ar['area'][None]/.001-1,h.shape),np.broadcast_to(ar['velocity'][None]/.2,h.shape),np.tanh(gh*gg/den)]
 for tau in [.02,.1]:
  hh=history_numpy(gh.transpose(1,0,2),dt,tau).transpose(1,0,2)
  hg=history_numpy(gg.transpose(1,0,2),dt,tau).transpose(1,0,2)
  features.append(np.tanh(hh*hg/den))
 return np.stack(features,-1)
class Closure(torch.nn.Module):
 def __init__(self,ar):
  super().__init__();self.transport=Head(8,4,graph(COUNTS,True));self.reaction=Head(10,1,graph(COUNTS))
  self.register_buffer('tau_scale',torch.tensor([.1,.025],dtype=torch.float32))
  for k in ['features','U','A','R']:self.register_buffer(k,torch.tensor(ar[k],dtype=torch.float32))
 def coefficients(self):
  z=self.transport(self.features)
  gain=torch.nn.functional.softplus(z[...,0::2])/np.log(2)
  amp=self.U.abs()[...,None]*torch.sqrt(gain/6)
  tau=self.R.square()[...,None]/D*self.tau_scale*torch.exp(2*torch.tanh(z[...,1::2]))
  return amp,tau
 def factor(self,x):return torch.exp(.25*torch.tanh(self.reaction(x)[...,0]))
 def flux(self,gradient,dt):
  amp,tau=self.coefficients();amp=amp[1:];tau=tau[1:]
  decay=torch.exp(-dt/tau);h=-tau*torch.expm1(-dt/tau)
  f=decay[...,None];b=-h[...,None]*amp[...,None]*gradient[1:,:,None,:]
  step=1;T=len(b)
  while step<T:
   b=torch.cat([b[:step],b[step:]+f[step:]*b[:-step]],0)
   f=torch.cat([f[:step],f[step:]*f[:-step]],0)
   step*=2
  return self.A[1:,:,None]*(amp[...,None]*b).sum(2)
