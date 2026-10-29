"""Fair 1-D rollout: same base FV operators, volumes, boundary conditions and reaction splitting."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from .protocol import COUNTS as NC,LENGTH as NL,KNOTS,inputs
from .reference import flow as network_flow,geometry as network_geometry,operators as network_operators
COUNTS=NC['seven'];LENGTH=NL['seven'];RATE=2.
def flow():return network_flow('seven',tuple(COUNTS),1,.002)[:2]
def geometry(n,nr):return network_geometry('seven',n,nr)
def operators(blood,j,g,n,nr):return (*network_operators('seven',blood,j,g,n,nr),None,None)
def inlets():return inputs(),{}
def solve(amp=None,tau=None,factor=None,uniform=False,extra_diffusion=None):
 n=COUNTS;blood,t=flow();dt=t[1];g=geometry(n,1);ns=g['ns'];N=g['size']
 amplitudes,split=inlets();B=len(amplitudes)
 cin=np.stack([np.interp(t,KNOTS,a) for a in amplitudes])
 x=g['x'];h0=.2*np.exp(-((x/.06-.25)/.1)**2)*(g['cell_v']==0)
 H=np.broadcast_to(np.r_[h0,np.zeros(N-ns)][:,None],(N,B)).copy();G=np.ones_like(H);Fn=np.zeros_like(H)
 if uniform:H[:]=1.
 idx=np.array([i for i,z in enumerate(g['tags']) if z[0]=='axial']);a,b=g['edges'][idx].T;E=len(idx)
 dx=np.concatenate([np.full(nv-1,LENGTH[v]/nv) for v,nv in enumerate(n)])
 M=np.zeros((2,E,2,B));out=[];mass=[];defects=[];volume=[];minimum=0.;start=time.perf_counter();volold=None
 for j,tt in enumerate(t):
  L,bc,vol,q,cond=operators(blood,j,g,n,1)
  if j:
   if extra_diffusion is not None:
    af=np.concatenate([d['af'][j,1:-1] for d in blood])
    ex=af*extra_diffusion[j]/dx
    L=L+coo_matrix((np.r_[ex,-ex,-ex,ex],(np.r_[a,a,b,b],np.r_[a,b,a,b])),shape=(N,N)).tocsr()
   if amp is not None:
    am=amp[j];ta=tau[j];dec=np.exp(-dt/ta);hh=-ta*np.expm1(-dt/ta)
    af=np.concatenate([d['af'][j,1:-1] for d in blood])
    extra=af*np.sum(hh*am**2,axis=1)/dx
    L=L+coo_matrix((np.r_[extra,-extra,-extra,extra],(np.r_[a,a,b,b],np.r_[a,b,a,b])),shape=(N,N)).tocsr()
    memflux=af[None,:,None]*(am[None,:,:,None]*dec[None,:,:,None]*M).sum(2)
    rhs_mem=np.zeros((2,N,B))
    for k in [0,1]:
     np.add.at(rhs_mem[k],a,-memflux[k]);np.add.at(rhs_mem[k],b,memflux[k])
   else:rhs_mem=np.zeros((2,N,B))
   mat=L+diags(vol/dt);lu=splu(mat.tocsc())
   bH=np.ones((len(g["ends"]),B)) if uniform else np.vstack([cin[:,j],np.zeros((len(g["ends"])-1,B))])
   rhsH=volold[:,None]/dt*H+bc@bH+rhs_mem[0]
   rhsG=volold[:,None]/dt*G+bc@np.ones((len(g["ends"]),B))+rhs_mem[1]
   ans=lu.solve(np.c_[rhsH,rhsG]);H=ans[:,:B];GT=ans[:,B:]
   if amp is not None:
    for k,state in enumerate([H,GT]):
     gradient=(state[b]-state[a])/dx[:,None]
     M[k]=dec[:,:,None]*M[k]-hh[:,:,None]*am[:,:,None]*gradient[:,None,:]
   rate=0. if uniform else RATE
   fac=np.ones((N,B))
   if factor is not None:fac[:ns]=factor[:,j].T
   G=GT*np.exp(-rate*H*fac*dt)
   Fn[:ns]=volold[:ns,None]/vol[:ns,None]*Fn[:ns]+GT[:ns]-G[:ns];Fn[ns:]=0
   after=((G+Fn)*vol[:,None]).sum(0)
   balance=after-mass[-1]+dt*(L@GT-bc@np.ones((len(g["ends"]),B))-rhs_mem[1]).sum(0)
   defects.append(float(abs(balance).max()/vol.sum()))
  out.append(np.stack([H[:ns],G[:ns],Fn[:ns]]).transpose(2,0,1))
  mass.append(((G+Fn)*vol[:,None]).sum(0));volume.append(vol[:ns]);volold=vol
  minimum=min(minimum,float(min(H.min(),G.min(),Fn.min())))
 return np.stack(out,-1),dict(seconds=time.perf_counter()-start,mass_balance_max=max(defects),minimum=minimum,uniform_max_error=float(abs(np.stack(out,-1)[:,:2]-1).max()) if uniform else None),np.stack(volume).T
