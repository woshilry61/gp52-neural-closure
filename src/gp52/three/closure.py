"""Train graph closures on physical flux/reaction moments, keeping the one-dimensional states."""
from pathlib import Path
import json,time,argparse,subprocess
import numpy as np
import torch
from .reference import ROOT,COUNTS,LENGTH,D,RATE,flow
torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision('highest')
def graph(n,edge=False):
 counts=n-1 if edge else n
 N=sum(counts);off=np.r_[0,np.cumsum(counts)];a=np.zeros((N,N))
 for v,nv in enumerate(counts):
  for j in range(nv-1):i=off[v]+j;a[i,i+1]=a[i+1,i]=1
 ends=[off[1]-1,off[1],off[2]]
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
def arrays(dt=.002):
 blood,t=flow(dt=dt);E=sum(COUNTS-1)
 U=np.concatenate([d['q'][:,1:-1]/d['af'][:,1:-1] for d in blood],1)
 A=np.concatenate([d['af'][:,1:-1] for d in blood],1);R=np.sqrt(A/np.pi)
 dx=np.concatenate([np.full(n-1,LENGTH[v]/n) for v,n in enumerate(COUNTS)])
 xpos=np.concatenate([d['xf'][1:-1]/LENGTH[v] for v,d in enumerate(blood)])
 # Dimensionless history windows are fractions of the characteristic transverse mixing time.
 tau0=float(np.mean(R*R/D))
 hist=[history_numpy(U,dt,k*tau0) for k in [.01,.05,.2]]
 features=np.stack([np.broadcast_to(xpos,U.shape),np.log(A/.001),U/.2,*[h/.2 for h in hist],np.broadcast_to(dx/.002,U.shape),R/.02],-1)
 area=np.concatenate([d['area'] for d in blood],1)
 vel=np.concatenate([.5*(d['q'][:,1:]/d['af'][:,1:]+d['q'][:,:-1]/d['af'][:,:-1]) for d in blood],1)
 return dict(U=U,A=A,R=R,dx=dx,features=features,area=area,velocity=vel,t=t)
def reaction_features(y,ar,dt):
 h=y[:,0].transpose(0,2,1);g=y[:,1].transpose(0,2,1);off=np.r_[0,np.cumsum(COUNTS)]
 def gradient(z):
  return np.concatenate([np.gradient(z[:,:,off[v]:off[v+1]],LENGTH[v]/COUNTS[v],axis=2,edge_order=1) for v in range(3)],2)
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
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=3000);ap.add_argument('--seed',type=int,default=42)
 a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.manual_seed(a.seed)
 raw=dict(np.load(a.data));dt=float(raw['t'][1]);ar=arrays(dt)
 y=raw['y'];off=np.r_[0,np.cumsum(COUNTS)]
 grad=np.concatenate([np.diff(y[:,0,off[v]:off[v+1]],axis=1)/(LENGTH[v]/COUNTS[v]) for v in range(3)],1).transpose(2,1,0)
 rf=reaction_features(y,ar,dt).astype(np.float32)
 train=np.arange(24);val=np.arange(24,32)
 tensor=lambda x:torch.tensor(x,device='cuda',dtype=torch.float32)
 m=Closure(ar).cuda();gt=tensor(grad[:,:,train]);gv=tensor(grad[:,:,val])
 ft=tensor(raw['flux'][train].transpose(1,2,0)[1:]);fv=tensor(raw['flux'][val].transpose(1,2,0)[1:])
 fs=float(np.mean(raw['flux'][train]**2))
 xt=tensor(rf[train]);xv=tensor(rf[val])
 rt=tensor(raw['reaction'][train].transpose(0,2,1));rv=tensor(raw['reaction'][val].transpose(0,2,1))
 bt=tensor((RATE*y[train,0]*y[train,1]).transpose(0,2,1));bv=tensor((RATE*y[val,0]*y[val,1]).transpose(0,2,1))
 rs=float(np.mean(raw['reaction'][train]**2))
 monfile=(a.output/'gpu.csv').open('w');mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=monfile)
 start=time.perf_counter()
 try:
  for stage in ['transport','reaction']:
   params=list(getattr(m,stage).parameters());lr=torch.tensor(.001,device='cuda')
   opt=torch.optim.Adam(params,lr=lr,fused=True,capturable=True)
   lossfn=(lambda:((m.flux(gt,dt)-ft)**2).mean()/fs) if stage=='transport' else (lambda:((m.factor(xt)*bt-rt)**2).mean()/rs)
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):opt.zero_grad(set_to_none=True);loss=lossfn();loss.backward();opt.step()
   torch.cuda.current_stream().wait_stream(stream);cg=torch.cuda.CUDAGraph()
   with torch.cuda.graph(cg):opt.zero_grad(set_to_none=True);loss=lossfn();loss.backward();opt.step()
   best=np.inf
   for j in range(a.steps):
    if j==a.steps*2//3:lr.fill_(.0002)
    cg.replay()
    if (j+1)%100==0:
     with torch.no_grad():
      v=(((m.flux(gv,dt)-fv)**2).mean()/fs if stage=='transport' else ((m.factor(xv)*bv-rv)**2).mean()/rs).item()
     if v<best:best=v;torch.save(getattr(m,stage).state_dict(),a.output/f'{stage}.pt')
     row=dict(stage=stage,step=j+1,training=float(loss),validation=v,best=best,seconds=time.perf_counter()-start)
     print(json.dumps(row),flush=True)
     with (a.output/'history.jsonl').open('a') as h:h.write(json.dumps(row)+'\n')
   del cg,opt
   getattr(m,stage).load_state_dict(torch.load(a.output/f'{stage}.pt',weights_only=True))
  torch.save(m.state_dict(),a.output/'model.pt')
  with torch.no_grad():amp,tau=[z.cpu().numpy() for z in m.coefficients()]
  np.savez_compressed(a.output/'coefficients.npz',amp=amp,tau=tau,t=ar['t'])
  report=dict(seconds=time.perf_counter()-start,seed=a.seed,steps_per_head=a.steps,training_cases=train.tolist(),validation_cases=val.tolist(),test_cases=list(range(32,48)),selection='Validation physical flux/source error only',transport='Two causal local flux-memory channels; dimensionless physical timescale; no radial concentration states',reaction='Physical cross-sectional mean source supervision',device=torch.cuda.get_device_name())
  (a.output/'metrics.json').write_text(json.dumps(report,indent=2))
 finally:mon.terminate();mon.wait();monfile.close()
if __name__=='__main__':main()
