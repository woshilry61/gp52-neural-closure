"""Adapters retain the existing single- and three-vessel neural architectures."""
import os
import numpy as np
import torch
from .protocol import RUN,ROOT,COUNTS,LENGTH,plan
from .reference import flow
from gp52.three_independent.frozen import FrozenModel,arrays_from_blood
from gp52.three.closure import Closure
def sample_weights(name,x,t):
 spec=plan();sites=spec['networks'][name]['sensors'];times=np.array(spec['observation_times']);counts=COUNTS[name]
 off=np.r_[0,np.cumsum(counts)];wx=np.zeros((3,len(x)));wt=np.stack([np.interp(times,t,row) for row in np.eye(len(t))],1)
 for i,(v,pos) in enumerate(sites):
  grid=x[off[v]:off[v+1]];wx[i,off[v]:off[v+1]]=np.array([np.interp(pos,grid,row) for row in np.eye(len(grid))])
 return wx,wt
class Single:
 def __init__(self,kind,seed=42):
  os.environ['GP52_PREPARED_ASSETS']=str(RUN/'single/assets')
  from .single_c0 import C0
  from gp52.inference import Solver
  self.name=kind if kind=='C0' else f'{kind}_{seed}';self.kind=kind;self.seed=seed
  if kind=='C0':self.solver=C0()
  elif kind=='old':self.solver=Solver()
  else:
   root=RUN/'single'/f'train-{seed}'
   self.solver=Solver(transport_weights=root/'transport.pt',weights=root/'reaction.pt')
  self.t=self.solver.t;self.x=self.solver.x
  self.wx,self.wt=sample_weights('single',self.x,self.t)
  self.wxg=torch.tensor(self.wx,device='cuda',dtype=torch.float64);self.wtg=torch.tensor(self.wt,device='cuda',dtype=torch.float64)
  d,_,_=flow('single',tuple(COUNTS['single']),1,.002);self.data={'volume':d[0]['volume']}
  a=np.r_[np.zeros((1,7)),np.eye(7)]
  obs=self.predict(a)
  self.obs_hbasis=np.r_[obs[0:1,0].reshape(1,-1),(obs[1:,0]-obs[:1,0]).reshape(7,-1)]
 @torch.no_grad()
 def predict(self,a,full=False,batch_size=128):
  out=[]
  for start in range(0,len(a),batch_size):
   y,_=self.solver.solve(a[start:start+batch_size])
   if full:out.append(y.cpu().numpy())
   else:out.append(torch.einsum('pn,bsnt,qt->bspq',self.wxg,y[:,:2],self.wtg).cpu().numpy())
  return np.concatenate(out)
class Three(FrozenModel):
 def __init__(self,kind,seed=42):
  self.dtype=torch.float64;self.seed=None if kind=='C0' else seed;self.name=kind if kind=='C0' else f'{kind}_{seed}'
  self.blood,self.t,_=flow('three',tuple(COUNTS['three']),1,.002)
  self.ar=arrays_from_blood(self.blood,self.t);self.T=len(self.t);self.K=64;self.model=None
  if self.seed is not None:
   self.model=Closure(self.ar).cuda().eval()
   path=ROOT/f'runs/closure-final-{seed}/model.pt' if kind=='old' else RUN/'three'/f'train-{seed}'/'model.pt'
   state=torch.load(path,weights_only=True);state={k:v for k,v in state.items() if k not in ['features','U','A','R']}
   missing,unexpected=self.model.load_state_dict(state,strict=False)
   assert set(missing)=={'features','U','A','R'} and not unexpected
   for n,p in self.model.named_parameters():assert torch.equal(p.cpu(),state[n].cpu())
   self.model.requires_grad_(False)
  cache=RUN/'three/cache';cache.mkdir(exist_ok=True);p=cache/f'{self.name}.npz'
  if p.exists():data=dict(np.load(p))
  else:data=self.build();np.savez_compressed(p,**data)
  self.data=data
  for k,v in data.items():setattr(self,k,torch.as_tensor(v,device='cuda',dtype=torch.int32 if np.issubdtype(v.dtype,np.integer) else self.dtype).contiguous())
  self.Hflat=self.hbasis.reshape(self.T*64,8).T.contiguous()
  x=np.concatenate([d['x'] for d in self.blood]);wx,wt=sample_weights('three',x,self.t)
  w=np.einsum('pn,qt->pqtn',np.pad(wx,((0,0),(0,7))),wt).reshape(72,-1)
  self.obsw=torch.tensor(w.T,device='cuda',dtype=self.dtype)
  self.obs_hbasis=data['hbasis'].reshape(-1,8).T@w.T
  self.sensors=np.array([p[1] for p in plan()['networks']['three']['sensors']]);self.sample_t=np.array(plan()['observation_times'])
  if self.model is not None:self.model.reaction.double()
 @torch.no_grad()
 def batch(self,amps,full=False):
  from gp52.three.inverse_gpu import integrate
  aa=torch.as_tensor(amps,device='cuda',dtype=self.dtype)
  B=len(aa);H=(torch.cat([torch.ones((B,1),device='cuda',dtype=self.dtype),aa],1)@self.Hflat).reshape(B,self.T,64).contiguous()
  G=torch.empty_like(H);F=torch.empty_like(H);fac=torch.empty_like(H)
  args=(H,self.inv,self.old,self.bcg,self.vr,self.dec,self.ha,self.afad,self.el,self.er,self.ea,self.eb,self.dx)
  integrate[(B,)](*args,fac,G,F,self.T,64,self.seed is not None,False,num_warps=4)
  if self.seed is not None:
   from .network_gpu import features_kernel as precise_features
   from .reaction_inference import factor
   feat=torch.empty((B,self.T,57,10),device='cuda',dtype=self.dtype)
   precise_features[(B,)](H,G,self.radius,self.area,self.vel,self.left,self.right,self.gden,feat,self.T,64,57,num_warps=4)
   fac.fill_(1);fac[:,:,:57]=factor(self.model.reaction,feat)
   integrate[(B,)](*args,fac,G,F,self.T,64,True,True,num_warps=4)
  obs=torch.stack([H.reshape(B,-1)@self.obsw,G.reshape(B,-1)@self.obsw],1)
  if full:return torch.stack([H[:,:,:57],G[:,:,:57],F[:,:,:57]],1).permute(0,1,3,2).cpu().numpy()
  return obs.reshape(B,2,3,24).cpu().numpy()


def model(name,kind,seed=42):
 if name=='seven':
  from .seven_model import FrozenModel
  return FrozenModel(kind,seed)
 return {'single':Single,'three':Three}[name](kind,seed)
