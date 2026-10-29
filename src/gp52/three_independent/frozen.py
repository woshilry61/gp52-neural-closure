"""Frozen original solver with explicit blood input and freshly computed flow buffers.
The three learned checkpoints, transport equations and inverse baseline are unchanged.
"""
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from gp52.three.reference import ROOT,COUNTS,LENGTH,RATE,D,flow,geometry,operators,KNOTS
from gp52.three.closure import Closure,history_numpy
from gp52.three.inverse_gpu import integrate,features_kernel
from .transfer import RUN,load_flow
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
class FrozenModel:
 def __init__(self,seed=None,cache=RUN/'cache',observation_end=1.23,precision='double',blood_path=None):
  self.dtype=torch.float64 if precision=='double' else torch.float32
  self.seed=seed;self.name='ordinary_1d' if seed is None else f'neural_{seed}'
  self.blood,self.t=(flow() if blood_path is None else load_flow(str(blood_path),tuple(COUNTS),1,.002)[:2])
  self.ar=arrays_from_blood(self.blood,self.t);self.T=len(self.t);self.K=64
  self.model=None
  if seed is not None:
   self.model=Closure(self.ar).cuda().eval()
   state=torch.load(ROOT/f'runs/closure-final-{seed}/model.pt',weights_only=True)
   state={k:v for k,v in state.items() if k not in ['features','U','A','R']}
   missing,unexpected=self.model.load_state_dict(state,strict=False)
   assert set(missing)=={'features','U','A','R'} and not unexpected
   for name,parameter in self.model.named_parameters():
    assert torch.equal(parameter.cpu(),state[name].cpu()),name
   for p in self.model.parameters():p.requires_grad_(False)
  cache.mkdir(parents=True,exist_ok=True);p=cache/f'{self.name}.npz'
  if p.exists():data=dict(np.load(p))
  else:data=self.build();np.savez_compressed(p,**data)
  self.data=data
  for k,v in data.items():
   setattr(self,k,torch.as_tensor(v,device='cuda',dtype=torch.int32 if np.issubdtype(v.dtype,np.integer) else self.dtype).contiguous())
  self.Hflat=self.hbasis.reshape(self.T*64,8).T.contiguous()
  self.sensors=np.array([.03,.01,.017]);self.sample_t=np.linspace(.002,min(observation_end,self.t[-1]),24)
  w=np.zeros((3*24, self.T*64))
  off=np.r_[0,np.cumsum(COUNTS)]
  for v in range(3):
   xx=(np.arange(COUNTS[v])+.5)*LENGTH[v]/COUNTS[v]
   xs=np.interp(self.sensors[v],xx,np.arange(len(xx)));lo=int(np.floor(xs));hi=min(lo+1,len(xx)-1);wx=xs-lo
   for k,tt in enumerate(self.sample_t):
    jt=tt/.002;j0=int(np.floor(jt));j1=min(j0+1,self.T-1);wt=jt-j0
    for ji,tw in [(j0,1-wt),(j1,wt)]:
     for xi,xw in [(lo,1-wx),(hi,wx)]:w[v*24+k,ji*64+off[v]+xi]+=tw*xw
  self.obsw=torch.tensor(w.T,device='cuda',dtype=self.dtype)
  self.obs_hbasis=data['hbasis'].reshape(-1,8).T@w.T

 def build(self):
  blood,t=self.blood,self.t;dt=.002;n=COUNTS;g=geometry(n,1);N=58;K=64;T=len(t)
  idx=[i for i,z in enumerate(g['tags']) if z[0]=='axial'];a,b=g['edges'][idx].T;E=len(a)
  dx=np.concatenate([np.full(nv-1,LENGTH[v]/nv) for v,nv in enumerate(n)])
  amp=tau=None
  if self.model is not None:
   with torch.no_grad():amp,tau=[x.cpu().numpy().astype(float) for x in self.model.coefficients()]
  h=np.zeros((N,8));h[:57,0]=.2*np.exp(-((g['x']/.06-.25)/.1)**2)*(g['cell_v']==0)
  memory=np.zeros((E,2,8));basis=np.zeros((T,K,8));basis[0,:N]=h
  inv=np.zeros((T,K,K));old=np.zeros((T,K));bcg=old.copy();vr=old.copy()
  dec=np.zeros((T,K,2));ha=dec.copy();afad=dec.copy();volumes=[]
  knotbasis=np.stack([np.interp(t,KNOTS,z) for z in np.eye(7)],-1)
  for j in range(T):
   L,bc,vol,_,_=operators(blood,j,g,n,1);volumes.append(vol)
   if j:
    rhs_mem=np.zeros((N,8))
    if amp is not None:
     am=amp[j];ta=tau[j];dc=np.exp(-dt/ta);hs=-ta*np.expm1(-dt/ta);af=self.ar['A'][j]
     extra=af*np.sum(hs*am**2,1)/dx
     L=L+coo_matrix((np.r_[extra,-extra,-extra,extra],(np.r_[a,a,b,b],np.r_[a,b,a,b])),shape=(N,N)).tocsr()
     flux=(af[:,None,None]*am[:,:,None]*dc[:,:,None]*memory).sum(1)
     np.add.at(rhs_mem,a,-flux);np.add.at(rhs_mem,b,flux)
     dec[j,:E]=dc;ha[j,:E]=hs*am;afad[j,:E]=af[:,None]*am*dc
    lu=splu((L+diags(vol/dt)).tocsc())
    inv[j,:N,:N]=lu.solve(np.eye(N))
    old[j,:N]=vo/dt;bcg[j,:N]=bc.sum(1);vr[j,:57]=vo[:57]/vol[:57]
    h=lu.solve(vo[:,None]/dt*h+bc[:,0,None]*np.r_[0,knotbasis[j]][None,:]+rhs_mem)
    if amp is not None:memory=dc[:,:,None]*memory-hs[:,:,None]*am[:,:,None]*(h[b]-h[a])[:,None,:]/dx[:,None,None]
    basis[j,:N]=h
   vo=vol
  el=np.full(K,-1);er=el.copy();el[b]=np.arange(E);er[a]=np.arange(E)
  ea=np.zeros(K,int);eb=ea.copy();ea[:E]=a;eb[:E]=b;dd=np.ones(K);dd[:E]=dx
  le=np.arange(K);ri=le.copy();gd=np.ones(K);off=np.r_[0,np.cumsum(n)]
  for v in range(3):
   for i in range(off[v],off[v+1]):
    le[i]=max(i-1,off[v]);ri[i]=min(i+1,off[v+1]-1);gd[i]=(ri[i]-le[i])*LENGTH[v]/n[v]
  area=np.pad(self.ar['area'],((0,0),(0,7)));vel=np.pad(self.ar['velocity'],((0,0),(0,7)))
  return dict(hbasis=basis,inv=inv,old=old,bcg=bcg,vr=vr,dec=dec,ha=ha,afad=afad,
   el=el,er=er,ea=ea,eb=eb,dx=dd,left=le,right=ri,gden=gd,area=area,radius=np.sqrt(area/np.pi),vel=vel,volume=np.stack(volumes)[:,:57])

 @torch.no_grad()
 def batch(self,amps,full=False):
  aa=torch.as_tensor(amps,device='cuda',dtype=self.dtype)
  B=len(aa);H=(torch.cat([torch.ones((B,1),device='cuda',dtype=self.dtype),aa],1)@self.Hflat).reshape(B,self.T,64).contiguous()
  G=torch.empty_like(H);F=torch.empty_like(H);fac=torch.empty_like(H)
  args=(H,self.inv,self.old,self.bcg,self.vr,self.dec,self.ha,self.afad,self.el,self.er,self.ea,self.eb,self.dx)
  integrate[(B,)](*args,fac,G,F,self.T,64,self.seed is not None,False,num_warps=4)
  if self.seed is not None:
   feat=torch.empty((B,self.T,57,10),device='cuda')
   features_kernel[(B,)](H,G,self.radius,self.area,self.vel,self.left,self.right,self.gden,feat,self.T,64,num_warps=4)
   fac.fill_(1);fac[:,:,:57]=self.model.factor(feat)
   integrate[(B,)](*args,fac,G,F,self.T,64,True,True,num_warps=4)
  obs=torch.stack([H.reshape(B,-1)@self.obsw,G.reshape(B,-1)@self.obsw],1)
  if full:return torch.stack([H[:,:,:57],G[:,:,:57],F[:,:,:57]],1).permute(0,1,3,2).cpu().numpy()
  return obs.reshape(B,2,3,24).cpu().numpy()

 def predict(self,amps,full=False,batch_size=256):
  return np.concatenate([self.batch(amps[i:i+batch_size],full) for i in range(0,len(amps),batch_size)])
