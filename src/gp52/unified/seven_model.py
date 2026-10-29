"""Seven-vessel inference with unchanged three-vessel closure equations."""
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from .protocol import RUN,ROOT,COUNTS as NC,LENGTH as NL,KNOTS,plan
from .reference import flow as network_flow,geometry as network_geometry,operators as network_operators
from .network_closure import Closure,arrays_from_blood
from .network_gpu import integrate,features_kernel
COUNTS=NC['seven'];LENGTH=NL['seven'];NS=int(sum(COUNTS));KPAD=256
def geometry(n,nr):return network_geometry('seven',n,nr)
def operators(blood,j,g,n,nr):return (*network_operators('seven',blood,j,g,n,nr),None,None)
class FrozenModel:
 def __init__(self,kind,seed=42):
  assert kind in ['C0','new']
  self.dtype=torch.float64;self.seed=None if kind=='C0' else seed;self.name=kind if kind=='C0' else f'new_{seed}'
  self.blood,self.t,_=network_flow('seven',tuple(COUNTS),1,.002)
  self.ar=arrays_from_blood(self.blood,self.t);self.T=len(self.t);self.K=KPAD;self.model=None
  if self.seed is not None:
   self.model=Closure(self.ar).cuda().eval()
   state=torch.load(RUN/'seven'/f'train-{seed}'/'model.pt',weights_only=True)
   self.model.load_state_dict(state);self.model.requires_grad_(False)
  cache=RUN/'seven/cache';cache.mkdir(exist_ok=True);p=cache/f'{self.name}.npz'
  if p.exists():data=dict(np.load(p))
  else:data=self.build();np.savez_compressed(p,**data)
  self.data=data
  for k,v in data.items():setattr(self,k,torch.as_tensor(v,device='cuda',dtype=torch.int32 if np.issubdtype(v.dtype,np.integer) else self.dtype).contiguous())
  self.Hflat=self.hbasis.reshape(self.T*KPAD,8).T.contiguous()
  from .models import sample_weights
  x=np.concatenate([d['x'] for d in self.blood]);wx,wt=sample_weights('seven',x,self.t)
  w=np.einsum('pn,qt->pqtn',np.pad(wx,((0,0),(0,KPAD-NS))),wt).reshape(72,-1)
  self.obsw=torch.tensor(w.T,device='cuda',dtype=self.dtype)
  self.obs_hbasis=data['hbasis'].reshape(-1,8).T@w.T
  if self.model is not None:self.model.reaction.double()

 def build(self):
  blood,t=self.blood,self.t;dt=.002;n=COUNTS;g=geometry(n,1);N=g["size"];K=KPAD;T=len(t)
  idx=[i for i,z in enumerate(g['tags']) if z[0]=='axial'];a,b=g['edges'][idx].T;E=len(a)
  dx=np.concatenate([np.full(nv-1,LENGTH[v]/nv) for v,nv in enumerate(n)])
  amp=tau=None
  if self.model is not None:
   with torch.no_grad():amp,tau=[x.cpu().numpy().astype(float) for x in self.model.coefficients()]
  h=np.zeros((N,8));h[:NS,0]=.2*np.exp(-((g['x']/.06-.25)/.1)**2)*(g['cell_v']==0)
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
    old[j,:N]=vo/dt;bcg[j,:N]=bc.sum(1);vr[j,:NS]=vo[:NS]/vol[:NS]
    h=lu.solve(vo[:,None]/dt*h+bc[:,0,None]*np.r_[0,knotbasis[j]][None,:]+rhs_mem)
    if amp is not None:memory=dc[:,:,None]*memory-hs[:,:,None]*am[:,:,None]*(h[b]-h[a])[:,None,:]/dx[:,None,None]
    basis[j,:N]=h
   vo=vol
  el=np.full(K,-1);er=el.copy();el[b]=np.arange(E);er[a]=np.arange(E)
  ea=np.zeros(K,int);eb=ea.copy();ea[:E]=a;eb[:E]=b;dd=np.ones(K);dd[:E]=dx
  le=np.arange(K);ri=le.copy();gd=np.ones(K);off=np.r_[0,np.cumsum(n)]
  for v in range(len(COUNTS)):
   for i in range(off[v],off[v+1]):
    le[i]=max(i-1,off[v]);ri[i]=min(i+1,off[v+1]-1);gd[i]=(ri[i]-le[i])*LENGTH[v]/n[v]
  area=np.pad(self.ar['area'],((0,0),(0,KPAD-NS)));vel=np.pad(self.ar['velocity'],((0,0),(0,KPAD-NS)))
  return dict(hbasis=basis,inv=inv,old=old,bcg=bcg,vr=vr,dec=dec,ha=ha,afad=afad,
   el=el,er=er,ea=ea,eb=eb,dx=dd,left=le,right=ri,gden=gd,area=area,radius=np.sqrt(area/np.pi),vel=vel,volume=np.stack(volumes)[:,:NS])

 @torch.no_grad()
 def batch(self,amps,full=False):
  aa=torch.as_tensor(amps,device='cuda',dtype=self.dtype)
  B=len(aa);H=(torch.cat([torch.ones((B,1),device='cuda',dtype=self.dtype),aa],1)@self.Hflat).reshape(B,self.T,KPAD).contiguous()
  G=torch.empty_like(H);F=torch.empty_like(H);fac=torch.empty_like(H)
  args=(H,self.inv,self.old,self.bcg,self.vr,self.dec,self.ha,self.afad,self.el,self.er,self.ea,self.eb,self.dx)
  integrate[(B,)](*args,fac,G,F,self.T,KPAD,NS,self.seed is not None,False,num_warps=8)
  if self.seed is not None:
   from .reaction_inference import factor
   feat=torch.empty((B,self.T,NS,10),device='cuda',dtype=self.dtype)
   features_kernel[(B,)](H,G,self.radius,self.area,self.vel,self.left,self.right,self.gden,feat,self.T,KPAD,NS,num_warps=8)
   fac.fill_(1);fac[:,:,:NS]=factor(self.model.reaction,feat)
   integrate[(B,)](*args,fac,G,F,self.T,KPAD,NS,True,True,num_warps=8)
  obs=torch.stack([H.reshape(B,-1)@self.obsw,G.reshape(B,-1)@self.obsw],1)
  if full:return torch.stack([H[:,:,:NS],G[:,:,:NS],F[:,:,:NS]],1).permute(0,1,3,2).cpu().numpy()
  return obs.reshape(B,2,3,24).cpu().numpy()

 def predict(self,amps,full=False,batch_size=512):
  return np.concatenate([self.batch(amps[i:i+batch_size],full) for i in range(0,len(amps),batch_size)])
