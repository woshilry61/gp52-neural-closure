"""Batched CUDA evaluation of the frozen three-vessel forward models.

The time integrator, memory update and predictor/corrector are identical to
evaluate.solve. No model fitting occurs here. Th is affine in inlet amplitudes.
"""
from pathlib import Path
import json,time
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from .reference import ROOT,COUNTS,LENGTH,RATE,flow,geometry,operators,KNOTS
from .closure import Closure,arrays

@triton.jit
def integrate(H,INV,OLD,BC,VOLR,DEC,HA,AFAD,ELEFT,ERIGHT,EA,EB,DX,FACTOR,OUTG,OUTF,
              T:tl.constexpr,K:tl.constexpr,NEURAL:tl.constexpr,WITHFACTOR:tl.constexpr):
 c=tl.program_id(0);i=tl.arange(0,K)
 gg=tl.full((K,),1.,INV.dtype.element_ty);fn=tl.full((K,),0.,INV.dtype.element_ty)
 m0=tl.full((K,),0.,INV.dtype.element_ty);m1=tl.full((K,),0.,INV.dtype.element_ty)
 el=tl.load(ELEFT+i);er=tl.load(ERIGHT+i)
 ea=tl.load(EA+i);eb=tl.load(EB+i);dx=tl.load(DX+i)
 tl.store(OUTG+c*T*K+i,gg);tl.store(OUTF+c*T*K+i,fn)
 for j in range(1,T):
  hh=tl.load(H+c*T*K+j*K+i)
  rhs=tl.load(OLD+j*K+i)*gg+tl.load(BC+j*K+i)
  if NEURAL:
   d0=tl.load(DEC+j*K*2+i*2);d1=tl.load(DEC+j*K*2+i*2+1)
   mf=tl.load(AFAD+j*K*2+i*2)*m0+tl.load(AFAD+j*K*2+i*2+1)*m1
   rhs+=tl.where(el>=0,tl.gather(mf,tl.maximum(el,0),axis=0),0.)
   rhs-=tl.where(er>=0,tl.gather(mf,tl.maximum(er,0),axis=0),0.)
  mat=tl.load(INV+j*K*K+i[:,None]*K+i[None,:])
  gt=tl.sum(mat*rhs[None,:],1)
  if NEURAL:
   grad=(tl.gather(gt,eb,axis=0)-tl.gather(gt,ea,axis=0))/dx
   m0=d0*m0-tl.load(HA+j*K*2+i*2)*grad
   m1=d1*m1-tl.load(HA+j*K*2+i*2+1)*grad
  fac=tl.load(FACTOR+c*T*K+j*K+i) if WITHFACTOR else 1.
  if INV.dtype.element_ty == tl.float64:
   gg=gt*libdevice.exp(-.004*hh*fac)
  else:
   gg=gt*tl.exp(-.004*hh*fac)
  fn=tl.where(i<57,tl.load(VOLR+j*K+i)*fn+gt-gg,0.)
  tl.store(OUTG+c*T*K+j*K+i,gg);tl.store(OUTF+c*T*K+j*K+i,fn)

@triton.jit
def features_kernel(H,G,R,AR,UV,LEFT,RIGHT,GDEN,OUT,T:tl.constexpr,K:tl.constexpr):
 c=tl.program_id(0);i=tl.arange(0,K);l=tl.load(LEFT+i);r=tl.load(RIGHT+i);denx=tl.load(GDEN+i)
 h02=tl.full((K,),0.,H.dtype.element_ty);g02=h02;h10=h02;g10=h02
 for j in range(T):
  h=tl.load(H+c*T*K+j*K+i);g=tl.load(G+c*T*K+j*K+i)
  rr=tl.load(R+j*K+i)
  gh=(tl.gather(h,r,0)-tl.gather(h,l,0))/denx*rr
  gg=(tl.gather(g,r,0)-tl.gather(g,l,0))/denx*rr
  h02=tl.where(j==0,gh,.9048374180359595*h02+.09516258196404043*gh)
  g02=tl.where(j==0,gg,.9048374180359595*g02+.09516258196404043*gg)
  h10=tl.where(j==0,gh,.9801986733067553*h10+.019801326693244747*gh)
  g10=tl.where(j==0,gg,.9801986733067553*g10+.019801326693244747*gg)
  p=(c*T*57+j*57+i)*10;mask=i<57;hg=h*g;den=hg+1e-4
  tl.store(OUT+p,h,mask);tl.store(OUT+p+1,g,mask);tl.store(OUT+p+2,hg,mask)
  tl.store(OUT+p+3,gh,mask);tl.store(OUT+p+4,gg,mask)
  tl.store(OUT+p+5,tl.load(AR+j*K+i)/.001-1,mask)
  tl.store(OUT+p+6,tl.load(UV+j*K+i)/.2,mask)
  x=gh*gg/den;tl.store(OUT+p+7,2/(1+tl.exp(-2*x))-1,mask)
  x=h02*g02/den;tl.store(OUT+p+8,2/(1+tl.exp(-2*x))-1,mask)
  x=h10*g10/den;tl.store(OUT+p+9,2/(1+tl.exp(-2*x))-1,mask)

class GPUModel:
 def __init__(self,seed=None,cache=ROOT/'runs/inverse-three/cache',observation_end=.82,precision='single'):
  self.dtype=torch.float64 if precision=='double' else torch.float32
  self.seed=seed;self.name='ordinary_1d' if seed is None else f'neural_{seed}'
  self.ar=arrays();self.t=self.ar['t'];self.T=len(self.t);self.K=64
  self.model=None
  if seed is not None:
   self.model=Closure(self.ar).cuda().eval()
   self.model.load_state_dict(torch.load(ROOT/f'runs/closure-final-{seed}/model.pt',weights_only=True))
   for p in self.model.parameters():p.requires_grad_(False)
  cache.mkdir(parents=True,exist_ok=True);p=cache/f'{self.name}.npz'
  if p.exists():data=dict(np.load(p))
  else:data=self.build();np.savez_compressed(p,**data)
  self.data=data
  for k,v in data.items():
   setattr(self,k,torch.as_tensor(v,device='cuda',dtype=torch.int32 if np.issubdtype(v.dtype,np.integer) else self.dtype).contiguous())
  self.Hflat=self.hbasis.reshape(self.T*64,8).T.contiguous()
  self.sensors=np.array([.03,.01,.017]);self.sample_t=np.linspace(.002,observation_end,24)
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
  blood,t=flow();dt=.002;n=COUNTS;g=geometry(n,1);N=58;K=64;T=len(t)
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
