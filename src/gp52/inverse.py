"""Shared bounded seven-amplitude inverse with three initializations."""
import time,json
import numpy as np
import torch
from scipy.optimize import lsq_linear,least_squares
from .runtime import ASSETS

LOCATIONS=np.array([.006,.03,.054])
TIMES=np.linspace(.002,.82,24)
LAYOUTS={"one":[1],"two":[0,1],"three":[0,1,2]}
def time_weights(t):
 w=np.zeros_like(t);w[:-1]+=np.diff(t)/2;w[1:]+=np.diff(t)/2;return w
def relative(a,b,w=None):
 if w is None:return float(np.linalg.norm(a-b)/np.linalg.norm(b))
 return float(np.sqrt(np.sum(w*(a-b)**2)/np.sum(w*b*b)))
def second_difference(knots):
 h=np.diff(knots);R=np.zeros((5,7));scale=(.82/6)**2/1.2
 for i in range(5):
  R[i,i:i+3]=[2/(h[i]*(h[i]+h[i+1])),-2/(h[i]*h[i+1]),2/(h[i+1]*(h[i]+h[i+1]))]
 return R*scale
class Context:
 def __init__(self):
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
  with np.load(ASSETS/"inverse-context.npz") as z:
   for k in z.files:setattr(self,k,z[k].copy())
 def initial_guess(self,obs,scale):
  out=[]
  for y,w in zip(obs,scale):
   wh=w[:72];nz=wh>0
   out.append(lsq_linear(self.hb[nz]*wh[nz,None],(y[:72]-self.h0)[nz]*wh[nz],bounds=(0,1.2),tol=1e-11).x)
  return np.stack(out)
 def scaling(self,layout,noise,rms,weighted=True):
  mask=np.zeros((2,3,24));mask[:,LAYOUTS[layout]]=1.
  sigma=np.sqrt(rms[:,:,None]**2+noise**2) if weighted else np.ones((2,3,1))
  return (mask/sigma/np.sqrt(mask.sum())).ravel()

class Adapter:
    def __init__(self,ctx,model,theta,name):
        self.ctx=ctx;self.m=model;self.theta=np.asarray(theta);self.name=name
        self.wx=model.tensor(ctx.wx);self.wt=model.tensor(ctx.wt)
    def fields(self,a,flow=0):
        chunks=[]
        for j in range(0,len(a),128):
            y,_=self.m.solve(a[j:j+128],self.theta,np.full(len(a[j:j+128]),flow))
            chunks.append(y.cpu().numpy())
        return np.concatenate(chunks)
    def value(self,a):
        chunks=[]
        for j in range(0,len(a),256):
            y,_=self.m.solve(a[j:j+256],self.theta)
            chunks.append(torch.einsum("on,bsnt,kt->bsok",self.wx,y[:,:2],self.wt).reshape(len(y),-1).cpu().numpy())
        return np.concatenate(chunks)
    def value_jac(self,a):
        a=np.atleast_2d(a);eps=1e-5
        grid=np.concatenate([a[:,None],a[:,None]+eps*np.eye(7)[None],a[:,None]-eps*np.eye(7)[None]],axis=1)
        y=self.value(grid.reshape(-1,7)).reshape(len(a),15,-1)
        return y[:,0],((y[:,1:8]-y[:,8:])/(2*eps)).transpose(0,2,1)
def residual_jac(adapter,a,obs,scale,lam,R,jac=True):
 if jac:
  y,J=adapter.value_jac(a);J=J*scale[:,:,None]
 else:y=adapter.value(a)
 r=(y-obs)*scale;factor=np.sqrt(lam/5)
 r=np.concatenate([r,(a@R.T)*factor[:,None]],axis=1)
 if not jac:return r
 return r,np.concatenate([J,factor[:,None,None]*R[None]],axis=1)
def solve_one_start(adapter,obs,scale,lam,initial,R,maxiter=100,gradient_tolerance=1e-6,fallback=True):
 a=np.array(initial,copy=True);B=len(a);damp=np.full(B,1e-4)
 done=np.zeros(B,bool);iters=np.zeros(B,int);reason=np.full(B,"iteration_limit",dtype="U32")
 for it in range(maxiter):
  ix=np.flatnonzero(~done)
  if len(ix)==0:break
  r,J=residual_jac(adapter,a[ix],obs[ix],scale[ix],lam[ix],R)
  g=np.einsum("bok,bo->bk",J,r);H=J.transpose(0,2,1)@J
  pg=np.max(abs(a[ix]-np.clip(a[ix]-g,0,1.2)),axis=1)
  converged=pg<=gradient_tolerance
  done[ix[converged]]=True;reason[ix[converged]]="projected_gradient"
  iters[ix]=it+1
  keep=~converged
  if not np.any(keep):continue
  ix=ix[keep];r=r[keep];g=g[keep];H=H[keep]
  bound=((a[ix]<=1e-10)&(g>0))|((a[ix]>=1.2-1e-10)&(g<0))
  free=(~bound).astype(float);diag=np.maximum(np.diagonal(H,axis1=1,axis2=2),1e-10)
  matrix=H+np.eye(7)[None]*(damp[ix,None]*diag)[:,None,:]
  matrix=matrix*free[:,:,None]*free[:,None,:]+np.eye(7)[None]*bound[:,None,:]
  step=np.linalg.solve(matrix,(-g*free)[...,None])[...,0]
  alpha=np.array([1.,.5,.25,.125,.03125,0.])
  candidates=np.clip(a[ix,None,:]+alpha[None,:,None]*step[:,None,:],0,1.2)
  n=len(alpha)
  cr=residual_jac(adapter,candidates.reshape(-1,7),np.repeat(obs[ix],n,axis=0),np.repeat(scale[ix],n,axis=0),np.repeat(lam[ix],n),R,jac=False).reshape(len(ix),n,-1)
  costs=.5*np.sum(cr**2,axis=2);best=np.argmin(costs,axis=1);new=candidates[np.arange(len(ix)),best]
  oldcost=.5*np.sum(r**2,axis=1);newcost=costs[np.arange(len(ix)),best]
  improve=newcost<oldcost
  relchange=abs(oldcost-newcost)/np.maximum(oldcost,1e-16)
  smallstep=np.max(abs(new-a[ix]),axis=1)<=1e-7
  # Small objective change is recorded, but gradient convergence remains the
  # primary criterion. Stagnated cases are polished by identical SciPy TRF.
  stalled=smallstep&(relchange<=1e-9)
  a[ix]=new;damp[ix]=np.clip(np.where(improve,damp[ix]*.3,damp[ix]*10),1e-10,1e8)
  done[ix[stalled]]=True;reason[ix[stalled]]="stagnation"
 # Identical bounded TRF fallback for every method if the gradient target has
 # not been met. Counts and wall time remain in the end-to-end cost.
 r,J=residual_jac(adapter,a,obs,scale,lam,R)
 g=np.einsum("bok,bo->bk",J,r);pg=np.max(abs(a-np.clip(a-g,0,1.2)),axis=1)
 fallbacks=[];trf_nfev=np.zeros(B,int)
 for k in (np.flatnonzero(pg>gradient_tolerance)if fallback else []):
  cached_x=None;cached_r=None;cached_j=None
  def calculate(v):
   nonlocal cached_x,cached_r,cached_j
   if cached_x is None or not np.array_equal(v,cached_x):
    rr,jj=residual_jac(adapter,v[None],obs[k:k+1],scale[k:k+1],lam[k:k+1],R)
    cached_x=v.copy();cached_r=rr[0];cached_j=jj[0]
   return cached_r,cached_j
  fit=least_squares(lambda v:calculate(v)[0],a[k],jac=lambda v:calculate(v)[1],bounds=(0,1.2),ftol=1e-9,xtol=1e-10,gtol=1e-8,max_nfev=100)
  a[k]=fit.x;reason[k]="trf_"+str(fit.status);trf_nfev[k]=fit.nfev;fallbacks.append(int(k))
 r,J=residual_jac(adapter,a,obs,scale,lam,R)
 g=np.einsum("bok,bo->bk",J,r);pg=np.max(abs(a-np.clip(a-g,0,1.2)),axis=1)
 return a,dict(cost=(.5*np.sum(r*r,axis=1)).tolist(),projected_gradient=pg.tolist(),converged=(pg<=gradient_tolerance).tolist(),iterations=iters.tolist(),reason=reason.tolist(),fallback_indices=fallbacks,trf_nfev=trf_nfev.tolist())
def inverse(adapter,obs,scale,lam,ctx,maxiter=100):
 torch.cuda.synchronize();start=time.perf_counter()
 initial=ctx.initial_guess(obs,scale);init_seconds=time.perf_counter()-start
 starts=np.stack([initial,np.full_like(initial,.3),np.full_like(initial,.9)],axis=1)
 a,detail=solve_one_start(adapter,np.repeat(obs,3,0),np.repeat(scale,3,0),np.repeat(lam,3),starts.reshape(-1,7),ctx.R,maxiter=maxiter)
 cost=np.array(detail["cost"]).reshape(-1,3);which=cost.argmin(axis=1);indices=np.arange(len(obs))*3+which
 selected=a[indices]
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 return selected,dict(seconds=elapsed,initialization_seconds=init_seconds,selected_start=which.tolist(),selected_cost=cost[np.arange(len(obs)),which].tolist(),selected_projected_gradient=np.array(detail["projected_gradient"])[indices].tolist(),selected_converged=np.array(detail["converged"])[indices].tolist(),all_starts=detail)
