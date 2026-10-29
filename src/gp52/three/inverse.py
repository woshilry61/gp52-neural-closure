"""Fair, frozen-model sparse inverse benchmark. All amplitudes fit Th/Fg only."""
import argparse,json,time,subprocess,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from .inverse_gpu import GPUModel
from .reference import ROOT,KNOTS,inlets
OUT=ROOT/'runs/inverse-three'
SEEDS=[None,42,43,44]
LAMBDAS=[0.,.0001,.01]
WINDOW=.82
NOISE=.02
FD=.002
MAXITER=80
PGTOL=1e-6

def regularizer():
 h=np.diff(KNOTS);R=np.zeros((5,7));scale=(.82/6)**2/1.2
 for i in range(5):R[i,i:i+3]=np.array([2/(h[i]*(h[i]+h[i+1])),-2/(h[i]*h[i+1]),2/(h[i+1]*(h[i]+h[i+1]))])*scale
 return R
R=regularizer()

def predict(m,a):
 y=m.predict(a,batch_size=512).astype(float)
 # Use the exact affine Th observation operator; GPU float32 is only needed
 # for nonlinear Fg. This reduces finite-difference roundoff, not model error.
 y[:,0]=(np.c_[np.ones(len(a)),a]@m.obs_hbasis).reshape(-1,3,24)
 return y.reshape(len(a),-1)

def data(m):
 p=OUT/'observations.npz'
 if p.exists():return dict(np.load(p))
 raw=np.load(ROOT/'runs/reference-final/training-grid.npz')
 z=np.pad(raw['y'][:,:2].transpose(0,1,3,2),((0,0),(0,0),(0,0),(0,7)))
 truth=(z.reshape(96,-1)@m.obsw.cpu().numpy()).reshape(48,2,3,24)
 rng=np.random.default_rng(2026102802)
 obs=np.repeat(truth[:,None],4,axis=1)
 obs[:,1:]+=rng.normal(0,NOISE,obs[:,1:].shape)
 np.savez_compressed(p,observations=obs,truth=truth,amplitudes=raw['amplitudes'],sample_t=m.sample_t,sensors=m.sensors)
 return dict(np.load(p))

def problems(d,ids):
 cases=[];reps=[];layouts=[]
 for c in ids:
  for rep in range(4):
   for layout in [1,2,3]:cases.append(c);reps.append(rep);layouts.append(layout)
 cases=np.array(cases);reps=np.array(reps);layouts=np.array(layouts)
 y=d['observations'][cases,reps].reshape(-1,144)
 mask=np.zeros((len(cases),2,3,24))
 for i,n in enumerate(layouts):mask[i,:,:n]=1/np.sqrt(n*48)
 return dict(cases=cases,reps=reps,layouts=layouts,obs=y,scale=mask.reshape(-1,144))

def residual(m,a,obs,scale,lam):
 y=predict(m,a)
 return np.c_[(y-obs)*scale,np.sqrt(lam/5)*(a@R.T)]

def jacobian(m,a,scale,lam,step=FD):
 n=len(a);j=[]
 for sign in [-1,1]:
  pert=a[:,None,:]+sign*step*np.eye(7)[None,:,:]
  j.append(predict(m,pert.reshape(-1,7)).reshape(n,7,144))
 J=((j[1]-j[0])/(2*step)).transpose(0,2,1)*scale[:,:,None]
 J[:,:72]=m.obs_hbasis[1:].T[None]*scale[:,:72,None]
 return np.concatenate([J,np.broadcast_to(np.sqrt(lam/5)*R,(n,5,7))],axis=1)

def solve(m,p,lam,label):
 n=len(p['cases'])
 lin=[]
 for i in range(n):
  weights=p['scale'][i,:72];A=m.obs_hbasis[1:].T*weights[:,None]
  b=(p['obs'][i,:72]-m.obs_hbasis[0])*weights
  A=np.r_[A,np.sqrt(lam/5)*R];b=np.r_[b,np.zeros(5)]
  lin.append(lsq_linear(A,b,bounds=(0,1.2),tol=1e-12).x)
 init=np.stack([lin,np.full((n,7),.3),np.full((n,7),.9)],1)
 a=init.reshape(-1,7).copy();obs=np.repeat(p['obs'],3,0);scale=np.repeat(p['scale'],3,0)
 N=len(a);done=np.zeros(N,bool);iterations=np.zeros(N,int);damp=np.full(N,1e-4)
 reason=np.full(N,'iteration_limit',dtype='<U32');lastpg=np.full(N,np.nan);start=time.perf_counter()
 rr=residual(m,a,obs,scale,lam);cost=.5*np.sum(rr*rr,1)
 for it in range(MAXITER):
  idx=np.flatnonzero(~done)
  if not len(idx):break
  J=jacobian(m,a[idx],scale[idx],lam)
  g=np.einsum('bmi,bm->bi',J,rr[idx]);H=np.einsum('bmi,bmj->bij',J,J)
  pg=np.max(abs(a[idx]-np.clip(a[idx]-g,0,1.2)),1);lastpg[idx]=pg;iterations[idx]+=1
  conv=pg<PGTOL;done[idx[conv]]=True;reason[idx[conv]]='projected_gradient'
  free=~(((a[idx]<=1e-10)&(g>0))|((a[idx]>=1.2-1e-10)&(g<0)))
  diag=np.maximum(np.diagonal(H,axis1=1,axis2=2),1e-10)
  HH=H*free[:,:,None]*free[:,None,:]+np.eye(7)[None]*(damp[idx,None]*diag+~free)[:,None,:]
  step=np.linalg.solve(HH,-g*free)
  alphas=np.array([1,.5,.25,.125,.03125,0.])
  cand=np.clip(a[idx,None]+alphas[None,:,None]*step[:,None],0,1.2)
  cr=residual(m,cand.reshape(-1,7),np.repeat(obs[idx],6,0),np.repeat(scale[idx],6,0),lam).reshape(len(idx),6,-1)
  cc=.5*np.sum(cr*cr,2);best=cc.argmin(1);new=cand[np.arange(len(idx)),best];newcost=cc[np.arange(len(idx)),best]
  improved=newcost<cost[idx]-1e-14
  damp[idx]=np.clip(damp[idx]*np.where(improved,.3,10.),1e-10,1e10)
  # Do not label a tiny step "converged" if the projected gradient is large.
  stuck=(damp[idx]>=1e9)&~conv;done[idx[stuck]]=True;reason[idx[stuck]]='roundoff_or_stagnation'
  a[idx]=new;cost[idx]=newcost;rr[idx]=cr[np.arange(len(idx)),best]
  if it%10==0:
   print(json.dumps(dict(phase=label,method=m.name,lam=lam,iteration=it,active=len(idx),median_cost=float(np.median(cost)),max_pg=float(pg.max()),seconds=time.perf_counter()-start)),flush=True)
 # Record gradients at final points and all starts, not only selected fits.
 J=jacobian(m,a,scale,lam);g=np.einsum('bmi,bm->bi',J,rr)
 lastpg=np.max(abs(a-np.clip(a-g,0,1.2)),1)
 reason[lastpg<PGTOL]='projected_gradient'
 costs=cost.reshape(n,3);chosen=costs.argmin(1);aa=a.reshape(n,3,7)[np.arange(n),chosen]
 return dict(amplitudes=aa,cases=p['cases'],reps=p['reps'],layouts=p['layouts'],cost=costs[np.arange(n),chosen],
  all_amplitudes=a.reshape(n,3,7),all_cost=costs,all_pg=lastpg.reshape(n,3),all_reason=reason.reshape(n,3),
  all_iterations=iterations.reshape(n,3),chosen=chosen,initial=init,lam=lam,seconds=time.perf_counter()-start)

def inlet_errors(a,truth,t):
 basis=np.stack([np.interp(t,KNOTS,z) for z in np.eye(7)])
 y=truth@basis;delta=(a-truth)@basis
 return np.sqrt(np.sum(delta*delta,1)/np.sum(y*y,1))

def validate_acceleration():
 import torch
 from . import evaluate
 from .closure import reaction_features
 amps,_=inlets();records={}
 rng=np.random.default_rng(2026102803);randomamps=rng.uniform(0,1.2,(4,7))
 original_inlets=evaluate.inlets
 for seed in SEEDS:
  m=GPUModel(seed,observation_end=WINDOW);y=m.predict(amps,True)
  expected=np.load(ROOT/f"runs/evaluation-{seed or 42}/{'ordinary_1d' if seed is None else 'neural_combined'}.npz")['y']
  err=abs(y-expected).max((0,2,3))
  evaluate.inlets=lambda:(randomamps,{})
  if seed is None:reference=evaluate.solve()[0]
  else:
   with torch.no_grad():am,ta=[z.cpu().numpy() for z in m.model.coefficients()]
   predictor=evaluate.solve(am,ta)[0]
   feat=reaction_features(predictor,m.ar,.002)
   with torch.no_grad():fac=m.model.factor(torch.tensor(feat,device='cuda',dtype=torch.float32)).cpu().numpy()
   reference=evaluate.solve(am,ta,fac)[0]
  yr=m.predict(randomamps,True);errrandom=abs(yr-reference).max((0,2,3))
  assert max(err.max(),errrandom.max())<3e-5,(m.name,err,errrandom)
  scale=np.ones((4,144))/12
  j1=jacobian(m,randomamps,scale,0,step=.002);j2=jacobian(m,randomamps,scale,0,step=.001)
  jerr=np.linalg.norm(j1-j2)/np.linalg.norm(j1)
  assert jerr<.002,jerr
  records[m.name]=dict(known_max_absolute_by_species=err.tolist(),random_max_absolute_by_species=errrandom.tolist(),finite_difference_relative_change=jerr)
  print('VERIFIED',m.name,records[m.name],flush=True)
 evaluate.inlets=original_inlets
 (OUT/'gpu-verification.json').write_text(json.dumps(records,indent=2))
 return records

def main():
 global OUT,WINDOW
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['verify','validation','test','all'],default='all');ap.add_argument('--window',type=float,choices=[.82,1.23],default=.82);a=ap.parse_args()
 WINDOW=a.window
 if WINDOW>1:OUT=OUT/'long-window'
 OUT.mkdir(exist_ok=True);log=(OUT/f'gpu-{a.stage}.csv').open('w')
 mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=log)
 try:
  if a.stage in ['all','verify']:validate_acceleration()
  if a.stage in ['all','validation']:
   assert (ROOT/'runs/inverse-three/gpu-verification.json').exists()
   records=[]
   for seed in SEEDS:
    m=GPUModel(seed,observation_end=WINDOW);d=data(m);p=problems(d,range(24,32))
    for lam in LAMBDAS:
     path=OUT/f'validation-{m.name}-lambda{lam:g}.npz'
     if not path.exists():r=solve(m,p,lam,'validation');np.savez_compressed(path,**r)
     r=dict(np.load(path));e=inlet_errors(r['amplitudes'],d['amplitudes'][r['cases']],m.t[m.t<=.82+1e-9])
     records.append(dict(method=m.name,lam=lam,mean_inlet_error=float(e.mean())))
   # Give ordinary and neural families equal weight; average neural seeds first.
   scores={}
   for lam in LAMBDAS:
    vals=[z for z in records if z['lam']==lam]
    ordinary=[z['mean_inlet_error'] for z in vals if z['method']=='ordinary_1d'][0]
    neural=np.mean([z['mean_inlet_error'] for z in vals if z['method']!='ordinary_1d'])
    scores[lam]=float((ordinary+neural)/2)
   selected=min(scores,key=scores.get)
   (OUT/'selection.json').write_text(json.dumps(dict(lam=selected,scores=scores,records=records,selection='Validation only, all layouts, clean plus three noise replicates, equal method-family weighting'),indent=2))
   print('SELECTED',selected,scores,flush=True)
  if a.stage in ['all','test']:
   selected=json.loads((OUT/'selection.json').read_text())['lam']
   for seed in SEEDS:
    m=GPUModel(seed,observation_end=WINDOW);d=data(m);p=problems(d,range(32,48));path=OUT/f'test-{m.name}.npz'
    if not path.exists():r=solve(m,p,selected,'test');np.savez_compressed(path,**r)
    r=dict(np.load(path));y=m.predict(r['amplitudes'],True,batch_size=256)
    np.savez_compressed(OUT/f'prediction-{m.name}.npz',y=y,t=m.t,volume=m.data['volume'],cases=r['cases'],reps=r['reps'],layouts=r['layouts'])
 finally:mon.terminate();mon.wait();log.close()
if __name__=='__main__':main()
