"""Shared observations, optimizer, validation selection and scoring for all three topologies."""
import argparse,json,time,gc
import numpy as np
import torch
from .protocol import RUN,TRAIN,VALID,TEST,plan,inputs,COUNTS
from .models import model,sample_weights
from .reference import flow
from gp52.three import inverse as inv
def reference(name):
 z=dict(np.load(RUN/name/'reference/data.npz'));return dict(y=z['y'][...,::2],volume=z['volume'][::2],t=z['t'][::2],amplitudes=z['amplitudes'])
def observations(name):
 p=RUN/name/'observations.npz'
 if p.exists():return dict(np.load(p))
 # Observe the finer reference time grid, with the same spatial averaging grid as scoring.
 z=np.load(RUN/name/'reference/data.npz')
 blood,t,_=flow(name,tuple(COUNTS[name]),1,.002);x=np.concatenate([d['x'] for d in blood])
 wx,wt=sample_weights(name,x,z['t']);truth=np.einsum('pn,bsnt,qt->bspq',wx,z['y'][:,:2],wt,optimize=True)
 noise=np.random.default_rng(plan()['noise_seed']).normal(0,plan()['noise_std'],(112,3,2,3,24))
 obs=np.repeat(truth[:,None],4,1);obs[:,1:]+=noise
 np.savez_compressed(p,observations=obs,truth=truth,noise=noise,amplitudes=inputs())
 return dict(np.load(p))
def configure():
 cfg=plan()['inverse'];inv.MAXITER=cfg['max_iterations'];inv.PGTOL=cfg['projected_gradient_tolerance']
 def predict(m,a):
  y=m.predict(a,batch_size=128 if isinstance(m,__import__('gp52.unified.models',fromlist=['Single']).Single) else 512 if m.__class__.__module__.endswith('seven_model') else 1280).astype(float)
  y[:,0]=(np.c_[np.ones(len(a)),a]@m.obs_hbasis).reshape(-1,3,24)
  return y.reshape(len(a),-1)
 inv.predict=predict
def errors(pred,truth,volume,t):
 tw=np.r_[(t[1]-t[0])/2,(t[2:]-t[:-2])/2,(t[-1]-t[-2])/2]
 w=volume.T*tw[None]
 return np.sqrt(np.sum((pred-truth)**2*w[None,None],axis=(-2,-1))/np.maximum(np.sum(truth**2*w[None,None],axis=(-2,-1)),1e-30))
def inlet_error(a,b):
 t=np.linspace(0,.82,411);basis=np.stack([np.interp(t,__import__('gp52.unified.protocol',fromlist=['KNOTS']).KNOTS,z) for z in np.eye(7)])
 w=np.ones(len(t));w[[0,-1]]=.5
 return np.sqrt(np.sum(((a-b)@basis)**2*w,1)/np.sum((b@basis)**2*w,1))
def main():
 p=argparse.ArgumentParser();p.add_argument('network',choices=['single','three','seven']);p.add_argument('--kind',choices=['C0','new','old'],required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--stage',choices=['validation','test','forward'],required=True);a=p.parse_args()
 configure();out=RUN/a.network/'evaluation';out.mkdir(exist_ok=True);m=model(a.network,a.kind,a.seed);d=observations(a.network)
 if a.stage=='validation':
  problem=inv.problems(d,VALID);rows=[]
  for lam in plan()['inverse']['lambdas']:
   path=out/f'validation-{m.name}-{lam:g}.npz'
   if path.exists():r=dict(np.load(path))
   else:r=inv.solve(m,problem,lam,'validation');np.savez_compressed(path,**r)
   e=inlet_error(r['amplitudes'],inputs()[r['cases']]);rows.append(dict(lam=lam,error=float(e.mean())))
  best=min(rows,key=lambda r:r['error'])
  (out/f'selection-{m.name}.json').write_text(json.dumps(dict(selected=best,records=rows),indent=2))
 elif a.stage=='forward':
  start=time.perf_counter();y=m.predict(inputs()[TEST],True)
  z=reference(a.network);err=errors(y,z['y'][TEST],z['volume'],z['t'])
  np.savez_compressed(out/f'forward-{m.name}.npz',y=y,case_ids=TEST,errors=err,t=m.t,seconds=time.perf_counter()-start)
  print('FORWARD',m.name,err.mean(0),flush=True)
 else:
  lam=json.loads((out/f'selection-{m.name}.json').read_text())['selected']['lam'];problem=inv.problems(d,TEST)
  path=out/f'test-{m.name}.npz'
  if path.exists():r=dict(np.load(path))
  else:r=inv.solve(m,problem,lam,'test');np.savez_compressed(path,**r)
  y=m.predict(r['amplitudes'],True,batch_size=128);z=reference(a.network);err=errors(y,z['y'][r['cases']],z['volume'],z['t'])
  np.savez_compressed(out/f'prediction-{m.name}.npz',y=y,errors=err,inlet_errors=inlet_error(r['amplitudes'],inputs()[r['cases']]),cases=r['cases'],reps=r['reps'],layouts=r['layouts'],t=m.t)
if __name__=='__main__':main()
