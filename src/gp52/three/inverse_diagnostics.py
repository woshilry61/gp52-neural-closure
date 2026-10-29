"""Numerical and observability audits; never changes the fitted test results."""
import json
import numpy as np
from .inverse import *
def main():
 result={}
 rng=np.random.default_rng(2026102804);aa=rng.uniform(.1,1.1,(2,7))
 for window in [.82,1.23]:
  folder=OUT if window<1 else OUT/'long-window';win={}
  for seed in SEEDS:
   m=GPUModel(seed,observation_end=window);d=data(m) if window<1 else dict(np.load(folder/'observations.npz'))
   f=dict(np.load(folder/f'test-{m.name}.npz'));p=problems(d,range(32,48))
   J=jacobian(m,f['amplitudes'],p['scale'],float(f['lam']),step=.01)
   rr=residual(m,f['amplitudes'],p['obs'],p['scale'],float(f['lam']))
   g=np.einsum('bmi,bm->bi',J,rr);pg=np.max(abs(f['amplitudes']-np.clip(f['amplitudes']-g,0,1.2)),1)
   sens={}
   for n in [1,2,3]:
    A=m.obs_hbasis[1:].T.reshape(3,24,7)[:n].reshape(-1,7)
    sv=np.linalg.svd(A,compute_uv=False)
    sens[n]=dict(singular_values=sv.tolist(),condition_number=float(sv[0]/sv[-1]),last_knot_sensitivity=float(np.linalg.norm(A[:,-1])),first_knot_sensitivity=float(np.linalg.norm(A[:,0])))
   if window>1:
    # Noise-free same-model recovery tests the inversion implementation, not
    # scientific generalization. The resolved-reference benchmark is separate.
    pp=problems(d,range(2));keep=pp['reps']==0;pp={k:v[keep] for k,v in pp.items()}
    truth=aa[pp['cases']];pp['obs']=predict(m,truth)
    fit=solve(m,pp,0,'self_consistency')
    e=inlet_errors(fit['amplitudes'],truth,m.t[m.t<=.82+1e-9])
    assert e.max()<1e-4,(m.name,e)
    recovery=dict(max_inlet_error=float(e.max()),cases=2,layouts=[1,2,3],noise=0,regularization=0)
   else:recovery=None
   win[m.name]=dict(projected_gradient_fd_001_max=float(pg.max()),projected_gradient_fd_001_over_1e6=int(sum(pg>1e-6)),observability=sens,self_consistency=recovery)
  result[str(window)]=win
 (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
