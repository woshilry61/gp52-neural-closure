"""Independent checks and conventional physical-closure comparisons; no coefficient fitting."""
from pathlib import Path
import numpy as np,json,torch
from .reference import ROOT,D,COUNTS,LENGTH
from .closure import arrays,Closure,reaction_features
from .evaluate import solve,metrics
def main():
 ar=arrays();truth=dict(np.load(ROOT/'runs/reference-final/training-grid.npz'));ty=truth['y'];t=truth['t']
 out=ROOT/'runs/physical-baselines';out.mkdir(exist_ok=False);ids=np.arange(32,48);report={}
 amp=np.stack([abs(ar['U'])/np.sqrt(3),np.zeros_like(ar['U'])],-1)
 tau=np.stack([ar['R']**2/(16*D),np.ones_like(ar['R'])],-1)
 variants=[('physical_single_memory',dict(amp=amp,tau=tau)),('quasisteady_taylor',dict(extra_diffusion=ar['U']**2*ar['R']**2/(48*D)))]
 for name,kwargs in variants:
  y,run,vol=solve(**kwargs)
  report[name]=dict(rollout=run,test=metrics(y,ty,vol,t,ids),parameters='No fitted parameters; Taylor-Aris long-time dispersion U^2 R^2/(48D). Single-memory relaxation R^2/(16D), amplitude |U|/sqrt(3). Transient physical probes, not exact unsteady closures.')
  np.savez_compressed(out/f'{name}.npz',y=y,t=t,volume=vol)
 fine=np.load(ROOT/'runs/plain-1d-fine/data.npz')['y'][...,::2]
 report['ordinary_1d_fine']=dict(test=metrics(fine,ty,vol,t,ids),resolution='4x axial, 0.001 s; averaged onto the same output cells')
 off=np.r_[0,np.cumsum(COUNTS)]
 gradient=np.concatenate([np.diff(ty[:,0,off[v]:off[v+1]],axis=1)/(LENGTH[v]/COUNTS[v]) for v in range(3)],1).transpose(2,1,0)
 features=reaction_features(ty,ar,.002)
 source=truth['reaction'][ids].transpose(0,2,1);base=2*ty[ids,0].transpose(0,2,1)*ty[ids,1].transpose(0,2,1)
 mom={}
 for seed in [42,43,44]:
  model=Closure(ar).cuda();model.load_state_dict(torch.load(ROOT/f'runs/closure-final-{seed}/model.pt',weights_only=True))
  with torch.no_grad():
   factor=model.factor(torch.tensor(features[ids],device='cuda',dtype=torch.float32)).cpu().numpy()
   fp=model.flux(torch.tensor(gradient[:,:,ids],device='cuda',dtype=torch.float32),.002).cpu().numpy()
  ft=truth['flux'][ids].transpose(1,2,0)[1:]
  mom[seed]=dict(source_mean_product_relative_l2=float(np.linalg.norm(base-source)/np.linalg.norm(source)),source_neural_relative_l2=float(np.linalg.norm(base*factor-source)/np.linalg.norm(source)),teacher_forced_flux_relative_l2=float(np.linalg.norm(fp-ft)/np.linalg.norm(ft)))
 report['physical_moments']=mom
 # Compare reference convergence and preserve ranking on the prespecified held-out case.
 ex=np.load(ROOT/'runs/reference-extra-fine/data.npz')['y'][...,::4]
 report['reference_refinement']={}
 for j,k in enumerate([0,32]):
  err=np.sqrt((((ty[k]-ex[j])**2)*vol[None]).sum((1,2))/((ex[j]**2)*vol[None]).sum((1,2)))
  report['reference_refinement'][str(k)]={'relative_l2':err.tolist()}
  if k==32:
   for name in ['ordinary_1d','neural_combined']:
    pred=np.load(ROOT/f'runs/evaluation-42/{name}.npz')['y'][k]
    e=np.sqrt((((pred-ex[j])**2)*vol[None]).sum((1,2))/((ex[j]**2)*vol[None]).sum((1,2)))
    report['reference_refinement'][str(k)][name]=e.tolist()
 # This equality check verifies the two ordinary-solver entrypoints implement the same algorithm.
 a=np.load(ROOT/'runs/plain-1d-v2/data.npz')['y'];b=np.load(ROOT/'runs/evaluation-42/ordinary_1d.npz')['y']
 report['ordinary_solver_entrypoint_max_abs']=float(abs(a-b).max())
 assert report['ordinary_solver_entrypoint_max_abs']<1e-12
 (out/'metrics.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
