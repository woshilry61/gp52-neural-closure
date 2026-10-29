"""Equal-budget model-specific validation tuning, with frozen forward weights."""
import json,shutil,subprocess
import numpy as np
from . import inverse as inv
from .inverse_gpu import GPUModel
from .reference import ROOT
def main():
 inv.PGTOL=1e-8;inv.MAXITER=100;base=ROOT/'runs/inverse-three'
 log=(base/'gpu-tuned.csv').open('w');mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=log)
 try:
  for window in [.82,1.23]:
   parent=base if window<1 else base/'long-window';source=parent/'precision';out=parent/'tuned';out.mkdir(exist_ok=True)
   selection=json.loads((parent/'selection.json').read_text())
   choices={}
   for seed in inv.SEEDS:
    name='ordinary_1d' if seed is None else f'neural_{seed}'
    rows=[z for z in selection['records'] if z['method']==name]
    choices[name]=min(rows,key=lambda z:z['mean_inlet_error'])['lam']
   (out/'selection.json').write_text(json.dumps(dict(choices=choices,records=selection['records'],selection='Each model independently minimizes validation inlet error on the identical three-value grid; no test tuning'),indent=2))
   shutil.copy2(source/'observations.npz',out/'observations.npz');d=dict(np.load(out/'observations.npz'));p=inv.problems(d,range(32,48))
   for seed in inv.SEEDS:
    name='ordinary_1d' if seed is None else f'neural_{seed}';lam=choices[name]
    original=np.load(source/f'test-{name}.npz')
    if float(original['lam'])==lam:
     for prefix in ['test','prediction']:shutil.copy2(source/f'{prefix}-{name}.npz',out/f'{prefix}-{name}.npz')
     (out/'replay').mkdir(exist_ok=True)
     for old in (source/'replay').glob(f'{name}-*'):
      link=out/'replay'/old.name
      if not link.exists():link.symlink_to(old.resolve(),target_is_directory=True)
     print('REUSED_IDENTICAL_INPUTS',window,name,lam,flush=True)
     continue
    m=GPUModel(seed,observation_end=window,precision='double');path=out/f'test-{name}.npz'
    if not path.exists():r=inv.solve(m,p,lam,f'tuned_{window}');np.savez_compressed(path,**r)
    r=dict(np.load(path));y=m.predict(r['amplitudes'],True,batch_size=256)
    np.savez_compressed(out/f'prediction-{name}.npz',y=y,t=m.t,volume=m.data['volume'],cases=r['cases'],reps=r['reps'],layouts=r['layouts'])
    print('TUNED_COMPLETE',window,name,lam,flush=True)
 finally:mon.terminate();mon.wait();log.close()
if __name__=='__main__':main()
