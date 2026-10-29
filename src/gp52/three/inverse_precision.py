"""Final double-precision inverse solves with locked validation regularization."""
import json,time,subprocess,shutil
import numpy as np
from . import inverse as inv
from .inverse_gpu import GPUModel
from .reference import ROOT,inlets
def main():
 inv.PGTOL=1e-8;inv.MAXITER=100
 base=ROOT/'runs/inverse-three';records={}
 monfile=(base/'gpu-precision.csv').open('w')
 mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=monfile)
 try:
  for window in [.82,1.23]:
   source=base if window<1 else base/'long-window';out=source/'precision';out.mkdir(exist_ok=True)
   for name in ['selection.json','observations.npz']:shutil.copy2(source/name,out/name)
   lam=json.loads((out/'selection.json').read_text())['lam'];d=dict(np.load(out/'observations.npz'));p=inv.problems(d,range(32,48));records[str(window)]={}
   for seed in inv.SEEDS:
    m=GPUModel(seed,observation_end=window,precision='double')
    amps,_=inlets();check=m.predict(amps,True)
    old=np.load(ROOT/f"runs/evaluation-{seed or 42}/{'ordinary_1d' if seed is None else 'neural_combined'}.npz")['y']
    difference=abs(check-old).max((0,2,3));assert difference.max()<5e-7,difference
    path=out/f'test-{m.name}.npz'
    if not path.exists():r=inv.solve(m,p,lam,f'double_{window}');np.savez_compressed(path,**r)
    r=dict(np.load(path));previous=dict(np.load(source/f'test-{m.name}.npz'))
    y=m.predict(r['amplitudes'],True,batch_size=256)
    np.savez_compressed(out/f'prediction-{m.name}.npz',y=y,t=m.t,volume=m.data['volume'],cases=r['cases'],reps=r['reps'],layouts=r['layouts'])
    pg=r['all_pg'][np.arange(len(r['cases'])),r['chosen']]
    records[str(window)][m.name]=dict(max_cpu_difference_by_species=difference.tolist(),selected_pg_max=float(pg.max()),selected_pg_over_1e8=int(sum(pg>1e-8)),selected_pg_over_1e6=int(sum(pg>1e-6)),max_amplitude_change=float(abs(r['amplitudes']-previous['amplitudes']).max()),seconds=float(r['seconds']))
    print('PRECISION_RESULT',window,m.name,records[str(window)][m.name],flush=True)
  (base/'precision-verification.json').write_text(json.dumps(records,indent=2))
 finally:mon.terminate();mon.wait();monfile.close()
if __name__=='__main__':main()
