"""Reproduce the isolated three-vessel benchmark in a fresh checkout."""
from pathlib import Path
import argparse,subprocess,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--oracle-python',type=Path,default=ROOT/'.venv-oracle/bin/python');a=ap.parse_args()
 python=ROOT/'.venv/bin/python'
 def run(*args):subprocess.run([str(x) for x in args],cwd=ROOT,check=True)
 external=ROOT/'external/sen2024'
 if not external.exists():
  external.parent.mkdir(exist_ok=True)
  run('git','clone','https://github.com/ahmetsenemse/PIGNN-1D-Blood-Flow-Simulation.git',external)
 run('git','-C',external,'checkout','bf0ec89f0aab8a639bdd2d363dc3c0cfae770c1d')
 (ROOT/'runs').mkdir(exist_ok=True)
 run(a.oracle_python,ROOT/'scripts/export_sen_three.py')
 run(python,'-m','gp52.three.sen','--epochs','100000','--seeds','32','--output','runs/sen-three-gpu-v2')
 # Refresh per-position blood metrics from all supplied reference locations.
 run(python,ROOT/'scripts/report_sen_three.py')
 ref=[('reference-final',4,24,.001,[]),('reference-extra-fine',8,32,.0005,['--indices','0','32']),
      ('plain-1d-v2',1,1,.002,[]),('plain-1d-fine',4,1,.001,[])]
 for name,n,r,dt,extra in ref:
  run(python,'-m','gp52.three.reference','--output',f'runs/{name}','--refine',str(n),'--radial',str(r),'--dt',str(dt),*extra)
 z=dict(np.load(ROOT/'runs/reference-final/data.npz'))
 for k in ['y','reaction']:z[k]=z[k][...,::2]
 z['flux']=z['flux'][:,::2];z['t']=z['t'][::2]
 np.savez_compressed(ROOT/'runs/reference-final/training-grid.npz',**z)
 jobs=[]
 for seed in [42,43,44]:
  f=(ROOT/f'runs/closure-final-{seed}.log').open('w')
  cmd=[str(python),'-m','gp52.three.closure','--data','runs/reference-final/training-grid.npz','--output',f'runs/closure-final-{seed}','--seed',str(seed),'--steps','3000']
  jobs.append((subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT),f))
 for proc,f in jobs:
  status=proc.wait();f.close()
  if status:raise RuntimeError(f'Closure training exited {status}; see its saved log.')
 for seed in [42,43,44]:
  run(python,'-m','gp52.three.evaluate','--model',f'runs/closure-final-{seed}','--reference','runs/reference-final/data.npz','--output',f'runs/evaluation-{seed}')
 run(python,'-m','gp52.three.checks')
 run(python,'-m','gp52.three.report')
if __name__=='__main__':main()
