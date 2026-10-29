"""Replay every recovered inlet through the unchanged resolved reference solver."""
import argparse,json,hashlib,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from . import reference as ref
from .reference import ROOT
BASE=ROOT/'runs/inverse-three'
METHODS=['ordinary_1d','neural_42','neural_43','neural_44']

def worker(job):
 folder,method,start,stop=job;folder=Path(folder)
 source=folder/f'test-{method}.npz';fit=dict(np.load(source))
 amps=fit['amplitudes'][start:stop].copy()
 output=folder/'replay'/f'{method}-{start:03d}-{stop:03d}'
 if (output/'metrics.json').exists():return str(output)
 # The physics source is unchanged. Substitute only inlet arrays through its
 # existing input function, and record their provenance in the run manifest.
 metadata=dict(input_source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=list(range(start,stop)),
  original_case_ids=fit['cases'][start:stop].tolist(),noise_replicates=fit['reps'][start:stop].tolist(),sensor_counts=fit['layouts'][start:stop].tolist())
 ref.inlets=lambda:(amps,metadata)
 ref.run(output,refine=4,nr=24,dt=.001)
 (output/'input-provenance.json').write_text(json.dumps(metadata,indent=2))
 return str(output)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--window',type=float,choices=[.82,1.23],default=.82);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--precision',action='store_true');ap.add_argument('--tuned',action='store_true');a=ap.parse_args()
 folder=BASE if a.window<1 else BASE/'long-window'
 if a.tuned:folder=folder/'tuned'
 elif a.precision:folder=folder/'precision'
 jobs=[]
 for method in METHODS:
  fit=np.load(folder/f'test-{method}.npz')
  for start in range(0,len(fit['cases']),48):jobs.append((str(folder),method,start,min(start+48,len(fit['cases']))))
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(worker,j) for j in jobs]
  for f in as_completed(futures):print('REPLAY_COMPLETE',f.result(),flush=True)
if __name__=='__main__':main()
