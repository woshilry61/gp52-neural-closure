"""Select a CUDA inference batch size using frozen weights and random inlets."""
import json,time
import numpy as np
import torch
from .frozen import FrozenModel
from .transfer import RUN
def main():
 m=FrozenModel(42,cache=RUN/'cache-sen');a=np.random.default_rng(2026102902).uniform(.05,1.15,(4096,7))
 records=[];ref=None
 m.predict(a[:32])
 for bs in [1024,1280,1536]:
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
  times=[]
  for repeat in range(3):
   start=time.perf_counter();y=m.predict(a,batch_size=bs);torch.cuda.synchronize();times.append(time.perf_counter()-start)
  seconds=float(np.median(times))
  if ref is None:ref=y
  r=dict(batch_size=bs,seconds=seconds,peak_GiB=torch.cuda.max_memory_allocated()/2**30,max_absolute_difference=float(abs(y-ref).max()));records.append(r);print(json.dumps(r),flush=True)
 best=min(records,key=lambda r:r['seconds'])['batch_size']
 (RUN/'gpu-batch-benchmark-large.json').write_text(json.dumps(dict(selected=best,records=records),indent=2))
if __name__=='__main__':main()
