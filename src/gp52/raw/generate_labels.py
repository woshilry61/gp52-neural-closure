"""Generate all 32 CFD chemical trajectories from declared inlet seeds."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,os,subprocess,sys,time
import numpy as np
from ..runtime import ASSETS
def main():
 p=argparse.ArgumentParser();p.add_argument("--flow",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=4);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);plan=json.loads((ASSETS/"splits.json").read_text())
 seeds=plan["train_seeds"]+plan["development_seeds"]+plan["fresh_seeds"]
 knots=np.array([0.,.06,.14,.24,.4,.6,.82]);inputs={}
 for seed in seeds:
  lo,hi=(.08,.9) if seed in plan["train_seeds"] else (0.,1.2)
  values=np.random.default_rng(seed).uniform(lo,hi,7)
  target=a.output/f"inlet-{seed}.npz"
  np.savez_compressed(target,t=knots,boundary=values,unknown_knots=knots,unknown_amplitudes=values);inputs[seed]=target
 start=time.perf_counter()
 def batch(group):
  work=a.output/f"batch-{group[0]}";tick=time.perf_counter()
  with (a.output/f"batch-{group[0]}.log").open("w")as log:
   subprocess.run([sys.executable,"-m","gp52.raw.cfd_chemistry","--flow",str(a.flow),"--output",str(work),"--inlets",*[str(inputs[s])for s in group]],env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"},stdout=log,stderr=subprocess.STDOUT,check=True)
  for seed in group:(work/f"chemical-{seed}").rename(a.output/f"chemical-{seed}")
  return {"seeds":group,"seconds":time.perf_counter()-tick,"metrics":json.loads((work/"batch-metrics.json").read_text())}
 with ThreadPoolExecutor(max_workers=a.workers)as pool:rows=list(pool.map(batch,[seeds[i:i+8]for i in range(0,len(seeds),8)]))
 (a.output/"generation.json").write_text(json.dumps({"seconds":time.perf_counter()-start,"batches":rows},indent=2))
 print(json.dumps({"cases":len(seeds),"seconds":time.perf_counter()-start}))
if __name__=="__main__":main()
