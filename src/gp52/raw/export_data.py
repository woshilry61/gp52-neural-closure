"""Export generated finite-element means, reaction moments and sparse observations."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from .prepare_assets import sample
from ..runtime import ASSETS,tensor
from ..reaction import features
from ..inverse import LOCATIONS,TIMES

def interpolate(value,x0,t0,x,t):
 xx,tt=np.meshgrid(x,t,indexing="ij")
 return RegularGridInterpolator((x0,t0),value,bounds_error=False,fill_value=None)(np.stack([xx,tt],-1))
def main():
 p=argparse.ArgumentParser();p.add_argument("--references",type=Path,required=True);p.add_argument("--assets",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;a.output.mkdir(parents=True,exist_ok=True)
 plan=json.loads((ASSETS/"splits.json").read_text());grid=np.load(a.assets/"flow.npz");x=grid["x"];t=grid["t"]
 flow=dict(np.load(a.references.parent/"cfd/extended/flow.npz"))
 q=tensor(np.stack([np.interp(t,flow["t"],v)for v in flow["face_flow"]]).T,torch.float32)
 af=tensor(np.stack([np.interp(t,flow["t"],v)for v in flow["face_area"]]).T,torch.float32)
 manifest={}
 for split,key in [("train","train_seeds"),("development","development_seeds"),("test","fresh_seeds")]:
  seeds=plan[key];raw=[dict(np.load(a.references/f"chemical-{s}/fields.npz"))for s in seeds]
  y=np.stack([np.stack([interpolate(v,r["x"],r["t"],x,t)for v in r["y"][1:]])for r in raw])
  area=np.stack([interpolate(r["area"],r["x"],r["t"],x,t)for r in raw])
  with torch.no_grad():xf=features(tensor(y,torch.float32),tensor(area[0].T,torch.float32),q,af,float(grid["dx"]),float(grid["dt"])).cpu().numpy()
  values=dict(seeds=np.array(seeds),amps=np.stack([np.load(a.references/f"inlet-{s}.npz")["unknown_amplitudes"]for s in seeds]),
              y=y,area=area,rate_features=xf,exact_observations=np.stack([sample(r["y"][1:3],r["x"],r["t"],LOCATIONS,TIMES)for r in raw]))
  for k in ["reaction_physical","reaction_discrete"]:values[k]=np.stack([interpolate(r[k],r["x"],r["t"],x,t)for r in raw])
  values={k:values[k]for k in ["seeds","amps","y","area","reaction_physical","reaction_discrete","rate_features","exact_observations"]}
  path=a.output/(split+".npz");np.savez_compressed(path,**values)
  manifest[path.name]={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size,"cases":len(seeds)}
 (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 print(json.dumps(manifest))
if __name__=="__main__":main()
