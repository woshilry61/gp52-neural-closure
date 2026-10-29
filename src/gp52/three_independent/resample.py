"""Resample saved finite elements without rerunning or altering blood dynamics."""
import argparse,time,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from . import blood as b
from gp52.three.reference import ROOT,COUNTS
def vessel(args):
 path,k,nr=args;raw=np.load(path/f'finite-elements-{k}.npz');src=np.load(ROOT/'runs/sen-input/input.npz')
 z=raw['wall_x'];base=b.MeshTri(raw['reference_points'],raw['triangles'],sort_t=False)
 base=base.with_boundaries({'inlet':lambda x:np.isclose(x[0],0),'outlet':lambda x:np.isclose(x[0],z[-1]),'axis':lambda x:np.isclose(x[1],0),'wall':lambda x:np.isclose(x[1],1)})
 u=raw['velocity'];pp=raw['pressure'];rr=raw['wall_radius'];out=[];start=time.perf_counter()
 for j in range(len(raw['t'])):
  mesh=b.f.mesh_at(base,z,rr[j]);v,p=b.f.bases(mesh)
  s=dict(v=v,pbasis=p,u=u[j],p=pp[j],z=z,rad=rr[j])
  out.append(b.sample(s,k,int(COUNTS[k]*4),nr))
  if j%200==0:print(json.dumps(dict(vessel=k,step=j,seconds=time.perf_counter()-start)),flush=True)
 d={f'v{k}_{key}':out[0][key] if key in ['x','xf'] else np.stack([r[key] for r in out]) for key in out[0]}
 return d
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--blood',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--radial',type=int,default=96);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 start=time.perf_counter()
 with ProcessPoolExecutor(max_workers=3) as pool:
  result=list(pool.map(vessel,[(a.blood,k,a.radial) for k in range(3)]))
 d={key:value for zz in result for key,value in zz.items()};d['t']=np.load(a.blood/'flow.npz')['t']
 np.savez_compressed(a.output/'flow.npz',**d)
 (a.output/'metrics.json').write_text(json.dumps(dict(source=str(a.blood),radial_bins=a.radial,operation='Quadrature resampling of the identical finite-element states; no new blood solve',seconds=time.perf_counter()-start),indent=2))
if __name__=='__main__':main()
