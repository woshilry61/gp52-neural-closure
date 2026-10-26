"""Generate the declared sparse blood-flow observations from the new CFD solve."""
from pathlib import Path
import argparse,json
import numpy as np
from .numerics import map_field
def main():
 p=argparse.ArgumentParser();p.add_argument("--flow",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 r=dict(np.load(a.flow));a.output.mkdir(parents=True,exist_ok=True)
 x=np.array([.006,.03,.054]);t=np.r_[np.linspace(.002,.82,24),np.linspace(.84,1.23,24)]
 u=map_field(r["face_velocity"],r["xf"],r["t"],x,t);ar=map_field(r["face_area"],r["xf"],r["t"],x,t)
 us=np.sqrt(np.mean(u*u));asc=.05*np.mean(ar);arrays={};meta=[]
 for i in range(4):
  name="clean" if i==0 else f"noise{i-1}";sigma=0. if i==0 else .02;rng=np.random.default_rng(993000+i)
  arrays[name+"_x"]=x;arrays[name+"_t"]=t;arrays[name+"_u"]=u+sigma*us*rng.normal(size=u.shape);arrays[name+"_area"]=ar+sigma*asc*rng.normal(size=ar.shape)
  meta.append(dict(name=name,area_observed=True,u_scale=us,area_scale=asc,duration=1.23,noise=sigma))
 np.savez_compressed(a.output/"blood-observations.npz",**arrays)
 (a.output/"blood-observations.json").write_text(json.dumps(meta,indent=2))
if __name__=="__main__":main()
