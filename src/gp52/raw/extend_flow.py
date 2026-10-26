"""Continue independent CFD in time; preserve every original-window value."""
from pathlib import Path
import argparse,sys,time,json,shutil,numpy as np
from scipy.interpolate import PchipInterpolator
from . import cfd_flow as cfd
from skfem import MeshTri
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--source",type=Path,required=True);ap.add_argument("--waveform",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--duration",type=float,default=1.23);a=ap.parse_args()
 a.output.mkdir(exist_ok=False);shutil.copy2(__file__,a.output/"executed-source.py")
 old=dict(np.load(a.source/"finite-elements.npz"));flow=dict(np.load(a.source/"flow.npz"));meta=json.loads((a.source/"metrics.json").read_text())
 base=MeshTri(old["reference_points"],old["triangles"],sort_t=False)
 base=base.with_boundaries({"inlet":lambda x:np.isclose(x[0],0),"outlet":lambda x:np.isclose(x[0],.06/cfd.L0),"axis":lambda x:np.isclose(x[1],0),"wall":lambda x:np.isclose(x[1],1)})
 z=old["wall_x"];wn=np.where(np.isclose(base.p[1],1))[0];wn=wn[np.argsort(base.p[0,wn])]
 rad=old["wall_radius"][:,-1].copy();u=old["velocity"][:,-1].copy();pressure=old["pressure"][:,-1].copy()
 source=dict(np.load(a.waveform));tt=source["time_s"]-source["time_s"][0];period=float(tt[-1])
 vel=meta["velocity_scale"]*source["velocity_m_s"]+meta["harmonic"]*abs(source["velocity_m_s"]).max()*np.sin(4*np.pi*tt/period)[None,:]
 fu=PchipInterpolator(tt,vel[0]/cfd.U0);fo=PchipInterpolator(tt,vel[-1]/cfd.U0)
 dt=float(flow["dt"]);times=np.arange(round(flow["t"][-1]/dt)+1,round(a.duration/dt)+1)*dt
 records=[];rawu=[];rawp=[];rawr=[];hist=[];start=time.perf_counter()
 cfd.CONVECTION=float(meta["convection"]);cfd.STABILIZATION=meta["streamline_stabilization"];cfd.RELAXATION=meta["newton_relaxation"]
 for j,t in enumerate(times):
  rad,u,pressure,info,v,p=cfd.step(base,z,rad,u,u,dt/cfd.T0,float(fu(t%period)),float(fo(t%period)),wn)
  records.append(cfd.sample(v,p,u,pressure,z,rad));rawu.append(u.copy());rawp.append(pressure.copy());rawr.append(rad.copy());hist.append(info)
  if j%50==49:print(json.dumps(dict(t=t,seconds=time.perf_counter()-start,**info)),flush=True)
 fields={k:np.concatenate([v,np.stack([rec[k]for rec in records],axis=-1)],axis=-1)if k not in ["x","xf","t","dt","dx"]else v for k,v in flow.items()}
 fields["t"]=np.r_[flow["t"],times]
 np.savez_compressed(a.output/"flow.npz",**fields)
 np.savez_compressed(a.output/"finite-elements.npz",velocity=np.concatenate([old["velocity"],np.stack(rawu,1)],1),pressure=np.concatenate([old["pressure"],np.stack(rawp,1)],1),wall_radius=np.concatenate([old["wall_radius"],np.stack(rawr,1)],1),wall_x=z,reference_points=base.p,triangles=base.t,t=fields["t"])
 volume=fields["area"].sum(0)*float(fields["dx"]);net=fields["face_flow"][0]-fields["face_flow"][-1]
 defect=volume-volume[0]-np.r_[0,np.cumsum(dt*net[1:])]
 meta.update(source_seconds=meta["seconds"],max_relative_volume_balance=float(abs(defect).max()/volume[0]),seconds=time.perf_counter()-start,continued_from=str(a.source),duration=float(times[-1]),extension_boundary="Periodically repeated endpoint waveform; continuous stored wall/fluid state",original_window_preserved_exactly=True,nonlinear_history=hist)
 (a.output/"metrics.json").write_text(json.dumps(meta,indent=2));print("Independent flow continuation complete",meta["seconds"],flush=True)
if __name__=="__main__":main()
