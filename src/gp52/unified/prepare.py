"""Build single-vessel buffers from the common independent CFD blood."""
import json,subprocess,sys,os
import numpy as np
from .protocol import RUN,ROOT,COUNTS
from .reference import flow
def main():
 out=RUN/'single';blood=out/'prepared-input';blood.mkdir(exist_ok=True)
 d,t,stats=flow('single',tuple(COUNTS['single']),1,.002);d=d[0]
 fields=dict(x=d['x'],xf=d['xf'],t=t,dt=np.array(.002),dx=np.array(.06/41),
  area=d['area'].T[None],face_area=d['af'].T[None],face_flow=d['q'].T[None])
 np.savez_compressed(blood/'fields.npz',**fields)
 (blood/'metrics.json').write_text(json.dumps(dict(selected=dict(clean=0))))
 subprocess.run([sys.executable,'-m','gp52.raw.prepare_assets','--blood',str(blood),'--output',str(out/'assets')],check=True)
if __name__=='__main__':main()
