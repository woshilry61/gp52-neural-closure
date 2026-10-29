"""Evaluate every supplied Sen reference position without selecting on reference error."""
from pathlib import Path
import numpy as np,json,re
p=Path(__file__).resolve().parents[1]
z=dict(np.load(p/'runs/sen-three-gpu-v2/fields.npz'));src=dict(np.load(p/'runs/sen-input/input.npz'));best=int(z['selected']);rows=[]
for v,L in enumerate(src['length']):
 rawfile=p/f'external/sen2024/PIGNN_3_artery/Snaps/sim_1_{v+1}.his';locations={}
 for line in rawfile.open():
  m=re.match(r'# Point (\d+): x = ([\d.]+)',line)
  if m:locations[int(m[1])]=float(m[2])
  if not line.startswith('#'):break
 gx=np.linspace(0,L,10)
 for key,x in locations.items():
  ref=src[f'ref{v+1}_{key}']
  for name,col in [('velocity',3),('area',5)]:
   pred=np.array([np.interp(x,gx,z[name][best,v,:,j]) for j in range(len(src['t']))]);target=ref[:,col]
   observed=name=='velocity' and ((v==0 and key==1) or (v>0 and key==max(locations)))
   rows.append(dict(vessel=v+1,position_m=x,point_id=key,variable=name,observed_during_training=observed,relative_l2=float(np.linalg.norm(pred-target)/np.linalg.norm(target)),r2=float(1-np.sum((pred-target)**2)/np.sum((target-target.mean())**2))))
m=json.loads((p/'runs/sen-three-gpu-v2/metrics.json').read_text());m['metrics']=rows;m['optimizer_updates_per_seed']=m['epochs']+4
gpu=np.loadtxt(p/'runs/sen-three-gpu-v2/gpu.csv',delimiter=',',usecols=1);m['gpu_utilization_mean']=float(gpu.mean())
(p/'runs/sen-three-gpu-v2/metrics.json').write_text(json.dumps(m,indent=2))
print(json.dumps(m,indent=2))
