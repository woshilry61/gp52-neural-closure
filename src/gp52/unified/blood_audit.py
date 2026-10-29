"""Record blood-field validity and conservative transfer before biochemical generation."""
import json,hashlib
import numpy as np
from .protocol import RUN,COUNTS
from .reference import canonical_blood,flow
def main():
 canonical_blood()
 records={}
 for name in ['single','three','seven']:
  z=np.load(RUN/name/'blood.npz')
  assert all(np.isfinite(z[k]).all() for k in z.files)
  mins=[float(z[f'v{v}_volume'].min()) for v in range(len(COUNTS[name]))]
  assert min(mins)>0,mins
  _,_,transfer=flow(name,tuple(COUNTS[name]),1,.002)
  assert transfer['continuity_max']<1e-12
  assert transfer.get('junction_max',0)<1e-12
  records[name]=dict(minimum_cell_volume_m3=mins,transfer=transfer,
   canonical_sha256=hashlib.sha256((RUN/name/'blood.npz').read_bytes()).hexdigest())
 raw=json.loads((RUN/'seven/blood/metrics.json').read_text())
 h=raw['history']
 residual=max(r['weak_divergence'] for r in h)
 junction=max(max(r['junction_flux_abs_m3_s']) for r in h)
 linear=max(r['linear_relative_residual'] for r in h)
 assert residual<1e-8 and junction<1e-10 and linear<1e-8,(residual,junction,linear)
 records['seven']['blood_solve']=dict(max_weak_divergence=residual,max_junction_flux_m3_s=junction,max_linear_relative_residual=linear,
  nx=raw['nx'],nr=raw['nr'],dt=raw['dt'],duration=raw['duration'])
 records['seven']['resolution_scope']='Conservative independent CFD at the stated mesh; the biochemical refinement check does not establish full seven-vessel blood-mesh convergence.'
 (RUN/'blood-audit.json').write_text(json.dumps(records,indent=2))
 print(json.dumps(records),flush=True)
if __name__=='__main__':main()
