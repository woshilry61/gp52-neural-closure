"""Replay public inputs and compare frozen-model metrics with the archived experiment."""
from pathlib import Path
import argparse,csv,gc,json,time
import numpy as np
import torch
from gp52.unified.protocol import ROOT,RUN,TEST,inputs,SEEDS
from gp52.unified.models import model
from gp52.unified.evaluate import configure,reference,observations,errors,inlet_error
from gp52.three import inverse as inv
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--skip-inverse',action='store_true');a=p.parse_args()
 rows=list(csv.DictReader((ROOT/'reports/unified-cfd-baselines-v1/per-case-errors.csv').open()))
 selection=list(csv.DictReader((ROOT/'reports/unified-cfd-baselines-v1/inverse-selection.csv').open()))
 configure();result={'status':'running','forward_tolerance':1e-8,'inverse_tolerance':5e-4,'inverse_case':80,'records':[]};start=time.perf_counter()
 dest=RUN/'release-validation.json'
 def save():dest.write_text(json.dumps(result,indent=2)+'\n')
 save()
 for name in ['single','three','seven']:
  z=reference(name);obs=observations(name)
  for kind,seed in [('C0',42)]+[('new',s) for s in SEEDS]:
   m=model(name,kind,seed);pred=m.predict(inputs()[TEST],True)
   actual=errors(pred,z['y'][TEST],z['volume'],z['t'])
   expected=np.array([[float(next(r for r in rows if r['network']==name and r['setting']=='forward' and r['method']==m.name and int(r['case'])==c)[k]) for k in ['Th','Fg','Fn']] for c in TEST])
   gap=float(abs(actual-expected).max())
   assert gap<result['forward_tolerance'],(name,m.name,gap)
   record={'network':name,'method':m.name,'forward_cases':len(TEST),'forward_metric_max_abs_difference':gap}
   if not a.skip_inverse:
    lam=float(next(r['lambda_selected'] for r in selection if r['network']==name and r['method']==m.name))
    problem=inv.problems(obs,np.array([80]));fit=inv.solve(m,problem,lam,'release-replay')
    y=m.predict(fit['amplitudes'],True)
    ee=errors(y,z['y'][fit['cases']],z['volume'],z['t']);ie=inlet_error(fit['amplitudes'],inputs()[fit['cases']])
    vals=np.c_[ie,ee]
    expected=np.array([[float(next(r for r in rows if r['network']==name and r['setting']=='inverse' and r['method']==m.name and int(r['case'])==80 and int(r['rep'])==rep and int(r['sites'])==sites)[k]) for k in ['inlet','Th','Fg','Fn']] for rep,sites in zip(fit['reps'],fit['layouts'])])
    inverse_gap=float(abs(vals-expected).max())
    assert inverse_gap<result['inverse_tolerance'],(name,m.name,inverse_gap)
    record.update(inverse_fits=len(vals),inverse_metric_max_abs_difference=inverse_gap,lambda_from_archived_validation=lam)
   result['records'].append(record);save();print('VERIFIED',json.dumps(record),flush=True)
   del m,pred;gc.collect();torch.cuda.empty_cache()
 result.update(status='passed',seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),training_performed=False);save()
 print(json.dumps({'status':'passed','seconds':result['seconds']}),flush=True)
if __name__=='__main__':main()
