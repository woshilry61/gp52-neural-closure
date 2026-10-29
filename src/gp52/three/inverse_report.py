"""Reproducible tables, paired uncertainty, figures and provenance for sparse inversion."""
from pathlib import Path
import csv,json,hashlib,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .reference import ROOT,KNOTS,inlets
from .inverse import inlet_errors
BASE=ROOT/'runs/inverse-three'
REPORT=ROOT/'reports/three-vessel-inverse'
RUNLEAF='tuned'
METHODS=['ordinary_1d','neural_42','neural_43','neural_44']
METRICS=['inlet_082','fn_082','replay_fn_082','inlet_123','fn_123','replay_fn_123']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(pred,truth,w):return np.sqrt(np.sum((pred-truth)**2*w,axis=(1,2))/np.sum(truth**2*w,axis=(1,2)))
def gather():
 amps,_=inlets();raw=np.load(ROOT/'runs/reference-final/training-grid.npz');truth=raw['y'];rows=[];replay_check=[]
 for window in [.82,1.23]:
  parent=BASE if window<1 else BASE/'long-window';folder=parent/RUNLEAF
  for method in METHODS:
   fit=dict(np.load(folder/f'test-{method}.npz'));pred=dict(np.load(folder/f'prediction-{method}.npz'))
   replay=[]
   for start in range(0,len(fit['cases']),48):
    stop=min(start+48,len(fit['cases']));path=folder/'replay'/f'{method}-{start:03d}-{stop:03d}'
    rr=np.load(path/'data.npz')
    assert np.array_equal(rr['amplitudes'],fit['amplitudes'][start:stop])
    prov=json.loads((path/'input-provenance.json').read_text());assert prov['source_sha256']==sha(folder/f'test-{method}.npz')
    replay.append(rr['y'][:,2,:,::2])
    q=json.loads((path/'metrics.json').read_text());replay_check.append(dict(path=str(path.relative_to(ROOT)),minimum=q['minimum'],mass_balance_max=q['mass_balance_max']))
   replay=np.concatenate(replay);t=pred['t'];vol=pred['volume'].T
   scores={}
   for end,suffix in [(.82,'082'),(1.23,'123')]:
    mask=t<=end+1e-9;gt=truth[fit['cases'],2][...,mask];w=vol[:,mask]
    scores['inlet_'+suffix]=inlet_errors(fit['amplitudes'],amps[fit['cases']],t[mask])
    scores['fn_'+suffix]=rel(pred['y'][:,2][...,mask],gt,w)
    scores['replay_fn_'+suffix]=rel(replay[...,mask],gt,w)
   for i,c in enumerate(fit['cases']):
    start=int(fit['chosen'][i]);row=dict(observation_end=window,method=method,case_id=int(c),noise_rep=int(fit['reps'][i]),sensors=int(fit['layouts'][i]),regularization=float(fit['lam']),start=start,
     projected_gradient=float(fit['all_pg'][i,start]),iterations=int(fit['all_iterations'][i,start]),termination=str(fit['all_reason'][i,start]),objective=float(fit['cost'][i]))
    row.update({k:float(v[i]) for k,v in scores.items()});rows.append(row)
 return rows,replay_check

def aggregate(rows):
 summary={};paired=[];rng=np.random.default_rng(2026102805);draw=rng.integers(0,16,(10000,16))
 for window in [.82,1.23]:
  summary[str(window)]={}
  for noise in ['clean','noisy']:
   selected=[r for r in rows if r['observation_end']==window and ((r['noise_rep']==0) if noise=='clean' else (r['noise_rep']>0))]
   summary[str(window)][noise]={}
   for n in [1,2,3]:
    group=[r for r in selected if r['sensors']==n];out={}
    for method in METHODS+['neural_mean']:
     z=[r for r in group if (r['method'].startswith('neural_') if method=='neural_mean' else r['method']==method)]
     out[method]={k:float(np.mean([r[k] for r in z])) for k in METRICS}
     out[method]['selected_over_1e8']=sum(r['projected_gradient']>1e-8 for r in z)
     out[method]['selected_over_1e6']=sum(r['projected_gradient']>1e-6 for r in z)
    summary[str(window)][noise][str(n)]=out
    for metric in METRICS:
     a=np.array([np.mean([r[metric] for r in group if r['case_id']==c and r['method'].startswith('neural_')]) for c in range(32,48)])
     b=np.array([np.mean([r[metric] for r in group if r['case_id']==c and r['method']=='ordinary_1d']) for c in range(32,48)])
     delta=a-b;ci=np.quantile(delta[draw].mean(1),[.025,.975])
     paired.append(dict(observation_end=window,noise=noise,sensors=n,metric=metric,mean_neural_minus_ordinary=float(delta.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),neural_better_cases=int(sum(delta<0)),cases=16))
 return summary,paired

def plots(summary,rows):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
 fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
 for row,w in enumerate([.82,1.23]):
  for col,(metric,label) in enumerate([('inlet_082','Inlet recovery'),('fn_082','Unobserved Fn'),('replay_fn_082','Fn after reference replay')]):
   ax=axes[row,col]
   for method,color,title in [('ordinary_1d','#2962a0','Ordinary 1D'),('neural_mean','#d46628','Frozen neural closure')]:
    vals=[summary[str(w)]['noisy'][str(n)][method][metric]*100 for n in [1,2,3]]
    ax.plot([1,2,3],vals,'o-',color=color,label=title,lw=2)
   ax.set(title=f'{label} | observe to {w:.2f} s',xlabel='Number of sensor sites',ylabel='Relative L2 error (%)',xticks=[1,2,3]);ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=9)
 fig.suptitle('Three-vessel sparse inversion: 16 held-out inlets, three noise realizations\nScoring window 0–0.82 s; Th/Fg observed, Fn never fitted',fontsize=13)
 fig.savefig(REPORT/'comparison.png');plt.close(fig)
 # Fixed example: first held-out inlet, first noisy realization. No best-case selection.
 amps,_=inlets();truth=np.load(ROOT/'runs/reference-final/training-grid.npz');tt=truth['t'];case=32;rep=1
 for window in [.82,1.23]:
  folder=(BASE if window<1 else BASE/'long-window')/RUNLEAF;fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
  data={m:(dict(np.load(folder/f'test-{m}.npz')),dict(np.load(folder/f'prediction-{m}.npz'))) for m in METHODS}
  for col,n in enumerate([1,2,3]):
   ax=axes[0,col];ax.plot(tt,np.interp(tt,KNOTS,amps[case]),'k',lw=2,label='True inlet')
   for m,color in [('ordinary_1d','#2962a0'),('neural_42','#d46628')]:
    f,p=data[m];i=np.flatnonzero((f['cases']==case)&(f['reps']==rep)&(f['layouts']==n))[0]
    ax.plot(tt,np.interp(tt,KNOTS,f['amplitudes'][i]),color=color,label='Ordinary 1D' if m=='ordinary_1d' else 'Neural seed 42')
   ax.axvline(.82,color='gray',ls=':',lw=1);ax.set(title=f'{n} sensor site(s): recovered inlet',xlabel='Time (s)',ylabel='Th inlet');ax.grid(alpha=.2)
   # Fn at the unobserved daughter-2 midpoint is not an inversion target.
   node=48;ax=axes[1,col];ax.plot(tt,truth['y'][case,2,node],'k',lw=2,label='Resolved reference')
   for m,color in [('ordinary_1d','#2962a0'),('neural_42','#d46628')]:
    f,p=data[m];i=np.flatnonzero((f['cases']==case)&(f['reps']==rep)&(f['layouts']==n))[0]
    ax.plot(tt,p['y'][i,2,node],color=color,label='Ordinary 1D' if m=='ordinary_1d' else 'Neural seed 42')
   ax.set(title='Fn at daughter-2 midpoint',xlabel='Time (s)',ylabel='Fn');ax.grid(alpha=.2)
  axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
  fig.suptitle(f'Fixed example: test inlet 32, noise replicate 1, observations through {window:.2f} s')
  fig.savefig(REPORT/f'example-{window:.2f}.png');plt.close(fig)

def table(s,w,noise):
 out=['| Sites | Inlet: ordinary / neural | Fn: ordinary / neural | Replay Fn: ordinary / neural |','|---|---:|---:|---:|']
 for n in [1,2,3]:
  a=s[str(w)][noise][str(n)]['ordinary_1d'];b=s[str(w)][noise][str(n)]['neural_mean']
  out.append('| '+str(n)+' | '+' | '.join(f"{100*a[k]:.2f}% / {100*b[k]:.2f}%" for k in ['inlet_082','fn_082','replay_fn_082'])+' |')
 return '\n'.join(out)

def main():
 global REPORT,RUNLEAF
 ap=argparse.ArgumentParser();ap.add_argument('--common',action='store_true');a=ap.parse_args()
 if a.common:RUNLEAF='precision';REPORT=ROOT/'reports/three-vessel-inverse-common'
 REPORT.mkdir(exist_ok=True);rows,replay=gather();summary,paired=aggregate(rows)
 for name,data in [('per-case.csv',rows),('paired-comparisons.csv',paired)]:
  with (REPORT/name).open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
 (REPORT/'summary.json').write_text(json.dumps(summary,indent=2))
 protected=json.loads((BASE/'protected-before.json').read_text());changed=[name for name,h in protected.items() if sha(ROOT/name)!=h]
 assert not changed,changed
 gpu=[]
 for file in [BASE/'gpu-validation.csv',BASE/'gpu-test.csv',BASE/'long-window/gpu-validation.csv',BASE/'long-window/gpu-test.csv',BASE/'gpu-precision.csv',BASE/'gpu-tuned.csv']:
  if not file.exists():continue
  v=np.loadtxt(file,delimiter=',',usecols=(1,2,3),ndmin=2)
  gpu.append(dict(path=str(file.relative_to(ROOT)),samples=len(v),utilization_mean=float(v[:,0].mean()),utilization_max=float(v[:,0].max()),peak_memory_MiB=float(v[:,1].max()),peak_power_W=float(v[:,2].max())))
 precision=json.loads((BASE/'precision-verification.json').read_text())
 checks=dict(protected_files=len(protected),changed_protected_files=changed,replayed_fits=len(rows),replay_minimum=min(z['minimum'] for z in replay),replay_mass_balance_max=max(z['mass_balance_max'] for z in replay),gpu=gpu,precision=precision)
 (REPORT/'verification.json').write_text(json.dumps(checks,indent=2))
 for name in ['gpu-verification.json','precision-verification.json','diagnostics.json']:
  (REPORT/name).write_text((BASE/name).read_text())
 for w,folder in [('082',BASE),('123',BASE/'long-window')]:
  (REPORT/f'validation-selection-{w}.json').write_text((folder/RUNLEAF/'selection.json').read_text())
 manifest={str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('src/gp52/three/inverse*.py')}
 for w,folder in [('082',BASE),('123',BASE/'long-window')]:
  for p in (folder/RUNLEAF).glob('*.npz'):manifest[str(p.relative_to(ROOT))]=sha(p)
 for name in ['runs/sen-three-gpu-v2/fields.npz','runs/reference-final/training-grid.npz','runs/reference-final/data.npz']:
  manifest[name]=sha(ROOT/name)
 manifest.update(protected)
 (REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2))
 plots(summary,rows)
 maxpg=max(r['projected_gradient'] for r in rows);over8=sum(r['projected_gradient']>1e-8 for r in rows);over6=sum(r['projected_gradient']>1e-6 for r in rows)
 timings=[]
 for w in [.82,1.23]:
  f=(BASE if w<1 else BASE/'long-window')/RUNLEAF
  timings.append(f"- Observation end {w:.2f} s: "+', '.join(f"{m} {float(np.load(f/f'test-{m}.npz')['seconds']):.2f} s" for m in METHODS)+".")
 cirows=['| Sites | Inlet difference, percentage points (95% CI) | Fn difference, percentage points (95% CI) |','|---|---:|---:|']
 for n in [1,2,3]:
  cells=[]
  for metric in ['inlet_082','fn_082']:
   z=next(r for r in paired if r['observation_end']==1.23 and r['noise']=='noisy' and r['sensors']==n and r['metric']==metric)
   cells.append(f"{100*z['mean_neural_minus_ordinary']:.2f} ({100*z['ci95_low']:.2f}, {100*z['ci95_high']:.2f})")
  cirows.append('| '+str(n)+' | '+' | '.join(cells)+' |')
 citab='\n'.join(cirows)
 supported={}
 for metric in ['inlet_082','fn_082']:
  supported[metric]=[str(z['sensors']) for z in paired if z['observation_end']==1.23 and z['noise']=='noisy' and z['metric']==metric and z['ci95_high']<0]
 supporttext='The case-clustered 95% interval favors the neural model for inlet recovery at sensor counts '+(', '.join(supported['inlet_082']) or 'none of the')+', and for Fn at sensor counts '+(', '.join(supported['fn_082']) or 'none of the')+'. Intervals containing zero do not establish a reliable advantage.'
 tuningtext=('Each model independently selected its own regularization from the same validation grid and budget. Both observation windows select 0.0001 for ordinary 1D and 0 for each neural seed. The shared-regularization experiment is retained in ../three-vessel-inverse-common/REPORT.md.' if RUNLEAF=='tuned' else 'A common regularization was selected by equally weighting ordinary and neural families on validation cases: 0.0001 for the short window and 0 for the long window. This controlled secondary comparison fixes the prior strength across candidates; the main report allows each candidate its own validation optimum.')
 both=[str(n) for n in [1,2,3] if all(summary['1.23']['noisy'][str(n)]['neural_mean'][k]<summary['1.23']['noisy'][str(n)]['ordinary_1d'][k] for k in ['inlet_082','fn_082'])]
 text=f"""# Three-vessel sparse inlet inversion

The frozen three-vessel model has now been evaluated on one, two and three
sensor sites, against the ordinary 1D finite-volume model. No weights were
retrained. Both candidates use the same finite-volume time integrator; the
neural candidate adds the frozen learned transport/reaction closure. This is
a test of the hybrid closure model, rather than a replacement of the numerical
integrator by an unconstrained network. With the longer observation window, the neural model has lower
mean inlet and Fn errors for sensor counts {', '.join(both)}. Reliability is
assessed using paired case-clustered intervals below. The shorter-window
results remain separate; lower Fn error alone does not imply better inlet recovery.

## What is measured and what is predicted
At each sensor, observe Th and Fg at 24 times. Infer seven Th inlet amplitudes.
Fn is never supplied to the inverse solver. After inference, predict Fn over
all 57 axial cells. The one-site layout observes the parent midpoint (30 mm);
two sites add daughter 1 midpoint (10 mm); three add daughter 2 midpoint (17 mm).

The inlet varies on seven knots from 0 to 0.82 s and then stays at its final
concentration. The reference and predictions extend to 1.23 s. An observation
end of 0.82 s stops acquisition while the last input change is still weakly
visible downstream; 1.23 s allows another 0.41 s to observe its response.
These are acquisition windows, not two different models. Both contain 24
sample times per species and site, so the longer window has wider time spacing.

## Main comparison: noisy observations
Gaussian noise has absolute standard deviation 0.02 in the normalized
concentration units (not 2% of each instantaneous concentration).
Each table averages 16 held-out inlet cases and three noise realizations.
Neural entries average the errors of three independently frozen model seeds;
they are not predictions from a three-model ensemble.
All three error columns below are scored over **0–0.82 s**.
This is offline inverse reconstruction: the longer acquisition uses later
Th/Fg readings to reconstruct the earlier inlet and unobserved Fn. It is not
a causal forecast made at 0.82 s, and it is not a comparison of equal end times.

### Observe through 0.82 s
{table(summary,.82,'noisy')}

### Observe through 1.23 s
{table(summary,1.23,'noisy')}

For the longer window, paired neural-minus-ordinary differences are:

{citab}

Negative values favor the neural model. {supporttext}

“Replay Fn” takes each recovered inlet and runs the same resolved reference
solver again. Its error therefore measures the consequence of the recovered
boundary condition without the candidate model's forward error. It helps
detect cancellation between an incorrect inlet and an incorrect forward model.

![Noisy comparison](comparison.png)

## Noise-free reference observations
Noise-free observations still come from the resolved reference, so the 1D
candidate models retain forward-model discrepancy.

### Observe through 0.82 s
{table(summary,.82,'clean')}

### Observe through 1.23 s
{table(summary,1.23,'clean')}

## Why observation duration matters
The Th observation operator is affine in inlet amplitudes. Its singular values
are recorded in diagnostics.json. For one sensor, the ordinary model's
condition number falls from approximately 23,384 with the short window to
110 with the long window. Neural seed 42 falls from approximately 19,747
to 116. The downstream sensor has very little sensitivity to the last input
knot in the short window. Adding downstream sites does not by itself fix
this missing temporal information.

The long-window same-model, noise-free recovery check reconstructs known
inputs with relative inlet errors below 3e-7 for all models. This verifies
the inversion implementation; it is separate from the harder resolved-reference
test. The remaining reference-test error includes model discrepancy,
measurement noise and weakly observed inlet variations. A smaller Fn error
does not establish an accurate recovered inlet, so both are reported.

Regularization uses validation cases 24–31 only, never test cases. Candidate
values are 0, 0.0001 and 0.01. {tuningtext}
Validation scores for every model and candidate are included in the selection files.

## Fairness and numerical verification
- Same Sen-derived, conservatively projected blood field, 30/10/17 axial cells,
  dt=0.002 s, reaction parameters, initial state and boundary conditions.
- Same sensor sites, time samples, stored noise draws, seven input parameters,
  bounds [0,1.2], observation weights, second-difference prior form and validation search budget.
- Three starts: bounded linear Th fit, constant 0.3 and constant 0.9.
  Select the smallest observation/prior objective, never the true-input error.
- Both solvers use the same batched constrained Gauss–Newton/LM search and
  line search. Final double-precision solves allow at most 100 iterations
  per start and target projected gradient 1e-8.
- Float32 FV acceleration was first checked against all 48 saved CPU forward
  runs and four new random inputs. Final tables use double-precision FV,
  while the frozen neural head retains its original float32 weights.
- Final GPU/CPU concentration discrepancies and all optimization termination
  states are retained. Across {len(rows)} final selected fits, maximum projected
  gradient is {maxpg:.3g}; {over8} exceed 1e-8 and {over6} exceed 1e-6.
  No selected fit or failed start is removed.
- All {len(rows)} final recovered inputs were independently replayed on the
  resolved reference (120/40/68 axial cells, 24 radial rings, dt=0.001 s).
  Maximum normalized mass-balance residual is {checks['replay_mass_balance_max']:.3g}.
- {len(protected)} protected source, report and weight files have identical
  SHA256 hashes before and after this extension.

## Uncertainty and scope
paired-comparisons.csv contains case-clustered 95% bootstrap intervals for
neural-minus-ordinary errors, with noise realizations and neural seeds averaged
within each of the 16 independent test inlets. Negative differences favor the
neural model. Seed-specific results, full 0–1.23 s errors and every individual
case are in summary.json and per-case.csv. An average advantage does not mean
every individual inlet is improved.

This is one synthetic, fixed-geometry, fixed-blood three-vessel experiment.
Th/Fg/Fn are generated by reaction–transport equations, not mouse biochemical
measurements. The reference resolves axisymmetric prescribed-flow transport
inside three tubes connected by a perfectly mixed junction. It is not 3D
bifurcation Navier–Stokes CFD. The candidate models remain strictly 1D; radial
rings are used only in the independent reference.

The ordinary baseline is the same conservative 1D FV solver with mean-product
reaction and no learned closure. The forward report additionally contains its
mesh-refined and fixed-physics closure comparisons. This inverse report does
not establish superiority over every possible calibrated 1D closure.

## GPU work and runtime
The repeated FV rollouts use fused CUDA kernels; reaction-head evaluation and
candidate inputs are batched on the RTX 5090. No retraining was needed.
Both candidates use the same GPU backend. Final solve times for 192 inverse
problems per method (three starts each), excluding setup, file writing and
reference replays, are:

{chr(10).join(timings)}

These are recorded workload times, not a standalone latency benchmark.
The neural solver adds a predictor/corrector and graph-head evaluation, so an
accuracy advantage does not imply a speed advantage. GPU sampling, memory and
power measurements are included in verification.json. The resolved reference
uses the unchanged sparse CPU solver.

## Fixed examples
The examples use the first held-out inlet (32), first noise realization and
neural seed 42, with no selection for favorable appearance.

![Short-window example](example-0.82.png)
![Long-window example](example-1.23.png)

## Reproduction
See ../../docs/THREE_VESSEL_INVERSE_README.md for the full command sequence.
Existing final weights, earlier single-vessel work and the three-vessel forward
report are unchanged. New raw runs are under runs/inverse-three; final
double-precision results are in each observation window's {RUNLEAF} folder.
"""
 (REPORT/'REPORT.md').write_text(text)
 print(json.dumps(dict(report=str(REPORT),rows=len(rows),max_projected_gradient=maxpg,over_1e8=over8,over_1e6=over6,protected=len(protected)),indent=2))
if __name__=='__main__':main()
