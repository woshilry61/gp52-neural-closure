"""Create the auditable three-vessel report and figures from saved results."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .reference import ROOT,COUNTS,LENGTH
def main():
 out=ROOT/'reports/three-vessel';out.mkdir(parents=True,exist_ok=True)
 R={s:json.loads((ROOT/f'runs/evaluation-{s}/metrics.json').read_text()) for s in [42,43,44]}
 P=json.loads((ROOT/'runs/physical-baselines/metrics.json').read_text())
 sen=json.loads((ROOT/'runs/sen-three-gpu-v2/metrics.json').read_text())
 names=['Th','Fg','Fn']; rows=[]
 def add(label,v):
  rows.append(dict(method=label,**{name:float(100*v[i]) for i,name in enumerate(names)}))
 add('Ordinary 1D, matched grid',R[42]['ordinary_1d']['test']['0.82']['mean_relative_l2'])
 add('Ordinary 1D, refined',P['ordinary_1d_fine']['test']['0.82']['mean_relative_l2'])
 add('Physical single-memory closure',P['physical_single_memory']['test']['0.82']['mean_relative_l2'])
 add('Quasi-steady Taylor dispersion',P['quasisteady_taylor']['test']['0.82']['mean_relative_l2'])
 for method,label in [('neural_transport','Neural transport only'),('neural_reaction_only','Neural reaction only'),('neural_combined','Neural transport + reaction')]:
  add(label,np.mean([R[s][method]['test']['0.82']['mean_relative_l2'] for s in R],0))
 baseline=np.array(R[42]['ordinary_1d']['test']['0.82']['per_case'])
 nn=np.mean([R[s]['neural_combined']['test']['0.82']['per_case'] for s in R],0)
 delta=100*(baseline-nn);rng=np.random.default_rng(20261028);boot=delta[rng.integers(0,16,size=(10000,16))].mean(1)
 stats=dict(improvement_percentage_points=delta.mean(0).tolist(),paired_bootstrap_95_percent_interval=np.quantile(boot,[.025,.975],axis=0).tolist(),wins_over_ordinary=(nn<baseline).sum(0).tolist(),cases=16,unit='Held-out inlet waveform; training seeds averaged within each case')
 gp=[]
 for s in R:
  a=np.loadtxt(ROOT/f'runs/closure-final-{s}/gpu.csv',delimiter=',',usecols=1)
  gp.append(dict(seed=s,gpu_mean=float(a.mean()),gpu_max=float(a.max()),seconds=json.loads((ROOT/f'runs/closure-final-{s}/metrics.json').read_text())['seconds']))
 ref=dict(np.load(ROOT/'runs/reference-final/training-grid.npz'));t=ref['t']
 preds={name:dict(np.load(ROOT/f'runs/evaluation-42/{name}.npz')) for name in ['ordinary_1d','neural_transport','neural_combined']}
 volume=preds['ordinary_1d']['volume'];offset=np.r_[0,np.cumsum(COUNTS)]
 per_vessel={}
 for v in range(3):
  sl=slice(offset[v],offset[v+1]);mask=t<=.82+1e-9;w=volume[sl][:,mask]
  per_vessel[str(v+1)]={}
  for method in ['ordinary_1d','neural_combined']:
   a=preds[method]['y'][32:,:,sl,:][...,mask];b=ref['y'][32:,:,sl,:][...,mask]
   per_vessel[str(v+1)][method]=(100*np.sqrt(((a-b)**2*w[None,None]).sum((2,3))/((b*b)*w[None,None]).sum((2,3))).mean(0)).tolist()
 summary=dict(primary_window_s=[0,.82],metrics='Volume/time weighted relative L2 percent, averaged over 16 held-out inlet curves; neural table averages all three seeds',comparison=rows,paired_statistics=stats,per_vessel_seed42=per_vessel,gpu_training=gp,sen=sen,reference_checks=P['reference_refinement'],physical_moments=P['physical_moments'])
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 with (out/'comparison.csv').open('w') as f:
  wr=csv.DictWriter(f,fieldnames=['method',*names]);wr.writeheader();wr.writerows(rows)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
 case=32
 for v,ax in enumerate(axs):
  ix=offset[v]+COUNTS[v]//2
  ax.plot(t,ref['y'][case,2,ix],c='black',lw=2,label='Resolved reference')
  ax.plot(t,preds['ordinary_1d']['y'][case,2,ix],c='#d97b22',ls='--',lw=1.8,label='Ordinary 1D')
  ax.plot(t,preds['neural_combined']['y'][case,2,ix],c='#1967a3',lw=1.8,label='Neural closure + 1D solver')
  ax.set(title=['Parent vessel midpoint','Daughter 1 midpoint','Daughter 2 midpoint'][v],xlabel='Time (s)',ylabel='Fn (model concentration units)')
  ax.axvline(.82,color='gray',ls=':',alpha=.5)
 axs[0].legend(fontsize=8)
 fig.suptitle('Three-vessel transport: prespecified held-out inlet 32')
 fig.savefig(out/'fn-waveforms.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 selected=[0,1,3,4,6];labs=['1D','1D refined','Taylor','NN transport','NN full'];colors=['#d97b22','#b99b73','#9c6db0','#57a8c5','#1967a3']
 for i,ax in enumerate(axs):
  vals=[rows[k][names[i]] for k in selected]
  ax.bar(np.arange(len(vals)),vals,color=colors);ax.set_xticks(np.arange(len(vals)),labs,rotation=25,ha='right')
  ax.set(title=names[i],ylabel='Mean relative L2 error (%)')
  for j,value in enumerate(vals):ax.text(j,value+.08,f'{value:.2f}',ha='center',fontsize=8)
 fig.suptitle('16 held-out inlets, 0-0.82 s; neural results averaged over 3 seeds')
 fig.savefig(out/'prediction-errors.png',dpi=180);plt.close(fig)
 # Blood waveforms at the junction, which are not velocity training observations.
 sb=dict(np.load(ROOT/'runs/sen-three-gpu-v2/fields.npz'));si=dict(np.load(ROOT/'runs/sen-input/input.npz'));chosen=int(sb['selected'])
 fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained')
 for v in range(3):
  point=4 if v==0 else 1;ix=9 if v==0 else 0;refb=si[f'ref{v+1}_{point}']
  for j,(name,col,scale,unit) in enumerate([('velocity',3,1,'m/s'),('area',5,1e6,'mm2')]):
   ax=axs[j,v];ax.plot(sb['t'],refb[:,col]*scale,c='black',lw=1.5,label='Numerical reference')
   ax.plot(sb['t'],sb[name][chosen,v,ix]*scale,c='#1967a3',ls='--',label='Sen GPU reproduction')
   ax.set(title=f'Vessel {v+1}, junction end',xlabel='Time (s)',ylabel=f'{name.capitalize()} ({unit})')
 axs[0,0].legend(fontsize=8);fig.savefig(out/'sen-blood-waveforms.png',dpi=180);plt.close(fig)
 table='| Method | Th | Fg | Fn |\n|---|---:|---:|---:|\n'+'\n'.join('| '+r['method']+' | '+' | '.join(f"{r[n]:.2f}%" for n in names)+' |' for r in rows)
 maxbalance=max(R[s][m]['rollout']['mass_balance_max'] for s in R for m in ['ordinary_1d','neural_combined'])
 source0=100*P['physical_moments']['42']['source_mean_product_relative_l2']
 sourcen=100*np.mean([P['physical_moments'][str(s)]['source_neural_relative_l2'] for s in R])
 txt=f"""# Three-vessel extension: results and scope

## Completed work
- GPU reproduction of the pinned Sen 2024 three-vessel blood-flow example.
- Strict one-dimensional transport and catalytic Fg-to-Fn reaction on a Y network.
- A zero-volume junction with concentration mixing and exact discrete mass balance.
- Two learned temporal flux-memory variables per internal edge and a graph reaction head.
- Matched-grid ordinary 1D comparison, refined ordinary 1D comparison, two fixed physical dispersion probes, and transport/reaction ablations.
- Three independent biochemical training seeds and 16 held-out chemical inlet waveforms.

The original release, its checkpoints and the prior Sen/CCA experiments are unchanged.
This is a separate experiment; it is not an update to the published final model.

## Main result
{table}

Errors are volume/time-weighted relative L2, averaged across held-out inlets, over 0-0.82 s.
Neural rows average seeds 42, 43 and 44. Every seed is retained.
The full neural model improves Th/Fg/Fn on {stats['wins_over_ordinary']} of 16 paired cases (seed-averaged).
The improvement comes predominantly from transport closure. The added Fn improvement from
the reaction head is small and below the reference-refinement differences; it is not a
demonstrated practically significant Fn gain.

![Errors](prediction-errors.png)
![Fn curves](fn-waveforms.png)

The refined ordinary 1D result is worse against the cross-section-resolved reference:
coarse-grid numerical diffusion partially compensates for omitted cross-sectional mixing.
This is not evidence that refinement is mathematically wrong. It distinguishes numerical
discretization error from the model reduction error.

## Blood reproduction
The source is upstream commit bf0ec89f0aab8a639bdd2d363dc3c0cfae770c1d.
The PyTorch port is checked against the original TensorFlow forward pass, loss and all
parameter gradients; discrepancies are approximately 1e-16 in float64.
The architecture, observations and loss are retained. Adam implementation and initialization
differ across frameworks. Training uses 100,000 scheduled updates plus four warm-up/capture
updates, for each of 32 random initializations. Selection uses only the training objective.
Elapsed training time: {sen['seconds']:.2f} s. Mean GPU utilization: {sen['gpu_utilization_mean']:.2f}%.

Velocity errors at the three observed boundaries are below 0.1%. Errors at unobserved
recorded positions range from about 0.75% to 7.37%, with the largest at the smaller daughter
inlet. Area relative errors are about 0.25%-0.31%. The raw junction flow mismatch is
{100*sen['junction_flow_relative_l2']:.4f}% relative L2. All positions are in summary.json.

![Blood reconstruction](sen-blood-waveforms.png)

The original upstream three-vessel graph uses identical distal labels for both daughter
vessels. Its graph therefore contains a distal shortcut. That original graph is retained
for reproduction only. The biochemical Y network uses distinct distal outlets.

## What the reference is
Blood is the selected Sen prediction, explicitly projected onto discrete local volume
conservation and junction flow conservation. The flow correction is about 5.3%-5.5% in
relative L2 (depending on grid/time discretization), and is shared by all biochemical methods.
Raw Sen predictions are saved separately. This benchmark does not claim that the unprojected
network output is exactly conservative.

The biochemical reference resolves axial and radial concentration variation in each tube,
with a prescribed parabolic axial velocity profile and continuity-compatible radial flux.
The three tubes meet at a perfectly mixed, zero-volume junction. It is a resolved
advection-diffusion-reaction reference, NOT a full three-dimensional bifurcation Navier-Stokes
CFD simulation. It cannot validate vortex mixing in a real bifurcation.

Final reference: 120/40/68 axial cells, 24 radial cells, dt=0.001 s.
Reduced models: 30/10/17 axial cells, no radial concentration cells, dt=0.002 s.
Additional reference check: 240/80/136 axial cells, 32 radial cells, dt=0.0005 s,
on prespecified cases 0 and 32. Volume-weighted full-window Fn differences are
{100*P['reference_refinement']['0']['relative_l2'][2]:.3f}% and
{100*P['reference_refinement']['32']['relative_l2'][2]:.3f}%. The method ranking remains
unchanged for the checked held-out case.

## Data, chemistry and training
Only the seven control amplitudes of the chemical inlet are sampled randomly, from
[0.05,1.15], using seed 2026102801. Th/Fg/Fn fields are computed from physical transport
and reaction equations, not sampled independently.
There are 24 training, 8 validation and 16 test inlet curves.
Initial Th is a Gaussian in the parent vessel, initial Fg is 1, and initial Fn is 0.
The upstream reservoir supplies the sampled Th and Fg=1. Distal inflow reservoirs have
Th=0 and Fg=1. Diffusivity is 2e-4 m2/s; reaction rate is 2*Th*Fg; Fn is immobile.
These are synthetic verification settings, not calibrated mouse coagulation parameters.

Transport is trained on the cross-sectional advective covariance flux.
Reaction is trained on the cross-sectional physical reaction source.
No joint trajectory fitting is used to hide transport error in reaction coefficients.
Validation physical-moment errors select checkpoints. Test cases never select checkpoints.
The source error on true test concentration fields decreases from {source0:.3f}% for the
mean-product source to {sourcen:.3f}% for the learned source. Its small absolute size explains
the limited incremental Fn effect.

The two memory variables are temporal flux states, not radial concentration subdivisions.
The architecture retains graph closure learning and conservative one-dimensional solving;
the edge graph and physical time scaling are adapted to the Y network.
This is a retrained extension, not a zero-shot application of the original weights.

## Conservation and performance
Maximum normalized Fg+Fn mass-balance residual across the ordinary/full neural rollouts:
{maxbalance:.3e}. No negative concentration was found in these rollouts.
The uniform-concentration test passes to machine precision.
Three concurrent GPU closure fits used mean utilization approximately
{np.mean([g['gpu_mean'] for g in gp]):.1f}%, with peak 100%.

The neural method still advances a numerical 1D solver, and its extra memory/corrector work
has a cost. No inference-speed superiority over ordinary 1D is claimed.
Reference generation uses sparse CPU factorization; neural training uses the RTX 5090.

## Remaining scope
- This tests new chemical inlet curves on one three-vessel geometry and blood condition.
  It does not establish cross-geometry or cross-blood generalization.
- It does not use the real mouse In_Vivo data.
- Three-vessel sparse biochemical inversion and seven-vessel extension are not included.
- Independent real biochemical measurements and true three-dimensional junction CFD are absent.
- The graph's independent benefit has not been isolated by a three-vessel equal-parameter
  no-graph ablation; the reported gain is for the complete learned closure.
- The improved biochemical result is conditional on the supplied projected Sen blood field;
  it is not an independent validation of physiological blood/coagulation prediction.

See docs/THREE_VESSEL_PROTOCOL.md and docs/THREE_VESSEL_README.md for exact scope and commands.
"""
 (out/'REPORT.md').write_text(txt)
 files=list(out.glob('*'))+[ROOT/f'src/gp52/three/{n}' for n in ['sen.py','reference.py','closure.py','evaluate.py','checks.py','report.py']]
 (out/'manifest.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file() and f.name!='manifest.json'},indent=2))
 print(json.dumps(dict(comparison=rows,paired=stats,gpu=gp),indent=2))
if __name__=='__main__':main()
