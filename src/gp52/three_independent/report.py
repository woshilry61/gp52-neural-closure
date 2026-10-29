"""Auditable tables, paired comparisons and figures for independent blood validation."""
import base64,csv,hashlib,html,json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gp52.three.reference import ROOT,COUNTS,LENGTH,KNOTS,flow
from .transfer import RUN,load_flow
OUT=ROOT/'reports/independent-cfd-v1'
SPECIES=['Th','Fg','Fn'];METHODS=['ordinary_1d','neural_42','neural_43','neural_44']
def savecsv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def errors(pred,truth,vol):
 w=vol.T[None,None];return np.sqrt(np.sum((pred-truth)**2*w,(2,3))/np.sum(truth**2*w,(2,3)))
def readref(path):
 d=dict(np.load(path));k=round(.002/(d['t'][1]-d['t'][0]))
 return dict(y=d['y'][...,::k],volume=d['volume'][::k],t=d['t'][::k],case_ids=d['case_ids'],amplitudes=d['amplitudes'])
def sites(y):
 cuts=np.r_[0,np.cumsum(COUNTS)];out=[]
 for v,n in enumerate(COUNTS):
  xx=(np.arange(n)+.5)*LENGTH[v]/n;w=np.array([np.interp(LENGTH[v]/2,xx,z) for z in np.eye(n)])
  out.append(np.einsum('...nt,n->...t',y[...,cuts[v]:cuts[v+1],:],w))
 return np.stack(out,-2)
def collect(ref):
 rows=[];regional=[];minima=[];optimizer=[];curves={};t=ref['t'];ids=np.arange(32,48);basis=np.stack([np.interp(t[t<=.82+1e-9],KNOTS,z) for z in np.eye(7)])
 for blood in ['cfd','sen']:
  folder=RUN/f'evaluation-{blood}'
  for method in METHODS:
   f=dict(np.load(folder/f'forward-{method}.npz'));minima.append(dict(blood=blood,method=method,scenario='forward',**dict(zip(SPECIES,f['y'].min((0,2,3))))));e=errors(f['y'],ref['y'][ids],ref['volume'])
   curves[f'{blood}-{method}-forward']=sites(f['y'])
   cuts=np.r_[0,np.cumsum(COUNTS)]
   for v in range(3):
    sl=slice(cuts[v],cuts[v+1]);ev=errors(f['y'][:,:,sl],ref['y'][ids][:,:,sl],ref['volume'][:,sl])
    for i,c in enumerate(ids):regional.append(dict(blood=blood,method=method,scenario='forward',case=int(c),rep=-1,layout=0,vessel=v+1,**dict(zip(SPECIES,ev[i]))))
   for i,c in enumerate(ids):
    rows.append(dict(blood=blood,method=method,scenario='forward',case=int(c),rep=-1,layout=0,noise='known_inlet',inlet=0.,**dict(zip(SPECIES,e[i]))))
   d=dict(np.load(folder/f'prediction-{method}.npz'));fit=dict(np.load(folder/f'test-{method}.npz'))
   e=errors(d['y'],ref['y'][d['cases']],ref['volume']);minima.append(dict(blood=blood,method=method,scenario='inverse',**dict(zip(SPECIES,d['y'].min((0,2,3))))))
   exact=ref['amplitudes'][d['cases']]@basis;pred=fit['amplitudes']@basis
   ie=np.linalg.norm(pred-exact,axis=1)/np.linalg.norm(exact,axis=1)
   curves[f'{blood}-{method}-inverse']=sites(d['y']).reshape(16,4,3,3,3,len(t))
   for v in range(3):
    sl=slice(cuts[v],cuts[v+1]);ev=errors(d['y'][:,:,sl],ref['y'][d['cases']][:,:,sl],ref['volume'][:,sl])
    for i,c in enumerate(d['cases']):regional.append(dict(blood=blood,method=method,scenario='inverse',case=int(c),rep=int(d['reps'][i]),layout=int(d['layouts'][i]),vessel=v+1,**dict(zip(SPECIES,ev[i]))))
   chosen=fit['chosen'];pg=fit['all_pg'][np.arange(len(chosen)),chosen];reason=fit['all_reason'][np.arange(len(chosen)),chosen]
   for i,c in enumerate(d['cases']):
    rows.append(dict(blood=blood,method=method,scenario='inverse',case=int(c),rep=int(d['reps'][i]),layout=int(d['layouts'][i]),noise='clean' if d['reps'][i]==0 else 'noisy',inlet=float(ie[i]),**dict(zip(SPECIES,e[i]))))
   optimizer.append(dict(blood=blood,method=method,fits=len(pg),maximum_projected_gradient=float(pg.max()),median_projected_gradient=float(np.median(pg)),projected_gradient_count=int((reason=='projected_gradient').sum()),stagnation_count=int((reason=='roundoff_or_stagnation').sum()),iteration_limit_count=int((reason=='iteration_limit').sum())))
   savecsv(OUT/f'optimizer-{blood}-{method}.csv',[dict(case=int(c),rep=int(r),layout=int(l),projected_gradient=float(g),termination=str(z)) for c,r,l,g,z in zip(d['cases'],d['reps'],d['layouts'],pg,reason)])
 savecsv(OUT/'candidate-minima.csv',minima);savecsv(OUT/'optimizer-summary.csv',optimizer);savecsv(OUT/'per-case-errors.csv',rows);savecsv(OUT/'per-vessel-errors.csv',regional);np.savez_compressed(RUN/'report-curves.npz',**curves,truth=sites(ref['y'][ids]),t=t)
 return rows,curves
def aggregate(rows):
 summary=[];paired=[];impact=[]
 settings=[('forward',0,'known_inlet')]+[('inverse',layout,noise) for noise in ['clean','noisy'] for layout in [1,2,3]]
 rng=np.random.default_rng(2026102901);boot=rng.integers(0,16,(10000,16))
 for blood in ['cfd','sen']:
  for scenario,layout,noise in settings:
   rr=[r for r in rows if r['blood']==blood and r['scenario']==scenario and r['layout']==layout and r['noise']==noise]
   for family in ['C0','neural']:
    ss=[r for r in rr if (r['method']=='ordinary_1d')==(family=='C0')]
    summary.append(dict(blood=blood,scenario=scenario,layout=layout,noise=noise,method=family,**{key:float(np.mean([r[key] for r in ss])) for key in ['inlet',*SPECIES]}))
   for key in ['inlet',*SPECIES]:
    delta=[]
    for c in range(32,48):
     cr=[r for r in rr if r['case']==c];c0=np.mean([r[key] for r in cr if r['method']=='ordinary_1d']);nn=np.mean([r[key] for r in cr if r['method']!='ordinary_1d']);delta.append(c0-nn)
    delta=np.array(delta);ci=np.quantile(delta[boot].mean(1),[.025,.975])
    paired.append(dict(blood=blood,scenario=scenario,layout=layout,noise=noise,metric=key,mean_c0_minus_neural=float(delta.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),neural_wins_cases=int((delta>0).sum()),test_cases=16))
 for scenario,layout,noise in settings:
  for method in ['C0','neural']:
   aa=[r for r in summary if r['scenario']==scenario and r['layout']==layout and r['noise']==noise and r['method']==method]
   a=next(r for r in aa if r['blood']=='cfd');b=next(r for r in aa if r['blood']=='sen')
   impact.append(dict(scenario=scenario,layout=layout,noise=noise,method=method,**{k:b[k]-a[k] for k in ['inlet',*SPECIES]}))
 savecsv(OUT/'summary.csv',summary);savecsv(OUT/'paired-comparison.csv',paired);savecsv(OUT/'blood-substitution-impact.csv',impact)
 return summary,paired,impact
def convergence(ref):
 rows=[]
 for tag in ['chem-refined','chem-flow-refined','chem-time-refined','chem-sampling-refined','chem-wall-refined']:
  d=readref(RUN/tag/'data.npz');ids=d['case_ids'];e=errors(ref['y'][ids],d['y'],d['volume'])
  for i,c in enumerate(ids):rows.append(dict(comparison=tag,case=int(c),**dict(zip(SPECIES,e[i]))))
 savecsv(OUT/'reference-convergence.csv',rows)
 return rows
def profile_diagnostic(ref):
 alt=readref(RUN/'chem-parabolic-diagnostic/data.npz');rows=[]
 for method in METHODS:
  y=np.load(RUN/f'evaluation-cfd/forward-{method}.npz')['y']
  for name,target in [('solved_CFD_profile',ref),('parabolic_at_identical_CFD_AQ',alt)]:
   ee=errors(y,target['y'][32:48],target['volume']).mean(0)
   rows.append(dict(reference_profile=name,method=method,**dict(zip(SPECIES,ee))))
 savecsv(OUT/'radial-profile-diagnostic.csv',rows)
 z=np.load(RUN/'blood-main/flow.npz');t=z['t'];nr=z['v0_radial_axial_velocity'].shape[-1]
 se=np.linspace(0,1,nr+1);w=np.diff(se**2);pr=2-se[:-1]**2-se[1:]**2;r=np.sqrt(.5*(se[:-1]**2+se[1:]**2))
 fig,axes=plt.subplots(3,3,figsize=(11,9))
 for v in range(3):
  for col,tt in enumerate([.1,.2,.4]):
   j=np.argmin(abs(t-tt));ix=len(z[f'v{v}_xf'])//2;u=z[f'v{v}_radial_axial_velocity'][j,ix];mean=u@w
   ax=axes[v,col];ax.plot(r,u,color='black',label='Solved CFD profile');ax.plot(r,mean*pr,'--',color='#b16a31',label='Parabola with same mean')
   ax.set_title(f'{["Parent","Daughter 1","Daughter 2"][v]}, {tt:.1f} s');ax.set_xlabel('Radius / wall radius');ax.set_ylabel('Axial velocity (m/s)');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'radial-profiles.png',dpi=160);plt.close(fig)
 return rows
def blood():
 c,t,stats=load_flow(str(RUN/'blood-main'),tuple(COUNTS),1,.002)
 cf,tf,sf=load_flow(str(RUN/'blood-refined'),tuple(COUNTS),1,.002)
 sen,_=flow();raw=np.load(ROOT/'runs/sen-input/input.npz')
 records=[];fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True)
 for v,n in enumerate(COUNTS):
  for metric in ['area','q','velocity']:
   if metric=='area':truth=c[v]['area'];sp=sen[v]['area'];fine=cf[v]['area']
   elif metric=='q':truth=c[v]['q'];sp=sen[v]['q'];fine=cf[v]['q']
   else:truth=c[v]['q']/c[v]['af'];sp=sen[v]['q']/sen[v]['af'];fine=cf[v]['q']/cf[v]['af']
   records.append(dict(vessel=v+1,metric=metric,comparison='Sen projected vs independent CFD',relative_l2=float(np.linalg.norm(sp-truth)/np.linalg.norm(truth))))
   rawsp=sen[v]['area'] if metric=='area' else sen[v]['qraw'] if metric=='q' else sen[v]['qraw']/sen[v]['af']
   records.append(dict(vessel=v+1,metric=metric,comparison='Sen raw vs independent CFD projected',relative_l2=float(np.linalg.norm(rawsp-truth)/np.linalg.norm(truth))))
   if metric=='q':
    records.append(dict(vessel=v+1,metric=metric,comparison='Sen conservation projection change',relative_l2=float(np.linalg.norm(sen[v]['q']-sen[v]['qraw'])/np.linalg.norm(sen[v]['qraw']))))
   records.append(dict(vessel=v+1,metric=metric,comparison='Main vs refined independent CFD',relative_l2=float(np.linalg.norm(truth-fine)/np.linalg.norm(fine))))
  for col,(key,scale,label) in enumerate([('area',1e6,'Area (mm²)'),('q',1e6,'Flow (mL/s)'),('velocity',1.,'Mean velocity (m/s)')]):
   ax=axes[v,col]
   for d,color,name in [(c,'black','Independent CFD'),(sen,'#367db5','Sen projected')]:
    y=d[v]['area'] if key=='area' else d[v]['q'] if key=='q' else d[v]['q']/d[v]['af']
    x=d[v]['x'] if key=='area' else d[v]['xf'];val=np.array([np.interp(LENGTH[v]/2,x,z) for z in y])
    ax.plot(t,val*scale,color=color,label=name)
   if key!='area':
    vv=sen[v]['qraw'] if key=='q' else sen[v]['qraw']/sen[v]['af']
    mid=np.array([np.interp(LENGTH[v]/2,sen[v]['xf'],z) for z in vv])
    ax.plot(t,mid*scale,':',color='#a65f91',label='Sen before conservation projection')
   source=raw[f'ref{v+1}_2'];st=raw['t'];vals=source[:,{'area':5,'q':4,'velocity':3}[key]]
   val=np.interp(t%.82,st,vals);ax.plot(t,val*scale,'--',color='#ac7b30',label='Original numerical archive')
   ax.set_title(['Parent','Daughter 1','Daughter 2'][v]+' — '+label);ax.grid(alpha=.2)
   # Approximation error against original numerical source at the same recorded midpoint.
   y=sen[v]['area'] if key=='area' else sen[v]['qraw'] if key=='q' else sen[v]['qraw']/sen[v]['af']
   x=sen[v]['x'] if key=='area' else sen[v]['xf'];mid=np.array([np.interp(LENGTH[v]/2,x,z) for z in y])
   records.append(dict(vessel=v+1,metric=key,comparison='Sen raw vs original numerical midpoint',relative_l2=float(np.linalg.norm(mid-val)/np.linalg.norm(val))))
 axes[0,0].legend(fontsize=8);[ax.set_xlabel('Time (s)') for ax in axes[-1]];fig.tight_layout();fig.savefig(OUT/'blood-comparison.png',dpi=160);plt.close(fig)
 savecsv(OUT/'blood-errors.csv',records)
 rawcheck={}
 for tag in ['blood-main','blood-refined']:
  zz=np.load(RUN/tag/'flow.npz');tt=zz['t'];dt=tt[1]-tt[0]
  vv=sum(zz[f'v{v}_volume'].sum(1) for v in range(3))
  net=zz['v0_face_flow'][:,0]-zz['v1_face_flow'][:,-1]-zz['v2_face_flow'][:,-1]
  defect=vv-vv[0]-np.r_[0,np.cumsum(net[1:]*dt)]
  junction=zz['v0_face_flow'][:,-1]-zz['v1_face_flow'][:,0]-zz['v2_face_flow'][:,0]
  rawcheck[tag]=dict(raw_global_accumulated_volume_defect=float(abs(defect).max()/vv[0]),raw_junction_max_m3_s_after_initial=float(abs(junction[1:]).max()))
 result=dict(main_transfer=stats,refined_transfer=sf,unprojected_finite_element_checks=rawcheck)
 (OUT/'flow-transfer.json').write_text(json.dumps(result,indent=2))
 return records,result
def figures(ref,summary,curves):
 ids=np.arange(32,48);truth=sites(ref['y'][ids]);t=ref['t']
 for blood in ['cfd','sen']:
  fig,axes=plt.subplots(3,3,figsize=(12,9),sharex=True)
  for k,species in enumerate(SPECIES):
   for v in range(3):
    ax=axes[k,v];ax.plot(t,truth[0,k,v],color='black',label='Independent reference')
    ax.plot(t,curves[f'{blood}-ordinary_1d-forward'][0,k,v],color='#c76e24',label='Fixed C0')
    nn=np.stack([curves[f'{blood}-neural_{s}-forward'][0,k,v] for s in [42,43,44]])
    ax.plot(t,nn.mean(0),color='#167d9b',label='Frozen neural, mean seeds')
    ax.fill_between(t,nn.min(0),nn.max(0),color='#167d9b',alpha=.15)
    ax.set_title(f'{species} — {["Parent","Daughter 1","Daughter 2"][v]}');ax.grid(alpha=.2)
  axes[0,0].legend(fontsize=8);[ax.set_xlabel('Time (s)') for ax in axes[-1]];fig.suptitle(f'Prespecified test case 32; {blood.upper()} blood input');fig.tight_layout();fig.savefig(OUT/f'forward-{blood}.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True)
 for row,blood in enumerate(['cfd','sen']):
  for col,key in enumerate(['inlet',*SPECIES]):
   ax=axes[row,col]
   for method,color in [('C0','#c76e24'),('neural','#167d9b')]:
    rr=[r for r in summary if r['blood']==blood and r['scenario']=='inverse' and r['noise']=='noisy' and r['method']==method]
    ax.plot([r['layout'] for r in rr],[100*r[key] for r in rr],'-o',color=color,label=method)
   ax.set_title(f'{blood.upper()} blood: {key}');ax.set_ylabel('Relative L2 error (%)');ax.grid(alpha=.2);ax.set_xticks([1,2,3]);ax.set_xlabel('Number of sites')
 axes[0,0].legend();fig.tight_layout();fig.savefig(OUT/'inverse-comparison.png',dpi=160);plt.close(fig)
def table(rows,keys):
 head='| '+' | '.join(keys)+' |';bar='| '+' | '.join(['---']*len(keys))+' |'
 out=[head,bar]
 for r in rows:out.append('| '+' | '.join(f'{100*r[k]:.2f}%' if k in ['inlet',*SPECIES] else str(r[k]) for k in keys)+' |')
 return '\n'.join(out)
def render_markdown(text):
 def fmt(z):return re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',html.escape(z))
 lines=text.splitlines();out=[];i=0
 while i<len(lines):
  s=lines[i]
  if not s.strip():i+=1;continue
  if s.startswith('#'):
   n=len(s)-len(s.lstrip('#'));out.append(f'<h{n}>'+fmt(s[n:].strip())+f'</h{n}>');i+=1;continue
  if s.startswith('|'):
   rr=[]
   while i<len(lines) and lines[i].startswith('|'):
    vals=[v.strip() for v in lines[i].strip('|').split('|')]
    if not all(set(v)<=set('-: ') for v in vals):rr.append(vals)
    i+=1
   out.append('<table><thead><tr>'+''.join('<th>'+fmt(v)+'</th>' for v in rr[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+fmt(v)+'</td>' for v in row)+'</tr>' for row in rr[1:])+'</tbody></table>');continue
  if s.startswith('- '):
   out.append('<ul>')
   while i<len(lines) and lines[i].startswith('- '):out.append('<li>'+fmt(lines[i][2:])+'</li>');i+=1
   out.append('</ul>');continue
  pp=[]
  while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','- ')):pp.append(lines[i]);i+=1
  out.append('<p>'+fmt(' '.join(pp))+'</p>')
 return '\n'.join(out)
def main():
 OUT.mkdir(exist_ok=True);ref=readref(RUN/'chem-main/data.npz')
 rows,curves=collect(ref);summary,paired,impact=aggregate(rows);conv=convergence(ref);br,transfer=blood();diag=profile_diagnostic(ref);figures(ref,summary,curves)
 protected=json.loads((RUN/'protected-files.json').read_text());changed=[p for p,h in protected.items() if digest(ROOT/p)!=h]
 assert not changed,changed
 integrity=dict(protected_file_count=len(protected),changed_files=changed,checkpoint_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted(ROOT.glob('runs/closure-final-*/model.pt'))})
 (OUT/'integrity.json').write_text(json.dumps(integrity,indent=2))
 protocol=json.loads((RUN/'protocol.json').read_text());(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 bench=json.loads((RUN/'cfd-cpu-gpu-benchmark.json').read_text())
 uniform=json.loads((RUN/'chem-uniform/metrics.json').read_text());chem=json.loads((RUN/'chem-main/metrics.json').read_text())
 frows=[r for r in summary if r['scenario']=='forward'];irows=[r for r in summary if r['scenario']=='inverse' and r['noise']=='noisy']
 fc0=next(r for r in frows if r['blood']=='cfd' and r['method']=='C0')
 fnn=next(r for r in frows if r['blood']=='cfd' and r['method']=='neural')
 snn=next(r for r in frows if r['blood']=='sen' and r['method']=='neural')
 text=f"""# Independent three-vessel CFD blood validation

## Main result

With common independent CFD blood, the frozen neural models improve known-inlet prediction: Th error falls from {100*fc0['Th']:.2f}% to {100*fnn['Th']:.2f}%, Fg from {100*fc0['Fg']:.2f}% to {100*fnn['Fg']:.2f}%, and Fn from {100*fc0['Fn']:.2f}% to {100*fnn['Fn']:.2f}%.

However, the current frozen inverse setup recovers the inlet less accurately than C0 at all three sensor counts. Its Fn predictions remain better. This supports a concentration-prediction advantage on this test, not an overall claim that the neural method outperforms C0 for every task.

Substituting Sen blood increases the neural known-inlet Fn error from {100*fnn['Fn']:.2f}% to {100*snn['Fn']:.2f}%. The much larger change from the old results is associated with using solved radial velocity profiles instead of a prescribed parabola; the controlled profile diagnostic below tests this directly.

## What changed

This additive experiment replaces the former Sen-driven chemical reference with an independently solved blood field. Three axisymmetric, compliant tubes solve the ALE Navier–Stokes equations. Their open junction sections share one normal traction, whose Lagrange multiplier imposes exact net flow balance. The junction has zero volume and biochemical concentrations mix perfectly.

The boundary and initial data come from the original numerical three-vessel archive, not Sen predictions. External endpoint mean-velocity waveforms are prescribed, with periodic continuation after 0.82 s. Tube length, reference area and wall stiffness are unchanged. This is a coupled axisymmetric tube reference, not a three-dimensional Y-shaped bifurcation simulation.

No neural network was trained. The three existing biochemical checkpoints and the fixed C0 equations were retained.

## What was trained previously, and what is frozen now

The existing Sen three-vessel blood model was trained separately before this experiment. The biochemical graph model then learned two modules on the three-vessel prescribed-profile data: a transport memory-flux closure and a reaction-source closure.

The biochemical split is 24 complete inlet cases for training, 8 for validation and 16 for held-out testing. Each module used 3,000 main Adam iterations (3,004 optimizer updates including setup), with checkpoints selected by validation flux/source error. Training was staged: transport first, reaction second. The network was not trained directly to minimize inlet-recovery error. Training features used reference concentration fields; deployment obtains them from the one-dimensional rollout.

In this experiment all learned parameter tensors are frozen. The inverse optimizer adjusts only seven inlet-amplitude parameters per case. The new independent CFD data are never used to update network weights.

## Two controlled comparisons

1. **Common independent blood:** C0 and all frozen networks receive exactly the same CFD-derived area and mean flow. Both are compared against the same radially resolved biochemical reference.
2. **Blood substitution:** both candidates receive the original projected Sen area and flow. The independent biochemical reference, observations, noise, inverse regularization and fit settings remain identical.

The second-minus-first change measures the impact of substituting blood fields on this experiment. It is not an additive decomposition of error. Sen-versus-CFD discrepancy includes both neural reconstruction error and differences between the original one-dimensional blood model and axisymmetric CFD. A separate table compares raw Sen predictions with the original numerical midpoint samples.

## Blood discrepancy and the existing conservation projection

The Sen mean-flow relative L2 errors against the independent CFD are:

| Vessel | Raw Sen Q | Projected Sen Q used by both candidates |
| --- | --- | --- |
| Parent | 2.24% | 5.08% |
| Daughter 1 | 1.61% | 5.60% |
| Daughter 2 | 4.40% | 23.24% |

The projection reconciles the predicted area changes and branch flows with volume/junction conservation. It changes the smaller daughter branch substantially. This separates an additional projection effect from raw blood-network approximation error. The existing projection is retained in this experiment; C0 and the neural candidates always share its same output in the Sen-input arm. The original-source midpoint comparison is separately reported in blood-errors.csv.

## Known-inlet prediction

Volume-weighted space-time relative L2 error over 0–1.23 s, 16 held-out chemical inlet cases. Neural values average the three frozen seeds; no seed is selected on these results.

{table(frows,['blood','method',*SPECIES])}

## Sparse inversion

All fits observe **Th and Fg only**; Fn is an unobserved prediction. Sites are parent 30 mm, daughter 1 10 mm, daughter 2 17 mm, added in that order. Every site contributes 24 time samples through 1.23 s. Each case has a clean fit and three fixed noise realizations (absolute standard deviation 0.02). The following table averages the noisy fits.

Inlet error uses the agreed 0–0.82 s interval. Concentration errors use 0–1.23 s. All methods and blood inputs use the **independent reference volume weights**, preventing the scoring weights from changing with a candidate's blood field.

{table(irows,['blood','layout','method','inlet',*SPECIES])}

Clean-fit results are in summary.csv. Per-case errors, all seed results and optimizer termination diagnostics are provided separately. Paired uncertainty in paired-comparison.csv bootstraps the 16 inlet cases after averaging noise and seeds within each case; noise replicates are not treated as independent cases.

Inverse regularization is locked to the previous validation selection: C0 0.0001, neural seeds 0. The same three starts, bounds [0,1.2], maximum 100 iterations and projected-gradient tolerance 1e-8 apply to both blood inputs. There is no new tuning on the independent CFD test cases.

## Why the independent reference is harder

A diagnostic keeps the CFD area/mean flow identical but replaces the solved radial velocity profile with the former prescribed parabola. The candidate predictions are unchanged; only this diagnostic reference profile changes. This prescribed-profile calculation is not used as ground truth for the primary evaluation.

{table(diag,['reference_profile','method',*SPECIES])}

The difference tests sensitivity to the radial-profile assumption. Th has no reaction sink in these equations, so its degradation already identifies a transport-model mismatch. Fn additionally depends on the transported Th/Fg fields and their reaction covariance; its error cannot all be assigned to the reaction closure. The frozen network was fitted to prescribed-profile reference data and was not trained to accommodate the independently solved velocity profiles.

There is also a structural restriction in the frozen transport closure: its memory-flux amplitude is proportional to the absolute cross-section-mean velocity. At zero mean flow this correction vanishes, even though forward core flow and backward near-wall flow can coexist in the independent CFD. The same mean flow therefore need not imply the same unresolved solute flux. No change to that architecture or its parameters was made here.

These results apply to the current frozen model and its prescribed-profile training data. Improving the model is outside this frozen evaluation.

## Reference checks

{table(conv,['comparison','case',*SPECIES])}

chem-refined doubles axial/radial transport resolution with the same CFD blood and time step. chem-flow-refined changes the blood finite-element mesh and blood time step while retaining the reference chemical discretization. chem-time-refined additionally halves the chemical time step. chem-sampling-refined repeats the fine chemical mesh after independently resampling the saved finite-element velocity into 96 radial bins (rather than subdividing the original 24-bin velocity). chem-wall-refined uses a radially graded blood mesh concentrated near the wall. These comparisons use prespecified cases 0, 32 and 47; case 0 is a numerical check, not a new training sample.

Uniform concentration preservation error: {uniform['uniform_max_error']:.3e}. Maximum chemical mass-balance residual, normalized by network volume: {chem['mass_balance_max']:.3e}. The CUDA linear solver is checked against a CPU sparse direct solve and its residual is checked at every step.

Interpolated finite-element flow is transferred conservatively to the chemical mesh. Corrections and exact continuity residuals are saved in flow-transfer.json. This projection is applied to the independent CFD field, not learned from biochemical targets.

## Compute and preservation

On the identical 4,120-order CFD system, warm CPU sparse-LU median solve time was {1000*bench['cpu_median_seconds']:.2f} ms; the tested CUDA sparse-QR implementation took {1000*bench['gpu_median_seconds']:.2f} ms. Their relative solution difference was {bench['relative_solution_difference']:.3e}. The current CPU implementation therefore remains the blood linear solver. This benchmark does not claim that every GPU sparse solver is slower.

Resolved chemistry runs batched on CUDA, as do frozen neural inference and inverse forward/Jacobian evaluations. GPU utilization logs are retained with each run. Matrix assembly and small optimization control steps use the CPU.

The main prediction/inverse suite finished before the slower reference-refinement checks. Those checks dominated elapsed time; the sparse-solver microbenchmark covers only the linear solve, not finite-element assembly or interpolation.

All {len(protected)} protected pre-existing files are unchanged. Frozen parameter tensors are checked when loading; flow-dependent feature buffers are recomputed from the supplied blood. With the original blood, the new adapter reproduces all four previous solvers to approximately 1e-15.

## Interpretation limits

- This validates one three-vessel geometry and one external blood driving condition, with 16 held-out biochemical inlet waveforms. It does not establish population-wide blood-flow generalization.
- The reference omits a geometric bifurcation angle, local junction vortices, junction volume and detailed wall mechanics.
- Biochemical concentrations are synthetic solutions of the specified transport/reaction equations, not measured Th/Fg/Fn data.
- The initial axial velocity profile is assumed parabolic, with a divergence-compatible radial initial velocity reconstructed from the raw one-dimensional fields. There is no additional periodic CFD spin-up; this is the declared initial-value experiment.
- The blood solver uses raw numerical initial states and measured-in-the-archive endpoint waveforms; it is independent of the trained Sen model, not independent of the original numerical problem definition.
- Existing Sen-conditional results remain available and have not been replaced.

## Files

- summary.csv: complete clean/noisy aggregate tables
- per-case-errors.csv: all forward and inverse errors
- per-vessel-errors.csv: separate Th/Fg/Fn errors for each vessel
- paired-comparison.csv: paired case-level uncertainty
- blood-substitution-impact.csv: Sen-input minus CFD-input error, for each method
- blood-errors.csv: blood reconstruction and refinement errors
- reference-convergence.csv: biochemical reference checks
- optimizer-*.csv: selected-fit convergence diagnostics
- protocol.json, integrity.json: frozen protocol and preservation audit

## Reproduce

See README.md in this report directory. Open curves.html for all test cases, noise replicates and sensor layouts. The original experiment remains under reports/all-species-v3.
"""
 (OUT/'REPORT.md').write_text(text)
 images=['radial-profiles.png','blood-comparison.png','forward-cfd.png','forward-sen.png','inverse-comparison.png']
 lead,detail=text.split('## What changed',1)
 body=render_markdown(lead)+'<p><a href="curves.html">Interactive concentration curves: every case and inverse layout</a></p><h2>Figures</h2>'
 for f in images:body+=f'<figure><img src="data:image/png;base64,{base64.b64encode((OUT/f).read_bytes()).decode()}"><figcaption>{html.escape(f)}</figcaption></figure>'
 body+=render_markdown('## What changed'+detail)
 (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Three-vessel independent CFD</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px;color:#153044}img{width:100%}p,li{line-height:1.6}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{text-align:left;border-bottom:1px solid #d8e2e8;padding:9px}th{background:#e9f2f5}h2{margin-top:36px}figure{margin:20px 0}figcaption{color:#667}</style>'+body)
 print(json.dumps(dict(summary=summary,convergence=conv,protected_files=len(protected))),flush=True)
if __name__=='__main__':main()
