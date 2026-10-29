"""Create the unified baseline audit, paired metrics and figures."""
import json,csv,hashlib,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .protocol import ROOT,RUN,OUT,SEEDS,TEST,plan
from .evaluate import reference
from .models import sample_weights
from .reference import flow
from .protocol import COUNTS
from gp52.three_independent.report import render_markdown
SPECIES=['Th','Fg','Fn']
def csvwrite(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def collect():
 rows=[];optimizer=[];selections=[]
 for name in ['single','three','seven']:
  folder=RUN/name/'evaluation'
  for p in folder.glob('forward-*.npz'):
   method=p.stem[8:];z=np.load(p);family=method.split('_')[0]
   for c,e in zip(z['case_ids'],z['errors']):
    rows.append(dict(network=name,setting='forward',family=family,method=method,case=int(c),rep=-1,sites=0,inlet=None,**dict(zip(SPECIES,e))))
  for p in folder.glob('prediction-*.npz'):
   method=p.stem[11:];z=np.load(p);family=method.split('_')[0]
   for i,e in enumerate(z['errors']):
    rows.append(dict(network=name,setting='inverse',family=family,method=method,case=int(z['cases'][i]),rep=int(z['reps'][i]),sites=int(z['layouts'][i]),inlet=float(z['inlet_errors'][i]),**dict(zip(SPECIES,e))))
  for p in folder.glob('test-*.npz'):
   z=np.load(p);ii=np.arange(len(z['cases']));ch=z['chosen'];pg=z['all_pg'][ii,ch];reason=z['all_reason'][ii,ch]
   optimizer.append(dict(network=name,method=p.stem[5:],fits=len(ii),pg_max=float(pg.max()),pg_over_1e8=int(sum(pg>1e-8)),
    iteration_limits=int(sum(reason=='iteration_limit')),stagnated=int(sum(reason=='roundoff_or_stagnation')),seconds=float(z['seconds']),
    first_start_cost_gap_max=float(np.max(z['all_cost'][:,0]-z['all_cost'].min(1))),
    start_amplitude_difference_max=float(np.max(abs(z['all_amplitudes']-z['all_amplitudes'][:,0:1])))))
  for p in folder.glob('selection-*.json'):
   d=json.loads(p.read_text());selections.append(dict(network=name,method=p.stem[10:],lambda_selected=d['selected']['lam'],validation_inlet_error=d['selected']['error']))
 csvwrite('per-case-errors.csv',rows);csvwrite('optimizer-summary.csv',optimizer);csvwrite('inverse-selection.csv',selections)
 summary=[];paired=[]
 for name in ['single','three','seven']:
  for setting in ['forward','inverse']:
   for noise in (['known'] if setting=='forward' else ['clean','noisy']):
    for sites in ([0] if setting=='forward' else [1,2,3]):
     sub=[r for r in rows if r['network']==name and r['setting']==setting and r['sites']==sites and (setting=='forward' or (r['rep']==0 if noise=='clean' else r['rep']>0))]
     case_metrics={}
     for fam in sorted(set(r['family'] for r in sub)):
      rr=[r for r in sub if r['family']==fam];keys=SPECIES if setting=='forward' else ['inlet']+SPECIES
      bycase=np.array([[np.mean([r[k] for r in rr if r['case']==int(c)]) for k in keys] for c in TEST]);case_metrics[fam]=bycase
      summary.append(dict(network=name,setting=setting,noise=noise,sites=sites,family=fam,inlet=None if setting=='forward' else float(bycase[:,0].mean()),
       **{k:float(bycase[:,keys.index(k)].mean()) for k in SPECIES}))
     diff=case_metrics['new']-case_metrics['C0'];rng=np.random.default_rng(52703103)
     boot=diff[rng.integers(0,len(TEST),(3000,len(TEST)))].mean(1)
     for j,k in enumerate(keys):
      paired.append(dict(network=name,setting=setting,noise=noise,sites=sites,metric=k,
       difference_pp=100*float(diff[:,j].mean()),low_pp=100*float(np.quantile(boot[:,j],.025)),high_pp=100*float(np.quantile(boot[:,j],.975)),
       cases_favoring_new=int(sum(diff[:,j]<0)),cases=len(TEST)))
 csvwrite('summary.csv',summary);csvwrite('paired.csv',paired)
 return summary,optimizer
def figures(summary):
 colors={'C0':'#c77531','old':'#8895a4','new':'#13868a'}
 fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
 for ax,name in zip(axs,['single','three','seven']):
  x=np.arange(3)
  families=[f for f in ['C0','old','new'] if any(r['network']==name and r['setting']=='forward' and r['family']==f for r in summary)]
  for k,fam in enumerate(families):
   r=next(r for r in summary if r['network']==name and r['setting']=='forward' and r['family']==fam)
   ax.bar(x+(k-(len(families)-1)/2)*.25,[100*r[s] for s in SPECIES],.24,label={'C0':'Fixed C0','old':'Old weights','new':'Retrained baseline'}[fam],color=colors[fam])
  ax.set_xticks(x,SPECIES);ax.set_title(name.capitalize()+' vessel'+('s' if name!='single' else ''));ax.set_ylabel('Relative L2 error (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axs[0].legend(fontsize=8);fig.savefig(OUT/'forward.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(3,2,figsize=(10,10),layout='constrained')
 for row,name in enumerate(['single','three','seven']):
  for col,metric in enumerate(['inlet','Fn']):
   ax=axs[row,col]
   for fam in ['C0','new']:
    rr=[next(r for r in summary if r['network']==name and r['setting']=='inverse' and r['noise']=='noisy' and r['sites']==n and r['family']==fam) for n in [1,2,3]]
    ax.plot([1,2,3],[100*r[metric] for r in rr],'o-',color=colors[fam],label=fam)
   ax.set_title(f'{name.capitalize()}: {metric}');ax.set_xlabel('Measurement sites');ax.set_xticks([1,2,3]);ax.set_ylabel('Relative L2 error (%)');ax.grid(alpha=.2);ax.legend()
 fig.savefig(OUT/'inverse.png',dpi=180);plt.close(fig)
def curve_data():
 from .protocol import inputs,KNOTS
 curves={'inlet-truth':inputs()[TEST],'knots':KNOTS}
 for name in ['single','three','seven']:
  z=reference(name);blood,t,_=flow(name,tuple(COUNTS[name]),1,.002);x=np.concatenate([d['x'] for d in blood]);wx,wt=sample_weights(name,x,t)
  def samp(y):return np.einsum('pn,bsnt->bspt',wx,y,optimize=True)
  curves[name+'-truth']=samp(z['y'][TEST]);curves['t']=t
  for method,key in [('C0','c0'),('new_42','nn')]:
   folder=RUN/name/'evaluation';f=np.load(folder/f'forward-{method}.npz');iv=np.load(folder/f'prediction-{method}.npz')
   curves[name+'-'+key+'-forward']=samp(f['y'])
   out=np.empty((32,4,3,3,3,len(t)))
   sampled=samp(iv['y']);fits=np.load(folder/f'test-{method}.npz');ia=np.empty((32,4,3,7))
   for i in range(len(sampled)):
    idx=(int(iv['cases'][i])-80,int(iv['reps'][i]),int(iv['layouts'][i])-1)
    out[idx]=sampled[i];ia[idx]=fits['amplitudes'][i]
   curves[name+'-'+key+'-inlet']=ia
   curves[name+'-'+key+'-inverse']=out
 np.savez_compressed(RUN/'report-curves.npz',**curves)
def table(summary,name,setting,noise='noisy'):
 rows=[r for r in summary if r['network']==name and r['setting']==setting and (setting=='forward' or r['noise']==noise)]
 cols=['Method','Th','Fg','Fn'] if setting=='forward' else ['Sites','Method','Inlet','Th','Fg','Fn']
 text='| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'
 for r in rows:
  label={'C0':'Fixed C0','new':'Retrained baseline (3 seeds)','old':'Old weights'}[r['family']]
  vals=[label] if setting=='forward' else [str(r['sites']),label]
  vals += [f'{100*r[k]:.2f}%' for k in (SPECIES if setting=='forward' else ['inlet']+SPECIES)]
  text+='| '+' | '.join(vals)+' |\n'
 return text
def main():
 assert (RUN/'evaluation-complete.json').exists();OUT.mkdir(exist_ok=True)
 summary,optimizer=collect();figures(summary);curve_data()
 protected=json.loads((RUN/'protected.json').read_text());changed=[p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
 assert not changed,changed
 integrity=dict(protected_files=len(protected),changed_files=changed)
 (OUT/'integrity.json').write_text(json.dumps(integrity,indent=2))
 checks=json.loads((RUN/'reference-validation.json').read_text())
 blood_audit=json.loads((RUN/'blood-audit.json').read_text());(OUT/'blood-audit.json').write_text(json.dumps(blood_audit,indent=2))
 (OUT/'reference-validation.json').write_text(json.dumps(checks,indent=2))
 cfg=plan();(OUT/'protocol.json').write_text(json.dumps(cfg,indent=2))
 train=[]
 for name in ['single','three','seven']:
  for p in (RUN/name).glob('train-*/**/metrics.json'):
   d=json.loads(p.read_text());gpu=p.parent/'gpu.csv';vals=[]
   if gpu.exists():
    for line in gpu.read_text().splitlines():
     fields=line.split(',')
     if len(fields)==4:
      try:vals.append([float(z) for z in fields[1:]])
      except ValueError:pass
   v=np.array(vals);train.append(dict(network=name,path=str(p.relative_to(ROOT)),seconds=d.get('seconds_total',d.get('seconds')),
    gpu_mean=float(v[:,0].mean()) if len(v) else None,gpu_max=float(v[:,0].max()) if len(v) else None,memory_peak_MiB=float(v[:,1].max()) if len(v) else None))
 csvwrite('training-runs.csv',train)
 text="""# Unified independent-CFD neural baselines

## Scope and protocol

This experiment fixes the data-generation, partition and evaluation protocol before designing a new model. PROTOCOL_NOTES.md records the scope extension and numerical execution changes; the shared inlet arrays and partition IDs remain fixed. The existing single-vessel and three-vessel architectures are retained and freshly trained. The seven-vessel baseline reuses the three-vessel heads and losses, extending the adjacency and numerical operators to three junctions. C0 is the fixed conventional numerical comparison. The independent CFD calculations provide numerical reference data, not laboratory measurements.

All three topologies use exactly the same 112 seven-knot biochemical inlet curves: 64 training, 16 validation and 32 test cases. Every amplitude is sampled uniformly from 0.05 to 1.15. Splits contain complete network trajectories, not individual points, vessels or time samples. Training and validation alone determine weights and inverse regularization. Test labels are generated in advance but are excluded from all fitting and selection losses.

Each topology has separately trained weights, rather than one shared model across vessel counts. There is one independent blood-flow condition per topology. This tests new chemical inlet curves under fixed flow, not new-flow or new-patient generalization. The test cases are first scored after checkpoint selection in this run; once these results are inspected they must not be called a fresh blind test for subsequent model design.

## Shared reference generation

Blood comes from the existing independently solved compliant-wall axisymmetric ALE Navier–Stokes calculations. The single-vessel reference uses the verified raw-input reproduction; the three-vessel reference uses the refined independent coupled calculation. Seven-vessel blood is freshly solved from the original numerical inlet/outlet archives, with seven axisymmetric tubes coupled by three common-traction constraints. None of the biochemical references uses Sen-predicted blood. The blood solvers retain their declared initialization and prescribed external velocity conditions; these are not full three-dimensional bifurcations or periodic-steady clinical flow fields.

All three use the same conservative cross-section-resolved finite-volume biochemical generator, prescribed base diffusivity 2e-4 and reaction rate 2 Th Fg. This is a computational baseline with fixed chemical parameters; the errors measure agreement with that simulator. Concentrations and inlet amplitudes use the existing normalized model units, not measured biochemical concentrations. Th is catalytic; Fg is consumed and immobile Fn accumulates. There is no Fn-to-blood feedback. The upstream concentration reservoir stays connected; distal inflow has Th=0 and Fg=1. Junctions are perfectly mixed and have zero volume.

Reference grids use four axial cells per candidate cell, 24 radial cells and a 1 ms time step. Physical reaction means and advective flux covariance labels are obtained from the resolved radial states. A least-squares blood-flow transfer enforces the same discrete volume conservation in reference and candidate grids. Every compared method within a topology receives identical independent area/flow arrays.

## What remains unchanged

The single-vessel baseline retains its two-channel advected flux memory, local graph transport head and graph reaction predictor-corrector. The three-vessel baseline retains its original graph adjacency, two local memory channels, mean-speed-scaled transport amplitude and graph reaction head. No architecture limitation is repaired in this experiment.

The original training objectives remain topology-specific: single-vessel trajectory training and source-plus-trajectory reaction training; three-vessel physical flux/source supervision. Source hashes are recorded in protocol.json. Three fresh random seeds are trained per topology. These are three independent fits, not an ensemble used to improve the reported predictions.

Single-vessel training uses shuffled batches of 16 from the 64 training cases, with the original 1,500 transport steps, 1,000 reaction source steps and 500 reaction trajectory steps. Three source-loss weights are selected on validation only. Three- and seven-vessel training use 3,000 main steps per head on all 64 cases. CUDA warm-up updates are retained where used by the original source-training procedure.

C0 has no memory or fitted closure. Its existing per-topology temporal discretizations are retained: coupled implicit reaction in the single vessel and transport followed by exact local reaction in three and seven vessels. Both use the same physical equations and 2 ms candidate grid. This protocol does not silently replace the locked C0 implementations or claim bitwise-identical temporal schemes.

## Known-inlet prediction

Errors are per-case, reference-volume and trapezoidal-time weighted relative L2 norms over the whole network and 0–1.23 s, then averaged equally across test cases. Retrained neural errors are averaged across the three seeds. Old weights are replayed on the same new blood and reference for the forward comparison only; their historical results and files remain unchanged.
"""
 for name in ['single','three','seven']:
  text+=f'\n### {name.capitalize()}\n\n'+table(summary,name,'forward')
 text+="""
This is a harmonized baseline experiment, not a one-factor ablation of data provenance: the training case count and sampling range also change.

## Sparse inverse reconstruction

Only Th and Fg are observed, at 24 common times from 0.002 to 1.23 s. Fn is never observed. Layouts add the first one, two or three predeclared sites. Their distances along the network path from the upstream inlet are:

| Topology | First site | Second site | Third site |
| --- | --- | --- | --- |
| Single | 30 mm | 6 mm | 54 mm |
| Three | 30 mm | 70 mm | 77 mm |
| Seven | 30 mm | 99.5 mm | 141 mm |

The second single-vessel site is close to the inlet; the added branched-network sites are downstream. Comparisons between NN and C0 use identical sites within each topology. Absolute inverse errors across topologies also reflect different observation distances, geometry and blood flow, so they are not a controlled test of vessel count alone.

 Noise has absolute standard deviation 0.02 with three identical paired realizations across candidate methods. Clean results are also retained.

All methods fit seven inlet amplitudes bounded in [0,1.2], using the same objective normalization, three starts, bounded optimization and stopping criteria. Each method chooses from lambda={0,0.0001,0.01} using validation inlet error only. Inlet error is scored over 0–0.82 s; all three concentration errors use 0–1.23 s. Network weights are frozen during every inverse fit. Final inference uses FP64 arithmetic, including the reaction head and nonlinear reaction features. For the branched networks, sparse adjacency application evaluates the same fixed graph operator without dense zero multiplications; numerical parity is checked. Preliminary FP32 three-vessel validation exposed gradient stagnation and is retained separately as a numerical diagnostic, not included in the final results.

The following are noisy means. The complete clean/noisy and per-case results are in summary.csv and per-case-errors.csv.
"""
 for name in ['single','three','seven']:text+=f'\n### {name.capitalize()}\n\n'+table(summary,name,'inverse')
 text+="""
## Interpretation and uncertainty

Concentration prediction, inlet recovery and unobserved Fn reconstruction are distinct outcomes. A lower Fn error does not establish better inlet recovery. Paired intervals in paired.csv resample the 32 inlet cases after averaging noise repetitions and training-seed errors; neither noise nor seeds are counted as additional independent cases. These tests do not establish superiority over all possible calibrated traditional closures.

More observation sites do not guarantee lower inlet error when the candidate forward model differs from the reference physics. Additional sites constrain an approximate model differently; inlet regularization is selected across all layouts for each method. Convergence diagnostics distinguish this model/data limitation from an optimizer that stopped too early, but this experiment does not isolate the contribution of each cause.

Optimizer diagnostics include every selected fit, projected-gradient residual, stagnation and iteration-limit status. A selected fit that does not meet the gradient target is not silently labelled converged. See optimizer-summary.csv and the complete fit arrays. The same summary also reports whether additional starts improved the objective relative to the Th-based initializer and how far their recovered amplitudes differed. Multiple starts are a stability check, not a guarantee of global optimality. Inverse elapsed times are execution diagnostics; some overlapped reference-data generation and are not used to claim a speed advantage.

## Seven-vessel extension

The shared topology configuration includes Sen's seven-vessel tree with three junctions and five external endpoints. The generator handles multiple junctions using a topology-derived incidence matrix. The vessel ordering and source-file mapping are explicit, including the nonsequential upstream file order.

The same inlet IDs, partitions, reaction laws, noise schedule, inverse bounds and scoring rules apply. Independent seven-vessel blood and radial biochemical references have been generated, three new neural seeds have been trained, and fixed-C0 forward and sparse-inverse comparisons have been evaluated. A short-run serial/parallel blood-solver comparison was bitwise equal. There are no historical seven-vessel biochemical neural weights to replay.

## Reference and preservation checks

The seven-vessel CFD resolution is recorded in blood-audit.json. Its conservation and linear/nonlinear solver checks are retained. The biochemical mesh checks below refine the concentration solver; they do not by themselves establish convergence of the seven-vessel blood mesh.
"""
 for name in ['single','three','seven']:
  maxe=max(checks['refinement'][name]['max_by_species'])
  text+=f'\n- {name}: maximum concentration change on the three predeclared training-only mesh checks is {100*maxe:.3f}%.\n'
 text+=f'\n- All {len(protected)} protected pre-existing files match their original SHA-256 values.\n'
 text+='\n- Numerical replay and finite-difference checks are retained in each evaluation directory. GPU utilization and training timings are in training-runs.csv.\n'
 (OUT/'REPORT.md').write_text(text)
 zh="""# 统一数据协议后的神经网络基线

本轮保持已有架构不变，用统一生成的独立 CFD 生化数据重新训练单血管、三血管和七血管基线。旧模型和旧结果已保留。

三种血管使用同一批 112 条入口曲线：64 条训练、16 条验证、32 条测试，采样范围一致。每条完整网络轨迹只属于一个分组。C0 和神经网络共用血流、观测、噪声和评分规则。每种血管单独训练权重，七血管沿用三血管的结构和训练目标。

这里的 CFD 仍是数值参考。此次检验的是固定血流条件下的新生化入口，尚不代表对任意血流的泛化。

## 已知入口时，三种物质预测如何

以下误差越低越好。重训神经网络一栏是三个种子的平均误差；旧权重一栏是在同一新参考下重放历史模型得到的结果。
"""
 for name,label in [('single','单血管'),('three','三血管'),('seven','七血管')]:zh+=f'\n### {label}\n\n'+table(summary,name,'forward')
 zh+='\n## 只有一、二、三个测点时\n\n只观测 Th 和 Fg，同时评价入口恢复和三种物质。下面是带噪声结果，不能仅根据 Fn 更准就宣称入口也恢复得更好。\n'
 for name,label in [('single','单血管'),('three','三血管'),('seven','七血管')]:zh+=f'\n### {label}\n\n'+table(summary,name,'inverse')
 zh+="""
## 七血管统一到了哪一步

七血管已从原始数值血流档案的边界和几何出发，独立求解七段轴对称血流及三个分叉的耦合，再生成 Th、Fg、Fn 参考数据。它沿用三血管的网络结构和训练目标，扩展连接关系，并完成三个随机种子的训练，以及和固定 C0 的预测与稀疏反演对照。这里没有使用 Sen 神经网络预测的血流来生成标签。

## 这份基线用于什么

这套结果固定后，可以与后续自己设计的新模型比较。新模型应使用相同训练与验证划分；后续研究不要根据已公布测试成绩反复调参后，再把同一测试集称为盲测。

单血管和三血管的旧架构本来不同，本轮分别保留了它们。七血管沿用三血管的网络和训练目标。数据、划分和评分统一，不意味着把两套既有网络改成了同一个网络。

## 怎样理解结果的范围

这里比较的是固定的常规一维 C0。神经网络学习了截面内浓度分布对平均输运和反应的影响；这项对照不能证明图结构本身具有独有优势，也不能证明它优于所有可能校准过的数值闭合。

三血管带噪声的入口恢复平均误差范围为 THREE_INLET_RANGE。相比 C0 有改善，但入口恢复还不够准确。单血管的第二个测点靠近入口，分叉网络的新增测点位于更远的下游，因此不能把不同网络的反演差异全归因于血管数量。

七血管采用七段轴对称血流加分叉耦合，没有模拟完整三维分叉几何。生化网格加密检查已完成；七血管血流网格本身尚未做完整的收敛研究。

表中 C0 的入口误差不一定随测点增加而下降。多测点会同时约束更多位置，而近似模型可能无法同时匹配这些位置；正则化强度也按各方法的全部测点布局统一选择。这轮可以检查优化是否收敛，但没有单独量化上述因素各自贡献多少。

本轮没有进行受控的速度测试，不能根据这次运行耗时宣称神经网络更快。
"""
 inv_three=[100*r['inlet'] for r in summary if r['network']=='three' and r['setting']=='inverse' and r['noise']=='noisy' and r['family']=='new']
 zh=zh.replace('THREE_INLET_RANGE',f'{min(inv_three):.1f}%–{max(inv_three):.1f}%')
 for old,new in [('Fixed C0','固定 C0'),('Retrained baseline (3 seeds)','重训神经网络（3 种子平均）'),('Old weights','旧权重'),('| Sites |','| 测点数 |'),('| Method |','| 方法 |'),('| Inlet |','| 入口 |')]:zh=zh.replace(old,new)
 (OUT/'SUMMARY_ZH.md').write_text(zh)
 style='<style>body{font:16px system-ui;color:#173040;max-width:1160px;margin:40px auto;padding:0 24px;line-height:1.65}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:15px}th,td{padding:10px;text-align:left;border-bottom:1px solid #dce5eb}th{background:#edf4f7}img{max-width:100%}h1{line-height:1.25}h2{margin-top:36px}a{color:#146b82}</style>'
 for stem,body in [('report',text),('summary-zh',zh)]:
  page='<!doctype html><meta charset="utf-8"><title>Unified CFD baselines</title>'+style+'<nav><a href="summary-zh.html">中文说明</a> · <a href="report.html">English report</a> · <a href="curves.html">Concentration curves</a></nav>'+render_markdown(body)
  page+='<h2>Figures</h2><img src="forward.png" alt="Known inlet errors"><img src="inverse.png" alt="Sparse inverse errors">'
  (OUT/f'{stem}.html').write_text(page)
 source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/gp52/unified').glob('*.py')}
 artifacts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.rglob('*.pt') if 'pilot-' not in str(p)}
 (OUT/'manifest.json').write_text(json.dumps(dict(source_sha256=source,checkpoint_sha256=artifacts),indent=2))
 print(json.dumps(dict(summary_rows=len(summary),protected=len(protected),optimizer=optimizer)),flush=True)
if __name__=='__main__':main()
