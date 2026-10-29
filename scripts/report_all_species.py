"""Full Th/Fg/Fn display from frozen single/three-vessel experiments.

No training, inverse refitting or regularization selection is performed.
"""
import argparse,base64,csv,hashlib,json,subprocess,sys,zlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
OUT=ROOT/"reports/all-species-v3"
RUN=ROOT/"runs/all-species-v3"
SPECIES=("Th","Fg","Fn")
KNOTS=np.array([0,.06,.14,.24,.40,.60,.82])
METRICS=("inlet_0_082","Th","Fg","Fn")
COLORS={"C0":"#d77522","GG":"#087eaa","neural_42":"#087eaa","neural_43":"#38884c","neural_44":"#8a62ac"}
LABELS={"C0":"Fixed C0","GG":"Frozen neural","neural_42":"Neural 42","neural_43":"Neural 43","neural_44":"Neural 44"}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+"\n")
def read_npz(path):
 with np.load(path) as z:return {k:z[k].copy() for k in z.files}
def csv_read(path):
 with Path(path).open() as f:return list(csv.DictReader(f))
def csv_write(path,rows):
 with Path(path).open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def linear_weights(x,sites):
 return np.stack([np.interp(sites,x,u)for u in np.eye(len(x))],axis=1)

def configure(kind):
 single=kind=="single"
 cfg=dict(network=kind,base_commit="503ec33" if single else "9ab10bd",
  baseline_id="single-c0-fv-v1" if single else "three-c0-fv-v1",
  methods=["C0","GG"] if single else ["C0","neural_42","neural_43","neural_44"],
  input=ROOT/("runs/single-vessel-unified-v2" if single else "runs/inverse-three/long-window/tuned"),
  old_report=ROOT/("reports/single-vessel-unified-v2" if single else "reports/three-vessel-inverse-unified-v2"),
  prior_names={"C0":"C0","GG":"GG"} if single else {"C0":"ordinary_1d",**{f"neural_{s}":f"neural_{s}"for s in (42,43,44)}})
 if single:
  ref=read_npz(ROOT/"data/test.npz");f=read_npz(ROOT/"src/gp52/assets/flow.npz")
  cfg.update(ids=ref["seeds"],amps=ref["amps"],truth=ref["y"],t=f["t"],
    positions=np.array([.006,.03,.054]),site_labels=["6 mm","30 mm","54 mm"],
    wx=linear_weights(f["x"],[.006,.03,.054]),regions={"all":np.arange(41)},
    layouts={1:[1],2:[0,1],3:[0,1,2]},case_offset=0,
    ref_path=ROOT/"data/test.npz")
  ob=read_npz(cfg["input"]/"observations-test.npz")
  obs=np.zeros((8,4,2,3,24))
  for i in np.flatnonzero(ob["layouts"]==3):obs[ob["cases"][i],ob["reps"][i]]=ob["obs"][i].reshape(2,3,24)
  cfg.update(obs=obs,sample_t=ob["sample_times"])
 else:
  raw=read_npz(ROOT/"runs/reference-final/training-grid.npz");ids=np.arange(32,48)
  counts=[30,10,17];lengths=[.06,.02,.034];wx=np.zeros((3,57));regions={"all":np.arange(57)}
  for v,(nv,L,start) in enumerate(zip(counts,lengths,[0,30,40])):
   wx[v,start:start+nv]=linear_weights((np.arange(nv)+.5)*L/nv,[L/2])[0]
   regions[["parent","daughter1","daughter2"][v]]=np.arange(start,start+nv)
  cfg.update(ids=ids,amps=raw["amplitudes"][ids],truth=raw["y"][ids],t=raw["t"],
    positions=np.array([.03,.01,.017]),site_labels=["Parent: 30 mm","Daughter 1: 10 mm","Daughter 2: 17 mm"],
    wx=wx,regions=regions,layouts={1:[0],2:[0,1],3:[0,1,2]},case_offset=32,
    ref_path=ROOT/"runs/reference-final/training-grid.npz")
  ob=read_npz(cfg["input"]/"observations.npz")
  cfg.update(obs=ob["observations"][ids],sample_t=ob["sample_t"])
 assert cfg["truth"].shape==(len(cfg["ids"]),3,cfg["wx"].shape[1],616)
 assert np.isclose(cfg["t"][-1],1.23) and np.isclose(cfg["sample_t"][-1],1.23)
 return cfg

def protect(cfg):
 paths=set(ROOT/p for p in subprocess.check_output(["git","ls-tree","-r","--name-only",cfg["base_commit"]],cwd=ROOT,text=True).splitlines())
 paths.update(cfg["input"].glob("*.json"));paths.update(cfg["input"].glob("*.npz"));paths.add(cfg["ref_path"])
 if cfg["network"]=="single":
  lock=json.loads((ROOT/"benchmarks/single-vessel-method-lock-v2.json").read_text())
  for key in ("protected_sha256","added_source_sha256"):
   for p,h in lock[key].items():assert digest(ROOT/p)==h,p;paths.add(ROOT/p)
 else:
  lock=json.loads((ROOT/"benchmarks/three-vessel-method-lock-v1.json").read_text())["current_three_vessel"]
  for p,h in lock["sha256"].items():assert digest(ROOT/p)==h,p;paths.add(ROOT/p)
  for seed in (42,43,44):paths.add(ROOT/f"runs/closure-final-{seed}/model.pt")
  for p,h in json.loads((ROOT/"runs/inverse-three/protected-before.json").read_text()).items():
   assert digest(ROOT/p)==h,p;paths.add(ROOT/p)
 snapshot={str(p.relative_to(ROOT)):digest(p)for p in sorted(paths)if p.is_file()}
 save_json(RUN/"protected-before.json",snapshot)
 return snapshot

def forward(cfg,method):
 path=RUN/f"forward-{method}.npz"
 if path.exists():return read_npz(path)
 print("Known-inlet replay",method,flush=True)
 if cfg["network"]=="single":
  from gp52.single_v2.c0 import C0
  from gp52.inference import get_solver
  model=C0() if method=="C0" else get_solver("GG")
  y,diag=model.solve(cfg["amps"]);y=y.cpu().numpy()
  with np.load(ROOT/"src/gp52/assets/flow.npz") as f:vol=f["F"][0,:,0,:41].T*float(f["dx"])
  parity=0.
  if method=="GG":
   with np.load(ROOT/"data/reference-forward.npz") as z:parity=float(abs(y-z["GG"]).max())
   assert parity<1e-10,parity
 else:
  from gp52.three.inverse_gpu import GPUModel
  seed=None if method=="C0" else int(method.split("_")[1])
  model=GPUModel(seed,observation_end=1.23,precision="double")
  y=model.predict(cfg["amps"],True);vol=model.data["volume"].T
  if hasattr(vol,"cpu"):vol=vol.cpu().numpy()
  previous=ROOT/f"runs/evaluation-{seed or 42}/{'ordinary_1d' if seed is None else 'neural_combined'}.npz"
  with np.load(previous) as z:parity=float(abs(y-z["y"][32:48]).max())
  assert parity<5e-7,parity
 np.savez_compressed(path,y=y,volume=vol,t=cfg["t"],case_ids=cfg["ids"],previous_replay_max_abs=parity)
 return read_npz(path)

def metrics(y,truth,volume):
 return np.sqrt(np.sum((y-truth)**2*volume[None,None],axis=(2,3))/np.sum(truth**2*volume[None,None],axis=(2,3)))
def sample_fields(y,cfg):return np.einsum("pn,bsnt->bspt",cfg["wx"],y,optimize=True)

def collect(cfg):
 rows=[];frows=[];fcurves=[];icurves=[];max_parity=0.;forward_checks={}
 prior=csv_read(cfg["old_report"]/"per-case.csv")
 old={(r["method"],int(r["case_id"]),int(r["noise_rep"]),int(r["sensors"])):r for r in prior}
 C=len(cfg["ids"]);M=len(cfg["methods"]);T=len(cfg["t"]);volume=None
 basis=np.stack([np.interp(cfg["t"][cfg["t"]<=.82+1e-12],KNOTS,u)for u in np.eye(7)])
 for method in cfg["methods"]:
  name=cfg["prior_names"][method]
  fit=read_npz(cfg["input"]/f"test-{name}.npz")
  pred=read_npz(cfg["input"]/f"prediction-{name}.npz")
  for k in ("cases","reps","layouts"):assert np.array_equal(fit[k],pred[k])
  assert np.array_equal(pred["t"],cfg["t"])
  assert pred["y"].shape==(C*12,3,cfg["wx"].shape[1],T)
  vol=pred["volume"] if cfg["network"]=="single" else pred["volume"].T
  if volume is None:volume=vol
  else:assert np.array_equal(volume,vol)
  ci=fit["cases"]-cfg["case_offset"]
  assert set(zip(ci.tolist(),fit["reps"].tolist(),fit["layouts"].tolist()))=={(c,r,n)for c in range(C)for r in range(4)for n in (1,2,3)}
  assert np.isfinite(pred["y"]).all()
  inlet=fit["amplitudes"]@basis;true_inlet=cfg["amps"][ci]@basis
  ie=np.sqrt(np.sum((inlet-true_inlet)**2,1)/np.sum(true_inlet**2,1))
  for region,ix in cfg["regions"].items():
   es=metrics(pred["y"][:,:,ix],cfg["truth"][ci][:,:,ix],vol[ix])
   for j,c in enumerate(ci):
    row=dict(method=method,case_id=int(cfg["ids"][c]),noise_rep=int(fit["reps"][j]),
      sensors=int(fit["layouts"][j]),region=region,inlet_0_082=float(ie[j]),
      Th=float(es[j,0]),Fg=float(es[j,1]),Fn=float(es[j,2]))
    rows.append(row)
    if region=="all":
     previous=old[(name,row["case_id"],row["noise_rep"],row["sensors"])]
     max_parity=max(max_parity,abs(row["Fn"]-float(previous["fn_0_123"])),abs(row["inlet_0_082"]-float(previous["inlet_0_082"])))
  curves=np.empty((C,4,3,3,3,T))
  sampled=sample_fields(pred["y"],cfg)
  for j,c in enumerate(ci):curves[c,fit["reps"][j],fit["layouts"][j]-1]=sampled[j]
  icurves.append(curves)
  fwd=forward(cfg,method);assert np.allclose(fwd["volume"],volume,rtol=1e-12,atol=1e-18)
  forward_checks[method]=float(fwd["previous_replay_max_abs"])
  for region,ix in cfg["regions"].items():
   es=metrics(fwd["y"][:,:,ix],cfg["truth"][:,:,ix],volume[ix])
   for i,c in enumerate(cfg["ids"]):frows.append(dict(method=method,case_id=int(c),region=region,Th=float(es[i,0]),Fg=float(es[i,1]),Fn=float(es[i,2])))
  fcurves.append(sample_fields(fwd["y"],cfg))
  print("Scored",method,C*12,"inverse fits, all three species",flush=True)
 assert max_parity<1e-12,max_parity
 curves=dict(forward=np.stack(fcurves),inverse=np.stack(icurves),truth=sample_fields(cfg["truth"],cfg),
     observations=cfg["obs"],t=cfg["t"],sample_t=cfg["sample_t"],case_ids=cfg["ids"],
     methods=np.array(cfg["methods"]),site_positions=cfg["positions"])
 np.savez_compressed(RUN/"all-concentration-curves.npz",**curves)
 csv_write(OUT/"inverse-per-case.csv",rows);csv_write(OUT/"forward-per-case.csv",frows)
 save_json(OUT/"replay-checks.json",dict(old_inlet_and_fn_max_abs_difference=max_parity,
     known_inlet_replay_max_abs=forward_checks,selected_inverse_fits=C*12*M,
     neural_training=False,inverse_refitting=False,regularization_retuning=False))
 return rows,frows,curves

def aggregate(cfg,rows,frows):
 summary={};paired=[];summary_rows=[];forward_summary=[]
 ids=cfg["ids"];draws=np.random.default_rng(2026102809).integers(0,len(ids),(10000,len(ids)))
 def is_neural(r):return r["method"]!="C0"
 for region in cfg["regions"]:
  summary[region]={}
  for noise in ("clean","noisy"):
   summary[region][noise]={}
   for n in (1,2,3):
    sub=[r for r in rows if r["region"]==region and r["sensors"]==n and (r["noise_rep"]==0 if noise=="clean" else r["noise_rep"]>0)]
    means={}
    for method in cfg["methods"]+["neural_mean"]:
     chosen=[r for r in sub if is_neural(r)]if method=="neural_mean" else [r for r in sub if r["method"]==method]
     means[method]={k:float(np.mean([r[k]for r in chosen]))for k in METRICS}
     summary_rows.append(dict(region=region,noise=noise,sensors=n,method=method,**means[method]))
    summary[region][noise][str(n)]=means
    for key in METRICS:
     diffs=np.array([np.mean([r[key]for r in sub if r["case_id"]==c and is_neural(r)])-np.mean([r[key]for r in sub if r["case_id"]==c and not is_neural(r)])for c in ids])
     low,high=np.quantile(diffs[draws].mean(1),[.025,.975])
     paired.append(dict(region=region,noise=noise,sensors=n,metric=key,neural_minus_c0=float(diffs.mean()),
       ci95_low=float(low),ci95_high=float(high),neural_better_cases=int((diffs<0).sum()),independent_cases=len(ids)))
  for method in cfg["methods"]+["neural_mean"]:
   chosen=[r for r in frows if r["region"]==region and (is_neural(r)if method=="neural_mean" else r["method"]==method)]
   forward_summary.append(dict(region=region,method=method,**{k:float(np.mean([r[k]for r in chosen]))for k in SPECIES}))
 save_json(OUT/"summary.json",summary)
 csv_write(OUT/"inverse-summary.csv",summary_rows);csv_write(OUT/"forward-summary.csv",forward_summary);csv_write(OUT/"paired.csv",paired)
 return summary,paired,forward_summary

def plot_curves(cfg,curves,n=None,rep=1,case=0):
 fig,axes=plt.subplots(3,3,figsize=(13.4,8.6),layout="constrained")
 for s,sp in enumerate(SPECIES):
  for p,label in enumerate(cfg["site_labels"]):
   ax=axes[s,p]
   ax.plot(cfg["t"],curves["truth"][case,s,p],color="#20262b",lw=2,label="Resolved reference",zorder=5)
   for m,method in enumerate(cfg["methods"]):
    y=curves["forward"][m,case,s,p] if n is None else curves["inverse"][m,case,rep,n-1,s,p]
    ax.plot(cfg["t"],y,color=COLORS[method],lw=1.5,ls="--" if method=="C0" else "-",label=LABELS[method])
   if n is not None and s<2 and p in cfg["layouts"][n]:
    ax.scatter(cfg["sample_t"],cfg["obs"][case,rep,s,p],s=12,c="#77818c",alpha=.65,label="Measurements",zorder=6)
   ax.set_title(sp+" · "+label,fontsize=10);ax.set_xlim(0,1.23);ax.grid(alpha=.16)
   ax.spines[["top","right"]].set_visible(False)
   if p==0:ax.set_ylabel("Concentration (model units)")
   if s==2:ax.set_xlabel("Time (s)")
 handles,labels=axes[0,cfg["layouts"][n][0] if n else 0].get_legend_handles_labels()
 fig.legend(handles,labels,loc="outside lower center",ncol=len(labels),fontsize=9)
 setting="Known inlet" if n is None else f"{n} measurement site(s), "+("clean" if rep==0 else f"noise realization {rep}")
 fig.suptitle(f"{cfg['network'].title()} vessel | case {cfg['ids'][case]} | {setting}",fontsize=13)
 name="forward-curves.png" if n is None else f"inverse-curves-{n}-{'clean' if rep==0 else 'noisy'}.png"
 fig.savefig(OUT/name,dpi=150);plt.close(fig)

def plot_errors(cfg,summary):
 for noise in ("noisy","clean"):
  fig,axes=plt.subplots(1,3,figsize=(12.2,4.2),layout="constrained")
  for s,sp in enumerate(SPECIES):
   ax=axes[s]
   for name,offset,color in (("C0",-.18,"#d77522"),("neural_mean",.18,"#087eaa")):
    values=[summary["all"][noise][str(n)][name][sp]*100 for n in (1,2,3)]
    bars=ax.bar(np.arange(1,4)+offset,values,.32,color=color,label="Fixed C0" if name=="C0" else "Frozen neural")
    ax.bar_label(bars,fmt="%.2f",fontsize=9,padding=3)
   ax.set_title(sp+" reconstruction");ax.set_xticks([1,2,3]);ax.set_xlabel("Measurement sites")
   ax.set_ylabel("Whole-domain relative L2 error (%)");ax.set_ylim(0,ax.get_ylim()[1]*1.2)
   ax.spines[["top","right"]].set_visible(False)
  axes[0].legend(fontsize=9)
  fig.suptitle(f"{cfg['network'].title()} vessel | {noise} | all species scored over 0–1.23 s")
  fig.savefig(OUT/f"errors-{noise}.png",dpi=160);plt.close(fig)

def viewer(cfg,curves):
 names=("forward","inverse","truth","observations")
 arrays=[np.asarray(curves[name],dtype="<f4").ravel()for name in names]
 offset=0;fields={}
 for name,a in zip(names,arrays):
  fields[name]=dict(offset=offset,shape=list(curves[name].shape));offset+=len(a)
 payload=base64.b64encode(zlib.compress(np.concatenate(arrays).tobytes(),9)).decode()
 meta=dict(fields=fields,case_ids=cfg["ids"].tolist(),t=cfg["t"].tolist(),sample_t=cfg["sample_t"].tolist(),
   methods=cfg["methods"],labels=LABELS,colors=COLORS,sites=cfg["site_labels"],layouts=cfg["layouts"],network=cfg["network"])
 template=r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>All-species concentration viewer</title>
<style>
body{font:15px system-ui,sans-serif;color:#1f2933;background:#f3f6f8;margin:0;padding:24px}
main{max-width:1450px;margin:auto}h1{font-size:25px;margin:0 0 8px}p{line-height:1.5}
.controls{display:flex;gap:18px;flex-wrap:wrap;background:white;padding:16px;border-radius:8px}
label{font-weight:600}select{display:block;margin-top:6px;padding:8px;font:inherit;min-width:160px}
#legend{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}i{display:inline-block;width:23px;height:3px;vertical-align:middle;margin-right:6px}
#grid{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:12px}.panel{background:white;border-radius:8px;padding:12px}
h2{font-size:15px;margin:0 0 8px}canvas{width:100%;height:225px;display:block}.note{font-size:13px;color:#536575}
@media(max-width:850px){#grid{grid-template-columns:1fr}}
</style><main>
<h1 id="title">Th, Fg and Fn — frozen models versus fixed C0</h1>
<p>Compare every held-out inlet case. The resolved reference is black. Concentrations use the model's units.
The three-vessel neural curves are separate frozen seeds; no prediction ensemble is shown.</p>
<div class="controls">
<label>Test inlet case<select id="case"></select></label>
<label>Experiment<select id="mode"><option value="forward">Known inlet: forward</option><option value="1">Inverse: 1 measurement site</option><option value="2">Inverse: 2 measurement sites</option><option value="3">Inverse: 3 measurement sites</option></select></label>
<label>Observation noise<select id="noise"><option value="0">Clean</option><option value="1" selected>Noisy realization 1</option><option value="2">Noisy realization 2</option><option value="3">Noisy realization 3</option></select></label>
</div><div id="legend"></div><p id="status">Loading embedded curve data…</p><div id="grid"></div>
<p class="note">No external services, libraries or downloads are used. Every curve covers 0–1.23 s. Vertical axes are scaled to the displayed curves to make Fg changes visible. Dots mark the Th/Fg observations actually used by the selected inverse layout; Fn is never observed.
These point curves complement the report's volume-weighted whole-domain errors. The main tables average individual case/model errors, not averaged predictions.
Viewer samples use float32 only for display; saved curve arrays and error calculations retain float64.</p>
<p><a href="report.html">Full report</a> · <a href="inverse-summary.csv">All inverse summary errors</a> · <a href="inverse-per-case.csv">Per-case and per-vessel errors</a></p></main>
<script>
const meta=__META__;
const packed="__PAYLOAD__";
let data;
const species=["Th","Fg","Fn"];
const byId=id=>document.getElementById(id);
function series(name,indices){
 const f=meta.fields[name];let offset=f.offset,stride=1;
 for(let k=f.shape.length-1;k>=0;k--){if(k<indices.length)offset+=indices[k]*stride;stride*=f.shape[k]}
 return data.subarray(offset,offset+f.shape[f.shape.length-1]);
}
function draw(canvas,items,dots){
 const w=Math.max(260,canvas.clientWidth),h=225,dpr=window.devicePixelRatio||1;
 canvas.width=w*dpr;canvas.height=h*dpr;
 const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
 const left=47,right=w-13,top=12,bottom=h-34;
 let low=Infinity,high=-Infinity;
 for(const a of items)for(const v of a.values){low=Math.min(low,v);high=Math.max(high,v)}
 if(dots)for(const v of dots){low=Math.min(low,v);high=Math.max(high,v)}
 const span=Math.max(high-low,.03);low-=span*.06;high+=span*.09;
 const x=t=>left+t/1.23*(right-left),y=v=>bottom-(v-low)/(high-low)*(bottom-top);
 ctx.font="11px system-ui";ctx.lineWidth=1;ctx.strokeStyle="#dce3e8";ctx.fillStyle="#52606d";
 for(let k=0;k<=4;k++){
  const v=low+(high-low)*k/4,Y=y(v);ctx.beginPath();ctx.moveTo(left,Y);ctx.lineTo(right,Y);ctx.stroke();
  ctx.textAlign="right";ctx.fillText(v.toFixed(2),left-6,Y+4)
 }
 for(const t of [0,.3,.6,.9,1.23]){ctx.textAlign="center";ctx.fillText(t.toFixed(2),x(t),bottom+17)}
 ctx.fillText("Time (s)",(left+right)/2,h-4);
 for(const a of items){
  ctx.strokeStyle=a.color;ctx.lineWidth=a.width||1.6;ctx.setLineDash(a.dash||[]);ctx.beginPath();
  a.values.forEach((v,i)=>i?ctx.lineTo(x(meta.t[i]),y(v)):ctx.moveTo(x(meta.t[i]),y(v)));ctx.stroke()
 }
 ctx.setLineDash([]);
 if(dots){ctx.fillStyle="#747d88";dots.forEach((v,i)=>{ctx.beginPath();ctx.arc(x(meta.sample_t[i]),y(v),2.5,0,2*Math.PI);ctx.fill()})}
}
function redraw(){
 if(!data)return;
 const c=Number(byId("case").value),mode=byId("mode").value,rep=Number(byId("noise").value),n=Number(mode);
 byId("noise").disabled=mode==="forward";
 byId("status").textContent=meta.network+" vessel · case "+meta.case_ids[c]+" · "+(mode==="forward"?"known inlet":n+" measurement site(s), "+(rep?"noise σ = 0.02, realization "+rep:"clean observations"));
 for(let s=0;s<3;s++)for(let p=0;p<3;p++){
  const items=meta.methods.map((m,i)=>({values:mode==="forward"?series("forward",[i,c,s,p]):series("inverse",[i,c,rep,n-1,s,p]),color:meta.colors[m],dash:m==="C0"?[5,3]:[]}));
  items.push({values:series("truth",[c,s,p]),color:"#20262b",width:2});
  const observed=mode!=="forward"&&s<2&&meta.layouts[n].includes(p);
  draw(byId("plot-"+s+"-"+p),items,observed?series("observations",[c,rep,s,p]):null);
 }
}
async function main(){
 byId("title").textContent=meta.network[0].toUpperCase()+meta.network.slice(1)+" vessel: Th, Fg and Fn";
 meta.case_ids.forEach((id,i)=>{const o=document.createElement("option");o.value=i;o.textContent=id;byId("case").appendChild(o)});
 byId("legend").innerHTML='<span><i style="background:#20262b"></i>Resolved reference</span>'+meta.methods.map(m=>'<span><i style="background:'+meta.colors[m]+'"></i>'+meta.labels[m]+'</span>').join("")+'<span>● Measurements</span>';
 for(let s=0;s<3;s++)for(let p=0;p<3;p++){
  const panel=document.createElement("section");panel.className="panel";
  const title=document.createElement("h2");title.textContent=species[s]+" · "+meta.sites[p];panel.appendChild(title);
  const canvas=document.createElement("canvas");canvas.id="plot-"+s+"-"+p;canvas.setAttribute("aria-label",title.textContent+" concentration versus time");panel.appendChild(canvas);byId("grid").appendChild(panel)
 }
 const compressed=Uint8Array.from(atob(packed),c=>c.charCodeAt(0));
 const decoded=await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer();
 data=new Float32Array(decoded);
 for(const id of ["case","mode","noise"])byId(id).addEventListener("change",redraw);
 window.addEventListener("resize",redraw);redraw();
 window.reportViewerReady=true;
}
main().catch(e=>{byId("status").textContent="Could not open embedded curve data: "+e.message;console.error(e)});
</script></html>"""
 (OUT/"curves.html").write_text(template.replace("__META__",json.dumps(meta,separators=(",",":"))).replace("__PAYLOAD__",payload))

def report(cfg,summary,paired,fwd,snapshot):
 ncases=len(cfg["ids"]);three=cfg["network"]=="three"
 lines=[f"# {cfg['network'].title()}-vessel Th, Fg and Fn: fixed C0 comparison","",
 "This report completes the concentration display for the existing frozen models. It adds full-window Th and Fg scores to the previously reported Fn and inlet scores. Existing inverse fits, regularization choices, weights, C0 equations and historical reports are unchanged.",
 "",
 "**Open [the complete concentration viewer](curves.html)** to choose any test inlet, known-inlet prediction or one/two/three-site inversion, and clean or any of the three noisy realizations. All three species and three spatial locations are shown together.",
 "",
 "## Reading the results","",
 "- Known-inlet forward prediction: supply the true Th inlet and compare all predicted concentration fields with the resolved reference.",
 "- Sparse inverse reconstruction: fit the seven Th inlet amplitudes to Th/Fg observations, then assess all reconstructed fields. Fn is never supplied as an observation.",
 "- Th, Fg and Fn errors use the full 0–1.23 s interval and every cell, with shared cell-volume weights and equal time-sample weights. The inset curves show specific locations; the error tables score the whole domain.",
 "- Inlet recovery retains 0–0.82 s. Observations are the same 24 times from 0.002 to 1.23 s. The final inlet amplitude is held after 0.82 s.",
 f"- There are {ncases} existing held-out inlet cases. Noisy results average three Gaussian realizations per case, with absolute standard deviation 0.02. The two candidate families receive the same arrays.",
 "- Concentrations use the existing model units, not calibrated clinical concentration measurements.",
 ""]
 if three:
  lines+=["Three-vessel neural means average errors from frozen seeds 42, 43 and 44. They are not errors of an ensemble prediction. All seeds remain visible in the curve viewer and the per-method CSV tables. Sites are the parent midpoint (30 mm), then daughter 1 midpoint (10 mm), then daughter 2 midpoint (17 mm).",
  "",
  "The reference resolves axial and radial concentration variation in each tube with a prescribed blood field and a perfectly mixed junction. It is not a full three-dimensional Navier–Stokes bifurcation simulation.",""]
 else:
  lines+=["The neural candidate is the single published GG model. Sensor layouts are 30 mm; 6/30 mm; and 6/30/54 mm. The fixed C0 has no memory closure or learned reaction correction. The original calibrated C1 baseline remains separate and unchanged.",""]
 lines+=["## Known-inlet forward prediction","","| Method | Th | Fg | Fn |","|---|---:|---:|---:|"]
 for r in fwd:
  if r["region"]=="all":
   label="Neural mean (individual errors)" if r["method"]=="neural_mean" else LABELS[r["method"]]
   if not three and r["method"]=="neural_mean":continue
   lines.append("| "+label+" | "+" | ".join(f"{100*r[k]:.2f}%"for k in SPECIES)+" |")
 lines+=["","![Known-inlet concentration curves](forward-curves.png)",""]
 for noise in ("noisy","clean"):
  lines+=["## "+noise.title()+" inverse reconstruction","",
   "| Sites | C0 Th | Neural Th | C0 Fg | Neural Fg | C0 Fn | Neural Fn |",
   "|---:|---:|---:|---:|---:|---:|---:|"]
  for n in (1,2,3):
   means=summary["all"][noise][str(n)]
   vals=[means[m][k]for k in SPECIES for m in ("C0","neural_mean")]
   lines.append(f"| {n} | "+" | ".join(f"{100*v:.2f}%"for v in vals)+" |")
  lines+=["",f"![{noise.title()} field errors](errors-{noise}.png)","",
   "| Sites | C0 inlet, 0–0.82 s | Neural inlet, 0–0.82 s |","|---:|---:|---:|"]
  for n in (1,2,3):
   means=summary["all"][noise][str(n)]
   lines.append(f"| {n} | {100*means['C0']['inlet_0_082']:.2f}% | {100*means['neural_mean']['inlet_0_082']:.2f}% |")
  lines.append("")
  for n in (1,2,3):lines+=[f"### {n} measurement site(s)","",f"![{noise} inverse concentration curves, {n} sites](inverse-curves-{n}-{noise}.png)",""]
 lines+=["The figures above use the first test case by index and, for noisy figures, realization 1. They were not selected for the largest neural gain. The viewer includes every test case and all noise realizations. Gray dots are the observations used at the plotted site; panels without dots show unobserved locations/species.",""]
 if three:
  lines+=["## Noisy errors in each vessel","",
   "| Vessel | Sites | C0 Th | Neural Th | C0 Fg | Neural Fg | C0 Fn | Neural Fn |",
   "|---|---:|---:|---:|---:|---:|---:|---:|"]
  for region in ("parent","daughter1","daughter2"):
   for n in (1,2,3):
    means=summary[region]["noisy"][str(n)];vals=[means[m][k]for k in SPECIES for m in ("C0","neural_mean")]
    lines.append(f"| {region} | {n} | "+" | ".join(f"{100*v:.2f}%"for v in vals)+" |")
  lines.append("")
 lines+=["## Paired uncertainty and interpretation","",
 "Paired 95% bootstrap intervals resample independent inlet cases after averaging the noise realizations and, for the three-vessel experiment, individual seed errors. Noise realizations and training seeds are not counted as extra independent test inlets.",
 "",
 "| Sites | Species | Neural minus C0 | 95% interval | Cases favoring neural |","|---:|---|---:|---|---:|"]
 for r in paired:
  if r["region"]=="all" and r["noise"]=="noisy" and r["metric"]in SPECIES:
   lines.append(f"| {r['sensors']} | {r['metric']} | {100*r['neural_minus_c0']:.2f} pp | [{100*r['ci95_low']:.2f}, {100*r['ci95_high']:.2f}] pp | {r['neural_better_cases']}/{r['independent_cases']} |")
 lines+=["",
 "Species errors have different reference norms. In particular, Fg is initialized at 1 and continuously supplied at boundaries, so a small relative Fg concentration error should not be equated directly with a small error in Fg consumption or reaction rate.",
 "",
 "The inverse results are offline reconstruction using the full observation interval. They do not demonstrate prediction before those observations are available. All comparisons are conditional on the existing blood field and geometry, against the fixed basic C0 baseline. They are not new blind tests or comparisons against every calibrated numerical closure.",
 "",
 "## Preservation and reproducibility","",
 f"All {len(snapshot)} protected original files, including existing reports, method locks, weights and saved inverse arrays, passed before/after SHA256 equality. The old inlet and Fn metrics were recomputed and checked to within 1e-12. All newly added species scores use the same prediction arrays and spatial/time weights.",
 "",
 "Only known-inlet forward replay uses the GPU. No neural training, inverse optimization or regularization retuning is performed. Both C0 and neural known-inlet replays use the unchanged frozen implementations.",
 "",
 "From this repository root and its existing isolated environment:",
 "",
 f"    .venv/bin/python scripts/report_all_species.py --network {cfg['network']}",
 "",
 "- [All inverse summary rows](inverse-summary.csv)",
 "- [Per-case inverse errors, including per-vessel rows](inverse-per-case.csv)",
 "- [Known-inlet forward summary](forward-summary.csv)",
 "- [Known-inlet per-case errors](forward-per-case.csv)",
 "- [Paired case bootstrap intervals](paired.csv)",
 "- [Fixed reporting protocol](protocol.json)",
 "- [Replay verification](replay-checks.json)",
 "- [Preservation verification](integrity.json)",
 "",
 "Full-resolution float64 point curves, with case/method/species/site labels, are retained in runs/all-species-v3/all-concentration-curves.npz. The standalone HTML embeds the same sampled curves as float32 for display only. Plotting precision does not change any reported metric.",
 ""]
 (OUT/"REPORT.md").write_text("\n".join(lines))


def readable_html():
 import html,re
 def inline(text):
  text=html.escape(text,quote=False)
  text=re.sub(r"\[([^\]]+)\]\(([^)]+)\)",r'<a href="\2">\1</a>',text)
  text=re.sub(r"\*\*([^*]+)\*\*",r"<strong>\1</strong>",text)
  return text
 source=(OUT/"REPORT.md").read_text().splitlines()
 body=[];i=0
 while i<len(source):
  line=source[i]
  if not line.strip():i+=1;continue
  if line.startswith("|"):
   table=[]
   while i<len(source) and source[i].startswith("|"):table.append(source[i]);i+=1
   body.append('<div class="table"><table>')
   for j,row in enumerate(table):
    if j==1:continue
    tag="th" if j==0 else "td"
    body.append("<tr>"+"".join("<"+tag+">"+inline(c.strip())+"</"+tag+">"for c in row.strip("|").split("|"))+"</tr>")
   body.append("</table></div>");continue
  if line.startswith("- "):
   body.append("<ul>")
   while i<len(source) and source[i].startswith("- "):body.append("<li>"+inline(source[i][2:])+"</li>");i+=1
   body.append("</ul>");continue
  if line.startswith("    "):
   code=[]
   while i<len(source) and source[i].startswith("    "):code.append(source[i][4:]);i+=1
   body.append("<pre>"+html.escape("\n".join(code))+"</pre>");continue
  image=re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)",line)
  if image:body.append('<figure><img loading="lazy" alt="'+html.escape(image[1],quote=True)+'" src="'+html.escape(image[2],quote=True)+'"></figure>')
  elif line.startswith("#"):
   n=len(line)-len(line.lstrip("#"));body.append(f"<h{n}>"+inline(line[n:].strip())+f"</h{n}>")
  else:body.append("<p>"+inline(line)+"</p>")
  i+=1
 (OUT/"report.html").write_text("""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full Th/Fg/Fn report</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#f3f6f8;color:#24323e;margin:0}main{max-width:1250px;margin:24px auto;padding:30px;background:white}h1{font-size:28px}h2{margin-top:38px}a{color:#087eaa}img{width:100%;height:auto}figure{margin:20px 0}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:right;border-bottom:1px solid #dce4ea;padding:9px;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#eff5f8}pre{background:#eff5f8;padding:16px;overflow:auto}.nav{background:#e9f4fa;padding:12px 16px;border-radius:6px}li{margin:6px 0}@media(max-width:700px){main{padding:16px;margin:0}}</style><main><p class="nav"><a href="curves.html">Explore every concentration curve</a> · <a href="REPORT.md">Markdown source</a> · <a href="inverse-summary.csv">Summary data</a></p>"""+ "\n".join(body)+"</main></html>")

def main():
 p=argparse.ArgumentParser();p.add_argument("--network",required=True,choices=["single","three"]);args=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
 cfg=configure(args.network);snapshot=protect(cfg)
 protocol=dict(version="all-species-v3",network=args.network,baseline_id=cfg["baseline_id"],
   methods=cfg["methods"],observation_species=["Th","Fg"],observations_per_species_per_site=24,
   observation_end_s=1.23,inlet_score_interval_s=[0,.82],all_species_score_interval_s=[0,1.23],
   spatial_weight="shared fixed blood-flow cell volume",time_weight="equal samples",
   case_ids=cfg["ids"].tolist(),sensor_site_positions_m=cfg["positions"].tolist(),sensor_site_labels=cfg["site_labels"],
   nested_layout_site_indices=cfg["layouts"],noise_absolute_std=.02,noisy_realizations=3,
   training=False,inverse_refitting=False,regularization_retuning=False,
   first_case_curve_selection="first by existing case index, not performance",noisy_figure_realization=1)
 save_json(OUT/"protocol.json",protocol)
 rows,frows,curves=collect(cfg)
 summary,paired,fwd=aggregate(cfg,rows,frows)
 plot_curves(cfg,curves)
 for n in (1,2,3):
  for rep in (0,1):plot_curves(cfg,curves,n,rep)
 plot_errors(cfg,summary);viewer(cfg,curves)
 changed=[p for p,h in snapshot.items()if digest(ROOT/p)!=h];assert not changed,changed
 save_json(OUT/"integrity.json",dict(protected_original_files=len(snapshot),changed_files=changed,
   original_commit=cfg["base_commit"],protected_sha256=snapshot,report_script_sha256=digest(__file__)))
 report(cfg,summary,paired,fwd,snapshot);readable_html()
 print(json.dumps(dict(network=args.network,noisy=summary["all"]["noisy"],known_inlet=[r for r in fwd if r["region"]=="all"],
   protected_original_files=len(snapshot)),indent=2),flush=True)
if __name__=="__main__":main()
