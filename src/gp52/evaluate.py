import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from .runtime import ASSETS,tensor
from .data import load_split,metric
from .inference import get_solver,Solver
from .provenance import file_sha256,state_sha256
from .inverse import Context,Adapter,inverse,relative,LAYOUTS
from .transport import THETA

def main():
    p=argparse.ArgumentParser(description="Evaluate frozen models on the eight published test inlets.")
    p.add_argument("--methods",nargs="+",choices=["C1","GG","GL","LG","LL","transport","polynomial","retrained"],default=["C1","GG"])
    p.add_argument("--inverse",action="store_true")
    p.add_argument("--verify-reference",action="store_true")
    p.add_argument("--output",type=Path,default=Path("outputs/evaluation"))
    p.add_argument("--transport-weights",type=Path,help="Fresh transport checkpoint for the retrained method")
    p.add_argument("--reaction-weights",type=Path,help="Fresh reaction checkpoint for the retrained method")
    a=p.parse_args()
    custom=a.transport_weights is not None or a.reaction_weights is not None
    if ("retrained" in a.methods) != custom or (custom and (a.transport_weights is None or a.reaction_weights is None)):
        p.error("Use --methods ... retrained with both --transport-weights and --reaction-weights.")
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    d=load_split("test");ctx=Context();truth=tensor(d["y"]);area=tensor(d["area"]);t=tensor(ctx.t)
    models={name:(Solver("GG",weights=a.reaction_weights,transport_weights=a.transport_weights) if name=="retrained" else get_solver(name)) for name in a.methods}
    r={"status":"running","species":["Th","Fg","Fn"],"seeds":d["seeds"].tolist(),"windows":{"primary":.82,"full":1.23},"forward":{},"inverse":{},"direct_reaction":{}}
    r["model_sources"]={name:{"source":"released frozen model"}for name in a.methods}
    if custom:
        r["model_sources"]["retrained"]={"source":"explicit newly trained checkpoints",
            "transport_sha256":file_sha256(a.transport_weights),"reaction_sha256":file_sha256(a.reaction_weights),
            "loaded_transport_state_sha256":state_sha256(models["retrained"].model.transport.net.state_dict()),
            "loaded_reaction_fp32_state_sha256":state_sha256({k:v.float()for k,v in models["retrained"].model.reaction.state_dict().items()})}
    started=time.perf_counter()
    for name,model in models.items():
        y,diag=model.solve(d["amps"])
        r["forward"][name]={w:{"mean":metric(y,truth,area,t,cut).mean(0).cpu().tolist(),"cases":metric(y,truth,area,t,cut).cpu().tolist()}for w,cut in r["windows"].items()}
        r["forward"][name].update(mass_defect=float(diag[:,0].max()),minimum=float(y.min()))
        np.savez_compressed(a.output/f"forward-{name}.npz",y=y.cpu().numpy(),amplitudes=d["amps"])
        if hasattr(model,"model"):
            model.model.reaction.float()
            with torch.no_grad():
                rate=2*tensor(d["y"][:,0,:,1:],torch.float32).transpose(1,2)*tensor(d["y"][:,1,:,1:],torch.float32).transpose(1,2)*model.model.reaction(tensor(d["rate_features"],torch.float32))
                target=tensor(d["reaction_physical"][:,:,1:],torch.float32).transpose(1,2)
                source=rate.transpose(1,2).double()[:,None];ref=target.transpose(1,2).double()[:,None]
                r["direct_reaction"][name]={w:metric(source,ref,area[:,:,1:],t[1:],cut).mean(0).cpu().tolist()for w,cut in r["windows"].items()}
            model.model.reaction.double()
        print("FORWARD",name,r["forward"][name]["primary"]["mean"],flush=True)
    if a.inverse:
        settings=json.loads((ASSETS/"inverse-settings.json").read_text());obs=[];rowspec=[]
        for i,seed in enumerate(d["seeds"]):
            for rep in [-1,0,1,2]:
                value=d["exact_observations"][i].copy()
                if rep>=0:value+=.02*np.random.default_rng(np.random.SeedSequence([2026102620,int(seed),rep])).normal(size=value.shape)
                obs.append(value.ravel());rowspec.append({"seed":int(seed),"replicate":rep,"index":i})
        obs=np.stack(obs);ix=torch.tensor([v["index"]for v in rowspec],device="cuda")
        for layout in LAYOUTS:
            scales=[];lams=[]
            for row in rowspec:
                level="clean" if row["replicate"]<0 else "noisy"
                scales.append(ctx.scaling(layout,0. if level=="clean" else .02,np.array(settings["model_rms"])))
                lams.append(settings["lambda"][layout][level])
            r["inverse"][layout]={}
            for name,model in models.items():
                adapter=Adapter(ctx,model,np.array(THETA)if name=="C1"else np.zeros(5),name)
                inferred,detail=inverse(adapter,obs,np.stack(scales),np.array(lams),ctx,maxiter=100)
                y=tensor(adapter.fields(inferred))
                errors={w:metric(y,truth[ix],area[ix],t,cut).cpu().numpy()for w,cut in r["windows"].items()}
                rows=[]
                for j,row in enumerate(rowspec):
                    rows.append({"seed":row["seed"],"replicate":row["replicate"],"primary_error":errors["primary"][j].tolist(),"full_error":errors["full"][j].tolist(),"inferred_amplitudes":inferred[j].tolist(),"inlet_error":relative(inferred[j]@ctx.basis,d["amps"][row["index"]]@ctx.basis),"converged":detail["selected_converged"][j]})
                summary={}
                for level in ["clean","noisy"]:
                    sub=[v for v in rows if (v["replicate"]<0)==(level=="clean")]
                    summary[level]={"primary_mean":np.mean([v["primary_error"]for v in sub],0).tolist(),"full_mean":np.mean([v["full_error"]for v in sub],0).tolist(),"mean_inlet_error":float(np.mean([v["inlet_error"]for v in sub])),"converged":sum(v["converged"]for v in sub),"cases":len(sub)}
                r["inverse"][layout][name]={"cases":rows,"summary":summary,"optimizer":detail}
                np.savez_compressed(a.output/f"inverse-{layout}-{name}.npz",y=y.cpu().numpy(),amplitudes=inferred)
                print("INVERSE",layout,name,summary["noisy"]["primary_mean"],flush=True)
    if a.verify_reference:
        root=Path(__file__).resolve().parents[2]
        expected=json.loads((root/"benchmarks/reference-results.json").read_text())
        published=json.loads((root/"benchmarks/published-results.json").read_text())
        worst=0.
        for name in a.methods:
            reference=expected["forward"].get(name)
            if reference:
                for w in r["windows"]:worst=max(worst,float(np.max(abs(np.array(r["forward"][name][w]["cases"])-np.array(reference[w]["cases"])))))
            if a.inverse:
                for layout in LAYOUTS:
                    rows=r["inverse"][layout][name]["cases"]
                    if name in expected["inverse"][layout]:
                        before=expected["inverse"][layout][name]["cases"]
                        for w in r["windows"]:worst=max(worst,float(np.max(abs(np.array([x[w+"_error"]for x in rows])-np.array([x[w+"_error"]for x in before])))))
                    elif name=="C1":
                        before=published["inverse"][layout]["C1"]["cases"]
                        worst=max(worst,float(np.max(abs(np.array([x["primary_error"]for x in rows])-np.array([x["error"]for x in before])))))
        if worst>1e-8:raise AssertionError(f"Reference metric mismatch: {worst}")
        r["reference_metric_max_difference"]=worst
        r["reference_checked_methods"]=[name for name in a.methods if name in expected["forward"] or name=="C1"]
    r.update(status="complete",seconds=time.perf_counter()-started)
    (a.output/"metrics.json").write_text(json.dumps(r,indent=2)+"\n")
    print("COMPLETE",r["seconds"],"seconds",flush=True)
if __name__=="__main__":main()
