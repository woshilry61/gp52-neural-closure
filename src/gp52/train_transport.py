"""Full-trajectory supervised closure training with exact FV conservation.

All 16 known training inlets are in each update. Eight previously viewed
development cases select checkpoints; these are not untouched test cases.
"""
import os,json,time,threading,subprocess,argparse
import numpy as np
import torch
from pathlib import Path
from .runtime import ASSETS,tensor
from .data import transport_dataset as dataset,metric
from .transport import Rollout
plan=json.loads((ASSETS/"splits.json").read_text())
TRAIN=plan["train_seeds"];VALID=plan["development_seeds"]
O=Path("runs/transport")

def validation(m,a,y,area):
    with torch.no_grad():
        out=m(a)[0];t=tensor(m.t,torch.float32)
        e=metric(out,y,area,t).cpu().numpy()
        full=metric(out,y,area,t,1.23).cpu().numpy()
    return dict(mean_errors=e.mean(0).tolist(),case_errors=e.tolist(),full_mean_errors=full.mean(0).tolist(),minimum=float(out.min()),score=float((full**2).mean()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--kind",choices=["graph","linear","local"],default="graph")
    ap.add_argument("--steps",type=int,default=1500)
    ap.add_argument("--output",type=Path,default=Path("runs/transport"));args=ap.parse_args()
    global O;O=args.output.resolve();(O/"runs").mkdir(parents=True,exist_ok=True);(O/"results").mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261026)
    torch.backends.cuda.matmul.allow_tf32=False
    m=Rollout(kind=args.kind,channels=2)
    a,y,area=dataset(TRAIN,m);av,yv,areav=dataset(VALID,m)
    tw=torch.ones(m.T,device="cuda")*m.dt;tw[0]/=2;tw[-1]/=2
    weight=(area*tw)[:,None];norm=(y.square()*weight).sum((-2,-1))
    train_state=dict(a=a,y=y,weight=weight,norm=norm)
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-5,capturable=True)
    def step():
        opt.zero_grad(set_to_none=True)
        out=m(train_state["a"])[0]
        loss=(((out-train_state["y"]).square()*train_state["weight"]).sum((-2,-1))/train_state["norm"]).mean()
        loss.backward();opt.step()
        return loss
    outdir=O/"runs"/args.kind;outdir.mkdir(exist_ok=True)
    before=validation(m,av,yv,areav);print("INITIAL",json.dumps(before),flush=True)
    gpu=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                s=subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,power.draw","--format=csv,noheader,nounits"],text=True).strip()
                gpu.append(dict(seconds=time.perf_counter()-start,values=[float(v)for v in s.split(",")]))
            except subprocess.CalledProcessError:pass
            stop.wait(.25)
    start=time.perf_counter();thread=threading.Thread(target=monitor,daemon=True);thread.start()
    torch.cuda.reset_peak_memory_stats()
    warmstream=torch.cuda.Stream();warmstream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmstream):
        for i in range(5):loss=step()
    torch.cuda.current_stream().wait_stream(warmstream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured_loss=step()
    # All loss operands remain live in train_state throughout replay.
    history=[];best=float("inf");beststep=None
    for i in range(1,args.steps+1):
        graph.replay()
        if i%50==0 or i==1:
            torch.cuda.synchronize();stats=validation(m,av,yv,areav)
            row=dict(step=i,optimizer_updates=i+5,loss=float(captured_loss.detach()),elapsed=time.perf_counter()-start,**stats)
            history.append(row)
            if stats["score"]<best:
                best=stats["score"];beststep=i
                torch.save(dict(state=m.state_dict(),kind=args.kind,channels=2,step=i,optimizer_updates=i+5,train=TRAIN,development=VALID),outdir/"best.pt")
            (O/"results"/(args.kind+"-training.json")).write_text(json.dumps(dict(status="running",kind=args.kind,initial=before,history=history,beststep=beststep),indent=2))
            print(json.dumps(row),flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    stop.set();thread.join()
    arr=np.array([v["values"] for v in gpu])
    result=dict(status="completed",kind=args.kind,initial=before,history=history,beststep=beststep,
        total_optimizer_updates=args.steps+5,batch_cases=len(TRAIN),training_seconds=seconds,
        peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20,
        gpu_utilization_mean=float(arr[:,0].mean()),gpu_utilization_p95=float(np.quantile(arr[:,0],.95)),
        gpu_samples=gpu,training_data=TRAIN,development_data=VALID,
        scope="Previously viewed development cases; joint Th/Fg/Fn full 0..1.23s rollout. Mean-product reaction; frozen Sen blood flow.")
    (O/"results"/(args.kind+"-training.json")).write_text(json.dumps(result,indent=2))
    print("FINISHED",seconds,"beststep",beststep,"utilization",result["gpu_utilization_mean"],flush=True)

if __name__=="__main__":main()
