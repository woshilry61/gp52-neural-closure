"""GPU source pretraining and full-rollout reaction training, frozen transport."""
import json,time,copy,threading,subprocess,argparse,hashlib
import numpy as np
import torch
from pathlib import Path
from .reaction import Coupled
from .runtime import ASSETS
O=Path("runs/reaction")
from .data import dataset,metrics,metric,tensor

def capture(step):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):loss=step()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):loss=step()
    return graph,loss

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['graph','polynomial','local'],default='graph');ap.add_argument('--steps',type=int,default=500);ap.add_argument('--weight',type=float,default=.3);ap.add_argument('--source-steps',type=int,default=1000);ap.add_argument('--output',type=Path,default=Path('runs/reaction'));ap.add_argument('--transport-kind',choices=['graph','local'],default='graph');ap.add_argument('--transport-weights',type=Path);a=ap.parse_args()
    global O;O=a.output.resolve();(O/'runs').mkdir(parents=True,exist_ok=True);(O/'results').mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(2026102620)
    plan=json.loads((ASSETS/'splits.json').read_text());m=Coupled(a.kind,transport_kind=a.transport_kind,transport_weights=a.transport_weights)
    d=dataset(plan['train_seeds'],m);v=dataset(plan['development_seeds'],m)
    name=f'{a.kind}-{a.weight:g}';dest=O/'runs'/name;dest.mkdir(exist_ok=True)
    initial=metrics(m,v)
    norm_closure=(d['r0']-d['target']).square().mean().detach()
    baseline_rate=float(((v['r0']-v['target']).square().mean()/v['rate_norm']).sqrt())
    # All captured loss inputs live in d, v, and local variables until completion.
    gpu=[];stop=threading.Event();start=time.perf_counter()
    def monitor():
        while not stop.is_set():
            p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if p.returncode==0:gpu.append(dict(seconds=time.perf_counter()-start,values=[float(x)for x in p.stdout.strip().split(',')]))
            stop.wait(.25)
    th=threading.Thread(target=monitor,daemon=True);th.start();torch.cuda.reset_peak_memory_stats()
    opt=torch.optim.AdamW(m.reaction.parameters(),lr=.001,weight_decay=1e-5,capturable=True)
    def prestep():
        opt.zero_grad(set_to_none=True);r=d['r0']*m.reaction(d['rate_features']);loss=(r-d['target']).square().mean()/norm_closure
        loss.backward();opt.step();return loss
    graph,loss=capture(prestep)
    for _ in range(a.source_steps):graph.replay()
    torch.cuda.synchronize();pre=metrics(m,v);presteps=int(next(iter(opt.state.values()))['step'])
    torch.save(m.reaction.state_dict(),dest/'source-only.pt');del graph,loss,opt
    initial_loss=(((d['cache']['y']-d['y']).square()*d['weight']).sum((-2,-1))/d['norm'])[:,1:].mean().detach()
    opt=torch.optim.AdamW(m.reaction.parameters(),lr=.0002,weight_decay=1e-5,capturable=True)
    def step():
        opt.zero_grad(set_to_none=True);out,factor=m.correct(d['cache'])
        trajectory=(((out-d['y']).square()*d['weight']).sum((-2,-1))/d['norm'])[:,1:].mean()/initial_loss
        source=(d['r0']*m.reaction(d['rate_features'])-d['target']).square().mean()/norm_closure
        loss=trajectory+a.weight*source;loss.backward();opt.step();return loss
    graph,loss=capture(step);history=[];best=float('inf');best_any=float('inf');beststep=None
    # Selection also requires physical source fitting to beat mean-product closure.
    if pre['physical_rate_error']<baseline_rate:
        best=pre['score'];beststep=0;(dest/'best.pt').write_bytes((dest/'source-only.pt').read_bytes())
    for i in range(1,a.steps+1):
        graph.replay()
        if i%25==0 or i==1:
            torch.cuda.synchronize();stat=metrics(m,v);row=dict(step=i,loss=float(loss),seconds=time.perf_counter()-start,**stat);history.append(row)
            if stat['score']<best and stat['physical_rate_error']<baseline_rate:
                best=stat['score'];beststep=i;torch.save(m.reaction.state_dict(),dest/'best.pt')
            if stat['score']<best_any:
                best_any=stat['score'];torch.save(m.reaction.state_dict(),dest/'best-unconstrained.pt')
            (O/f'results/training-{name}.json').write_text(json.dumps(dict(status='running',name=name,initial=initial,source_pretraining=pre,baseline_physical_rate_error=baseline_rate,history=history,beststep=beststep),indent=2))
            print(json.dumps(row),flush=True)
    torch.cuda.synchronize();updates=int(next(iter(opt.state.values()))['step']);seconds=time.perf_counter()-start;stop.set();th.join()
    m.reaction.load_state_dict(torch.load(dest/'best.pt',map_location='cuda',weights_only=True));selected=metrics(m,v)
    report=dict(status='completed',name=name,kind=a.kind,source_weight=a.weight,initial=initial,source_pretraining=pre,baseline_physical_rate_error=baseline_rate,history=history,beststep=beststep,selected=selected,pretrain_optimizer_updates=presteps,rollout_optimizer_updates=updates,seconds=seconds,gpu_utilization_mean=float(np.mean([r['values'][0]for r in gpu])),peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20,gpu_samples=gpu,train=plan['train_seeds'],development=plan['development_seeds'],parameters=sum(p.numel()for p in m.reaction.parameters()),transport_frozen=True,scheme='Causal predictor-corrector; corrector factor uses own mean-reaction predictor, no true intermediate concentrations; source loss separately uses true means')
    (O/f'results/training-{name}.json').write_text(json.dumps(report,indent=2));print('DONE',name,selected,'GPU',report['gpu_utilization_mean'],flush=True)
if __name__=='__main__':main()
