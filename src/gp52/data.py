from pathlib import Path
import os,json
import numpy as np
import torch
from .runtime import ASSETS,tensor

def data_dir():return Path(os.environ.get("GP52_DATA_DIR",Path.cwd()/"data"))
def load_split(split):
    p=data_dir()/(split+".npz")
    if not p.exists():raise FileNotFoundError(f"{p} is missing. Run python scripts/download_data.py --split {split}.")
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def records(seeds):
    plan=json.loads((ASSETS/"splits.json").read_text())
    mapping={s:(split,i)for split,key in [("train","train_seeds"),("development","development_seeds"),("test","fresh_seeds")]for i,s in enumerate(plan[key])}
    loaded={split:load_split(split)for split in {mapping[s][0]for s in seeds}}
    keys=[k for k in next(iter(loaded.values())) if k!="seeds"]
    return {k:np.stack([loaded[mapping[s][0]][k][mapping[s][1]]for s in seeds])for k in keys}
def transport_dataset(seeds,m):
    r=records(seeds)
    return tuple(tensor(r[k],torch.float32)for k in ["amps","y","area"])
def dataset(seeds,m):
    r=records(seeds);a=tensor(r["amps"],torch.float32);y=tensor(r["y"],torch.float32);area=tensor(r["area"],torch.float32)
    rates={k:tensor(r[k],torch.float32)for k in ["reaction_physical","reaction_discrete"]}
    with torch.no_grad():cache=m.predictor(a)
    target=rates["reaction_physical"][:,:,1:].transpose(1,2)
    r0=2*y[:,0,:,1:].transpose(1,2)*y[:,1,:,1:].transpose(1,2)
    tw=torch.ones(y.shape[-1],device="cuda")*m.transport.dt;tw[[0,-1]]/=2
    weight=(area*tw)[:,None];norm=(y.square()*weight).sum((-2,-1))
    return dict(a=a,y=y,area=area,cache=cache,rate_features=tensor(r["rate_features"],torch.float32),target=target,r0=r0,weight=weight,norm=norm,rate_norm=target.square().mean(),rates=rates)

def metric(y,truth,area,t,cut=.82):
    ix=t<=cut+1e-10;tt=t[ix];w=torch.zeros_like(t);w[ix]=torch.cat([(tt[1:2]-tt[:1])/2,(tt[2:]-tt[:-2])/2,(tt[-1:]-tt[-2:-1])/2])
    weight=area*w
    return (((y-truth).square()*weight[:,None]).sum((-2,-1))/(truth.square()*weight[:,None]).sum((-2,-1)).clamp_min(1e-30)).sqrt()

def metrics(m,d):
    with torch.no_grad():
        out,factor=m.correct(d['cache']);t=tensor(m.transport.t,torch.float32)
        e=metric(out,d['y'],d['area'],t,.82);full=metric(out,d['y'],d['area'],t,1.23)
        rate=d['r0']*m.reaction(d['rate_features'])
        return dict(primary_mean=e.mean(0).cpu().tolist(),primary_cases=e.cpu().tolist(),full_mean=full.mean(0).cpu().tolist(),score=float(full[:,1:].square().mean()),physical_rate_error=float(((rate-d['target']).square().mean()/d['rate_norm']).sqrt()),factor_range=[float(factor.min()),float(factor.max())])
