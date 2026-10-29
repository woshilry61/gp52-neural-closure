"""Original single-vessel rollout and reaction head; shuffled 16-case GPU batches."""
import argparse,os,json,time,subprocess,gc
from pathlib import Path
import numpy as np
import torch
from .protocol import RUN,TRAIN,VALID,plan
os.environ['GP52_PREPARED_ASSETS']=str(RUN/'single/assets')
from gp52.transport import Rollout
from gp52.reaction import Coupled,features
from gp52.runtime import tensor
MICRO=16
def data(ids,m):
 z=np.load(RUN/'single/reference/data.npz')
 y=tensor(z['y'][ids,...,::2],torch.float32);a=tensor(z['amplitudes'][ids],torch.float32)
 area=tensor(z['volume'][::2].T/.06*41,torch.float32)
 tw=torch.full((y.shape[-1],),.002,device='cuda');tw[[0,-1]]/=2
 weight=(area*tw)[None,None].expand(len(ids),-1,-1,-1);norm=(y.square()*weight).sum((-2,-1))
 d=dict(a=a,y=y,weight=weight,norm=norm)
 if isinstance(m,Coupled):
  mm=m.transport;caches=[]
  with torch.no_grad():
   for i in range(0,len(ids),MICRO):caches.append(m.predictor(a[i:i+MICRO]))
   cache=dict(y=torch.cat([c['y'] for c in caches]),features=torch.cat([c['features'] for c in caches]),coeff=caches[0]['coeff'])
   rates=tensor(z['reaction'][ids,:,::2],torch.float32)[:,:,1:].transpose(1,2)
   blood=m.blood();rf=features(y,*blood,mm.dx,mm.dt)
  d.update(cache=cache,rate_features=rf,target=rates,r0=2*y[:,0,:,1:].transpose(1,2)*y[:,1,:,1:].transpose(1,2))
 return d
def slice_data(d,sl):
 r={k:v[sl] for k,v in d.items() if torch.is_tensor(v)}
 if 'cache' in d:r['cache']={k:(v if k=='coeff' else v[sl]) for k,v in d['cache'].items()}
 return r
@torch.no_grad()
def validate(m,d,reaction=False):
 es=[];rates=0.;rate_norm=0.
 for i in range(0,len(d['a']),MICRO):
  b=slice_data(d,slice(i,i+MICRO));out=m.correct(b['cache'])[0] if reaction else m(b['a'])[0]
  es.append((((out-b['y']).square()*b['weight']).sum((-2,-1))/b['norm']).cpu())
  if reaction:
   rates+=float((b['r0']*m.reaction(b['rate_features'])-b['target']).square().sum())
   rate_norm+=float(b['target'].square().sum())
 e=torch.cat(es)
 return dict(score=float(e[:,1:].mean() if reaction else e.mean()),errors=e.sqrt().mean(0).tolist(),
  physical_rate_error=(rates/rate_norm)**.5 if reaction else None)
def optimize(m,d,v,params,lossfn,steps,lr,out,reaction=False,eligible=None):
 opt=torch.optim.AdamW(params,lr=lr,weight_decay=1e-5,capturable=True,fused=True)
 n=len(d['a']);b=slice_data(d,slice(0,MICRO))
 # Static batch operands reuse a CUDA graph while cycling through all training cases.
 b={k:({kk:(vv if kk=='coeff' else vv.clone()) for kk,vv in val.items()} if isinstance(val,dict) else val.clone()) for k,val in b.items()}
 def back():
  loss=lossfn(b);loss.backward();return loss
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(2):opt.zero_grad(set_to_none=False);loss=back()
 torch.cuda.current_stream().wait_stream(stream);opt.zero_grad(set_to_none=False)
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):loss=back()
 opt.zero_grad(set_to_none=False)
 best=float('inf');history=[];start=time.perf_counter()
 # Source-only checkpoint is eligible for selection before trajectory correction.
 initial=validate(m,v,reaction)
 if eligible is None or initial['physical_rate_error']<eligible:
  best=initial['score'];torch.save(m.reaction.state_dict() if reaction else m.state_dict(),out/'best.pt')
 for j in range(steps):
  opt.zero_grad(set_to_none=False)
  if j%(n//MICRO)==0:order=torch.randperm(n,device='cuda')
  for i in [j%(n//MICRO)*MICRO]:
   src=slice_data(d,order[i:i+MICRO])
   for k,val in src.items():
    if isinstance(val,dict):
     for kk,vv in val.items():
      if kk!='coeff':b[k][kk].copy_(vv)
    else:b[k].copy_(val)
   cg.replay()
  opt.step()
  if (j+1)%50==0 or j==0:
   stat=validate(m,v,reaction);ok=eligible is None or stat['physical_rate_error']<eligible
   if ok and stat['score']<best:
    best=stat['score'];torch.save(m.reaction.state_dict() if reaction else m.state_dict(),out/'best.pt')
   row=dict(step=j+1,score=stat['score'],errors=stat['errors'],physical_rate_error=stat['physical_rate_error'],seconds=time.perf_counter()-start)
   history.append(row);print(json.dumps(row),flush=True)
   (out/'history.json').write_text(json.dumps(history,indent=2))
 del cg,opt,b;gc.collect();torch.cuda.empty_cache()
 return dict(best=best,steps=steps,seconds=time.perf_counter()-start,history=history)
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',choices=['transport','reaction'],required=True);p.add_argument('--weight',type=float,default=.3);a=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(a.seed)
 root=RUN/'single'/f'train-{a.seed}';out=root/(a.stage if a.stage=='transport' else f'reaction-{a.weight:g}');out.mkdir(parents=True,exist_ok=False)
 log=(out/'gpu.csv').open('w');mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=log)
 cfg=plan()['training'];start=time.perf_counter()
 try:
  if a.stage=='transport':
   m=Rollout();d=data(TRAIN,m);v=data(VALID,m)
   fn=lambda b:(((m(b['a'])[0]-b['y']).square()*b['weight']).sum((-2,-1))/b['norm']).mean()
   r=optimize(m,d,v,m.net.parameters(),fn,cfg['single_transport_steps'],.001,out)
   state=torch.load(out/'best.pt',weights_only=True);torch.save({k[4:]:v for k,v in state.items() if k.startswith('net.')},root/'transport.pt')
  else:
   m=Coupled(transport_weights=root/'transport.pt');d=data(TRAIN,m);v=data(VALID,m)
   norm=(d['r0']-d['target']).square().mean().detach();ratebase=float(((v['r0']-v['target']).square().mean()/v['target'].square().mean()).sqrt())
   opt=torch.optim.AdamW(m.reaction.parameters(),lr=.001,weight_decay=1e-5,fused=True,capturable=True)
   def pre():
    opt.zero_grad(set_to_none=True);loss=(d['r0']*m.reaction(d['rate_features'])-d['target']).square().mean()/norm;loss.backward();opt.step();return loss
   st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(st):
    for _ in range(3):loss=pre()
   torch.cuda.current_stream().wait_stream(st);cg=torch.cuda.CUDAGraph()
   with torch.cuda.graph(cg):loss=pre()
   for j in range(cfg['single_reaction_source_steps']):cg.replay()
   del cg,opt;gc.collect();torch.cuda.empty_cache()
   torch.save(m.reaction.state_dict(),out/'source-only.pt')
   initial=(((d['cache']['y']-d['y']).square()*d['weight']).sum((-2,-1))/d['norm'])[:,1:].mean().detach()
   def lossfn(b):
    y=m.correct(b['cache'])[0]
    trajectory=(((y-b['y']).square()*b['weight']).sum((-2,-1))/b['norm'])[:,1:].mean()/initial
    source=(b['r0']*m.reaction(b['rate_features'])-b['target']).square().mean()/norm
    return trajectory+a.weight*source
   r=optimize(m,d,v,m.reaction.parameters(),lossfn,cfg['single_reaction_steps'],.0002,out,True,ratebase)
   if not (out/'best.pt').exists():raise RuntimeError('No validation-eligible reaction checkpoint')
  r.update(seed=a.seed,stage=a.stage,source_weight=a.weight,seconds_total=time.perf_counter()-start,
   architecture='Original gp52.transport.Rollout and gp52.reaction.Coupled',train=TRAIN.tolist(),validation=VALID.tolist(),
   batch_cases=MICRO,training_cases=len(TRAIN),peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20)
  (out/'metrics.json').write_text(json.dumps(r,indent=2))
 finally:mon.terminate();mon.wait();log.close()
if __name__=='__main__':main()
