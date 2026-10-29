"""Original three-vessel heads and physical moment losses on unified CFD labels."""
import argparse,json,time,subprocess
import numpy as np
import torch
from .protocol import RUN,TRAIN,VALID,plan
from .reference import flow,COUNTS,LENGTH
from .network_closure import Closure,reaction_features,arrays_from_blood
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
 out=RUN/'seven'/f'train-{a.seed}';out.mkdir(exist_ok=False)
 torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False
 raw=dict(np.load(RUN/'seven/reference/data.npz'));raw={k:(v[...,::2] if k in ['y','reaction'] else v[::2] if k in ['t','volume'] else v[:,::2] if k=='flux' else v) for k,v in raw.items()}
 blood,t,_=flow('seven',tuple(COUNTS['seven']),1,.002);ar=arrays_from_blood(blood,t);dt=.002
 y=raw['y'];off=np.r_[0,np.cumsum(COUNTS['seven'])]
 grad=np.concatenate([np.diff(y[:,0,off[v]:off[v+1]],axis=1)/(LENGTH['seven'][v]/COUNTS['seven'][v]) for v in range(len(COUNTS['seven']))],1).transpose(2,1,0)
 rf=reaction_features(y,ar,dt).astype(np.float32);tensor=lambda x:torch.tensor(x,device='cuda',dtype=torch.float32)
 m=Closure(ar).cuda();gt=tensor(grad[:,:,TRAIN]);gv=tensor(grad[:,:,VALID])
 ft=tensor(raw['flux'][TRAIN].transpose(1,2,0)[1:]);fv=tensor(raw['flux'][VALID].transpose(1,2,0)[1:])
 fs=float(np.mean(raw['flux'][TRAIN]**2));xt=tensor(rf[TRAIN]);xv=tensor(rf[VALID])
 rt=tensor(raw['reaction'][TRAIN].transpose(0,2,1));rv=tensor(raw['reaction'][VALID].transpose(0,2,1))
 bt=tensor((2*y[TRAIN,0]*y[TRAIN,1]).transpose(0,2,1));bv=tensor((2*y[VALID,0]*y[VALID,1]).transpose(0,2,1))
 rs=float(np.mean(raw['reaction'][TRAIN]**2));steps=plan()['training']['three_steps_per_head']
 log=(out/'gpu.csv').open('w');mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=log)
 start=time.perf_counter()
 try:
  for stage in ['transport','reaction']:
   opt=torch.optim.Adam(getattr(m,stage).parameters(),lr=torch.tensor(.001,device='cuda'),fused=True,capturable=True)
   lossfn=(lambda:((m.flux(gt,dt)-ft)**2).mean()/fs) if stage=='transport' else (lambda:((m.factor(xt)*bt-rt)**2).mean()/rs)
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):opt.zero_grad(set_to_none=True);loss=lossfn();loss.backward();opt.step()
   torch.cuda.current_stream().wait_stream(stream);cg=torch.cuda.CUDAGraph()
   with torch.cuda.graph(cg):opt.zero_grad(set_to_none=True);loss=lossfn();loss.backward();opt.step()
   best=float('inf')
   for j in range(steps):
    if j==steps*2//3:opt.param_groups[0]['lr'].fill_(.0002)
    cg.replay()
    if (j+1)%100==0:
     with torch.no_grad():v=(((m.flux(gv,dt)-fv)**2).mean()/fs if stage=='transport' else ((m.factor(xv)*bv-rv)**2).mean()/rs).item()
     if v<best:best=v;torch.save(getattr(m,stage).state_dict(),out/f'{stage}.pt')
     row=dict(stage=stage,step=j+1,training=float(loss),validation=v,best=best,seconds=time.perf_counter()-start)
     with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
     print(json.dumps(row),flush=True)
   del cg,opt
   getattr(m,stage).load_state_dict(torch.load(out/f'{stage}.pt',weights_only=True))
  torch.save(m.state_dict(),out/'model.pt')
  (out/'metrics.json').write_text(json.dumps(dict(seed=a.seed,seconds=time.perf_counter()-start,architecture='Original three-vessel Closure; seven-vessel adjacency only',steps_per_head=steps,
   train=TRAIN.tolist(),validation=VALID.tolist(),parameters=sum(p.numel() for p in m.parameters())),indent=2))
 finally:mon.terminate();mon.wait();log.close()
if __name__=='__main__':main()
