"""Faithful GPU port of the pinned Sen three-vessel script (ARMA K=1, iteration=1)."""
from pathlib import Path
import argparse,json,time,subprocess
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision('highest')
class Sen(torch.nn.Module):
 def __init__(self,data,seeds=1,dtype=torch.float64,parity=False):
  super().__init__();self.B=seeds;self.T=len(data['t'])
  def buf(n,x):self.register_buffer(n,torch.tensor(x,device='cuda',dtype=dtype))
  for n in ['features','graph','labels','a0','beta','M','M2','U','p0','length']:buf(n,data[n])
  self.dt=float(data['dt'][0]); self.ws=torch.nn.ParameterList()
  for j in range(12):
   shape=data[f'w{j}'].shape
   if parity:w=np.broadcast_to(data[f'w{j}'],(seeds,*shape)).copy()
   else:
    w=np.stack([np.random.default_rng(10000+s*100+j).uniform(-np.sqrt(6/sum(shape)),np.sqrt(6/sum(shape)),shape) for s in range(seeds)])
   self.ws.append(torch.nn.Parameter(torch.tensor(w,device='cuda',dtype=dtype)))
 def forward(self):
  z=self.features[None].expand(self.B,-1,-1)
  for j in range(6):
   z=self.graph@torch.bmm(z,self.ws[2*j])+torch.bmm(z,self.ws[2*j+1])
   if j<5:z=torch.nn.functional.leaky_relu(z,float(np.float32(.3)))
   z=torch.tanh(z)
  return z
 def fields(self):
  z=self().reshape(self.B,3,10,2*self.T)
  return z[...,self.T:].exp(),z[...,:self.T]
 def terms(self):
  A,V=self.fields();M=self.M[None,:,None,None];U=self.U[None,:,None,None];M2=self.M2[None,:,None,None];p0=self.p0[None,:,None,None]
  P=(9986+self.beta[None,:,:,None]*(torch.sqrt(A*M2)-torch.sqrt(self.a0)[None,:,:,None]))/p0
  def gt(z):
   first=(z[...,1:2]-z[...,-2:-1])/2/self.dt*M/U
   return torch.cat([first,(z[...,2:]-z[...,:-2])/2/self.dt*M/U,first],-1)
  def gx(z):
   return torch.cat([z[:,:,1:2]-z[:,:,:1],(z[:,:,2:]-z[:,:,:-2])/2,z[:,:,-1:]-z[:,:,-2:-1]],2)*M/(self.length[None,:,None,None]/9)
  kr=-2*1.33*.0025*np.pi/1060/(1.33-1)
  ra=gt(A)+A*gx(V)+V*gx(A)
  rv=gt(V)+1.33*V*gx(V)+gx(P)-kr/U/M*V/A
  phys=(ra.square().mean((2,3))+rv.square().mean((2,3))).sum(1)
  bc=(V[:,0,0]-self.labels[0,:self.T]).square().mean(1)+(V[:,1,-1]-self.labels[19,:self.T]).square().mean(1)+(V[:,2,-1]-self.labels[29,:self.T]).square().mean(1)
  bc=bc+(A[:,0,0,0]-self.a0[0,0]/self.M2[0]*1.02).square()
  total=P*p0+.5*1060*(V*U)**2;flow=A*M2*V*U
  junction=((total[:,0,-1]-total[:,1,0])/self.p0[0]).square().mean(1)+((total[:,0,-1]-total[:,2,0])/self.p0[0]).square().mean(1)
  junction+=((flow[:,0,-1]-flow[:,1,0]-flow[:,2,0])/self.M2[0]/self.U[0]).square().mean(1)
  return phys,bc,junction
 def losses(self):return sum(self.terms())
def verify(data):
 m=Sen(data,parity=True);v=m.losses().sum();v.backward()
 out=m().detach().cpu().numpy()[0]
 report={'output_max_abs':float(abs(out-data['output']).max()),'loss_abs':abs(float(v)-float(data['loss'])),'gradient_max_abs':max(float(abs(w.grad.cpu().numpy()[0]-data[f'g{j}']).max()) for j,w in enumerate(m.ws))}
 assert report['output_max_abs']<1e-12 and report['loss_abs']<1e-12 and report['gradient_max_abs']<1e-10,report
 return report
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=100000);ap.add_argument('--seeds',type=int,default=32);ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 data=dict(np.load(ROOT/'runs/sen-input/input.npz'));parity=verify(data)
 (a.output/'port-equivalence.json').write_text(json.dumps(parity,indent=2));print('PARITY',parity,flush=True)
 m=Sen(data,seeds=a.seeds);lr=torch.tensor(.001,device='cuda',dtype=torch.float32)
 opt=torch.optim.Adam(m.parameters(),lr=lr,eps=1e-7,capturable=True,fused=True)
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(3):
   opt.zero_grad(set_to_none=True);v=m.losses().sum();v.backward();opt.step()
 torch.cuda.current_stream().wait_stream(stream)
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):
  opt.zero_grad(set_to_none=True);v=m.losses().sum();v.backward();opt.step()
 f=(a.output/'gpu.csv').open('w')
 mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=f)
 start=time.perf_counter()
 try:
  for j in range(a.epochs):
   if j==90001:lr.fill_(.0001)
   graph.replay()
   if (j+1)%2000==0:
    with torch.no_grad():loss=m.losses().cpu().numpy()
    row=dict(epoch=j+1,seconds=time.perf_counter()-start,min=float(loss.min()),median=float(np.median(loss)),max=float(loss.max()))
    print(json.dumps(row),flush=True)
    with (a.output/'history.jsonl').open('a') as h:h.write(json.dumps(row)+'\n')
  with torch.no_grad():
   loss=m.losses().cpu().numpy(); A,V=[z.cpu().numpy() for z in m.fields()]
  chosen=int(loss.argmin());torch.save(m.state_dict(),a.output/'all-seeds.pt')
  area=A*data['M2'][None,:,None,None];velocity=V*data['U'][None,:,None,None]
  pressure=9986+data['beta'][None,:,:,None]*(np.sqrt(area)-np.sqrt(data['a0'])[None,:,:,None])
  np.savez_compressed(a.output/'fields.npz',area=area,velocity=velocity,pressure=pressure,t=data['t'],length=data['length'],selected=chosen,loss=loss)
  rows=[]
  for v in range(3):
   for end,ix in [(1,0),(max(int(k.split('_')[-1]) for k in data if k.startswith(f'ref{v+1}_')),9)]:
    ref=data[f'ref{v+1}_{end}']
    for name,pred,col in [('velocity',velocity[:,v,ix],3),('area',area[:,v,ix],5)]:
     target=ref[:,col]
     rows.append(dict(vessel=v+1,end=end,variable=name,relative_l2=float(np.linalg.norm(pred[chosen]-target)/np.linalg.norm(target)),r2=float(1-np.sum((pred[chosen]-target)**2)/np.sum((target-target.mean())**2))))
  q=area*velocity
  report=dict(epochs=a.epochs,seeds=a.seeds,selected=chosen,selection='Minimum training objective; no reference fields used for selection',seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(),losses=loss.tolist(),metrics=rows,junction_flow_relative_l2=float(np.linalg.norm(q[chosen,0,-1]-q[chosen,1,0]-q[chosen,2,0])/np.linalg.norm(q[chosen,0,-1])),parity=parity)
  (a.output/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 finally:mon.terminate();mon.wait();f.close()
if __name__=='__main__':main()
