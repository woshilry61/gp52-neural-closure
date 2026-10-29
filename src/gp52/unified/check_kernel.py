"""Check the larger GPU kernel against the original on the same three-vessel data."""
import json,time
import numpy as np
import torch
from .models import Three
from .protocol import RUN,inputs
from .network_gpu import integrate,features_kernel
from gp52.three.inverse_gpu import integrate as original,features_kernel as old_features
def main():
 m=Three('new',42);T=m.T;B=16
 tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.int32 if np.issubdtype(x.dtype,np.integer) else torch.float64).contiguous()
 d={}
 for k,v in m.data.items():
  if k=='inv':z=np.pad(v,((0,0),(0,192),(0,192)))
  elif k=='hbasis':z=np.pad(v,((0,0),(0,192),(0,0)))
  elif k in ['dec','ha','afad']:z=np.pad(v,((0,0),(0,192),(0,0)))
  elif k in ['old','bcg','vr','area','radius','vel']:z=np.pad(v,((0,0),(0,192)))
  elif k=='volume':continue
  else:
   z=np.pad(v,(0,192))
   if k in ['dx','gden']:z[64:]=1
   if k in ['el','er']:z[64:]=-1
   if k in ['left','right']:z[64:]=np.arange(64,256)
  d[k]=tensor(z)
 amp=torch.tensor(np.c_[np.ones(B),inputs()[:B]],device='cuda',dtype=torch.float64)
 h=(amp@m.Hflat).reshape(B,T,64).contiguous();hp=torch.nn.functional.pad(h,(0,192)).contiguous()
 out=[]
 for K,H,fn,data in [(64,h,original,{k:getattr(m,k) for k in d}),(256,hp,integrate,d)]:
  G=torch.empty_like(H);F=torch.empty_like(H);fac=torch.full_like(H,1.01)
  args=[H]+[data[k] for k in ['inv','old','bcg','vr','dec','ha','afad','el','er','ea','eb','dx']]+[fac,G,F,T,K]
  if K==256:args+=[57]
  torch.cuda.synchronize();start=time.perf_counter()
  fn[(B,)](*args,True,True,num_warps=4 if K==64 else 8)
  torch.cuda.synchronize()
  out.append((G,F));print('kernel',K,time.perf_counter()-start,flush=True)
 err=[float(abs(a-b[:,:,:64]).max()) for a,b in zip(out[0],out[1])]
 assert max(err)<1e-10,err
 fs=[]
 for K,H,G,fn,data in [(64,h,out[0][0],old_features,{k:getattr(m,k) for k in d}),(256,hp,out[1][0],features_kernel,d)]:
  f=torch.empty((B,T,57,10),device='cuda')
  args=[H,G]+[data[k] for k in ['radius','area','vel','left','right','gden']]+[f,T,K]
  if K==256:args+=[57]
  fn[(B,)](*args,num_warps=4 if K==64 else 8);fs.append(f)
 fe=float(abs(fs[0]-fs[1]).max());assert fe<1e-6,fe
 record=dict(integrator_max_absolute=err,features_max_absolute=fe,check='Original and larger kernels on identical frozen three-vessel coefficients and blood')
 (RUN/'seven/kernel-parity.json').write_text(json.dumps(record,indent=2));print(record)
if __name__=='__main__':main()
