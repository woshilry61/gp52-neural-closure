"""Build every solver buffer and inverse initialization from freshly trained blood."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json
import numpy as np
import torch
from .config import LEGACY_PARAMETERS
from .numerical_closure import shear,solve
from .buffers import Buffers
from ..runtime import tensor
from ..inverse import LOCATIONS,TIMES,second_difference

def weights(grid,points):
 return np.stack([np.interp(points,grid,v)for v in np.eye(len(grid))],0).T
def sample(field,x,t,ox,ot):
 return np.einsum("on,...nt,kt->...ok",weights(x,ox),field,weights(t,ot),optimize=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("--blood",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 raw=dict(np.load(a.blood/"fields.npz"));selection=json.loads((a.blood/"metrics.json").read_text())["selected"]["clean"]
 f={k:raw[k]for k in ["x","xf","t","dt","dx"]}
 f.update({k:raw[k][selection]for k in ["area","face_area","face_flow"]});f["inlet_reservoir"]=np.array(True)
 pars=np.array(LEGACY_PARAMETERS);x=f["x"];t=f["t"];N=len(x);T=len(t);K=64;dt=float(f["dt"]);dx=float(f["dx"])
 amp,_,_=shear(f,pars);u=f["face_flow"]/f["face_area"];s=np.empty_like(u);s[:,0]=u[:,0];decay=np.exp(-dt/pars[2])
 for j in range(1,T):s[:,j]=decay*s[:,j-1]+(1-decay)*u[:,j]
 pack=np.zeros((T,6,K));pack[:,0,:N]=f["area"].T
 for k,v in enumerate([f["face_area"],f["face_flow"],amp,u/.2,s/.2],1):pack[:,k,:N+1]=v.T
 dep=f["xf"][:,None]-u*dt;ix=np.clip(np.searchsorted(f["xf"],dep,side="right")-1,0,N-1)
 w=(dep-f["xf"][ix])/dx;valid=(dep>=f["xf"][0])&(dep<=f["xf"][-1])
 ixpad=np.zeros((T,K),np.int32);w0=np.zeros((T,K));w1=w0.copy()
 ixpad[:,:N+1]=ix.T;w0[:,:N+1]=((1-w)*valid).T;w1[:,:N+1]=(w*valid).T
 initial=np.stack([.2*np.exp(-((x/.06-.25)/.1)**2),np.ones(N),np.zeros(N)])
 knots=np.array([0.,.06,.14,.24,.4,.6,.82]);basis=np.stack([np.interp(t,knots,v)for v in np.eye(7)])
 arrays=dict(F=pack[None],IX=ixpad[None],W0=w0[None],W1=w1[None],IC=initial,basis=basis,x=x,xf=f["xf"],t=t,knots=knots,dt=np.array(dt),dx=np.array(dx),pars=pars)
 old=SimpleNamespace(F=tensor(arrays["F"]),IX=tensor(arrays["IX"],torch.int32),W0=tensor(arrays["W0"]),W1=tensor(arrays["W1"]),IC=tensor(initial),basis=tensor(basis),f=f,N=N,T=T,dt=dt,dx=dx,x=x,t=t,tau=pars[3])
 buffers=Buffers(old)
 arrays.update({"buffer_"+k:v.detach().cpu().numpy()for k,v in buffers.state_dict().items()})
 np.savez_compressed(a.output/"flow.npz",**arrays)
 h0=solve(f,pars,inlet=np.zeros(len(t)))["y"][1]
 hb=np.stack([solve(f,pars,inlet=v)["y"][1]-h0 for v in basis])
 context=dict(x=x,t=t,basis=basis,wx=weights(x,LOCATIONS),wt=weights(t,TIMES),R=second_difference(knots),
              h0=sample(h0,x,t,LOCATIONS,TIMES).ravel(),hb=sample(hb,x,t,LOCATIONS,TIMES).reshape(7,-1).T)
 np.savez_compressed(a.output/"inverse-context.npz",**context)
 (a.output/"provenance.json").write_text(json.dumps({"blood_case_index":selection,"parameters":pars.tolist(),"construction":"New blood fields only; no bundled prepared arrays or learned biochemical weights loaded."},indent=2))
 print(json.dumps({"flow_buffers":len(arrays),"inverse_arrays":len(context),"selected_blood_case":selection}))
if __name__=="__main__":main()
