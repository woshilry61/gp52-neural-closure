"""Fair 1-D rollout: same base FV operators, volumes, boundary conditions and reaction splitting."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from .reference import ROOT,COUNTS,LENGTH,RATE,flow,geometry,operators,inlets,KNOTS
from .closure import Closure,arrays,reaction_features
def solve(amp=None,tau=None,factor=None,uniform=False,extra_diffusion=None):
 n=COUNTS;blood,t=flow();dt=t[1];g=geometry(n,1);ns=g['ns'];N=ns+1
 amplitudes,split=inlets();B=len(amplitudes)
 cin=np.stack([np.interp(t,KNOTS,a) for a in amplitudes])
 x=g['x'];h0=.2*np.exp(-((x/.06-.25)/.1)**2)*(g['cell_v']==0)
 H=np.broadcast_to(np.r_[h0,0.][:,None],(N,B)).copy();G=np.ones_like(H);Fn=np.zeros_like(H)
 if uniform:H[:]=1.
 idx=np.array([i for i,z in enumerate(g['tags']) if z[0]=='axial']);a,b=g['edges'][idx].T;E=len(idx)
 dx=np.concatenate([np.full(nv-1,LENGTH[v]/nv) for v,nv in enumerate(n)])
 M=np.zeros((2,E,2,B));out=[];mass=[];defects=[];volume=[];minimum=0.;start=time.perf_counter();volold=None
 for j,tt in enumerate(t):
  L,bc,vol,q,cond=operators(blood,j,g,n,1)
  if j:
   if extra_diffusion is not None:
    af=np.concatenate([d['af'][j,1:-1] for d in blood])
    ex=af*extra_diffusion[j]/dx
    L=L+coo_matrix((np.r_[ex,-ex,-ex,ex],(np.r_[a,a,b,b],np.r_[a,b,a,b])),shape=(N,N)).tocsr()
   if amp is not None:
    am=amp[j];ta=tau[j];dec=np.exp(-dt/ta);hh=-ta*np.expm1(-dt/ta)
    af=np.concatenate([d['af'][j,1:-1] for d in blood])
    extra=af*np.sum(hh*am**2,axis=1)/dx
    L=L+coo_matrix((np.r_[extra,-extra,-extra,extra],(np.r_[a,a,b,b],np.r_[a,b,a,b])),shape=(N,N)).tocsr()
    memflux=af[None,:,None]*(am[None,:,:,None]*dec[None,:,:,None]*M).sum(2)
    rhs_mem=np.zeros((2,N,B))
    for k in [0,1]:
     np.add.at(rhs_mem[k],a,-memflux[k]);np.add.at(rhs_mem[k],b,memflux[k])
   else:rhs_mem=np.zeros((2,N,B))
   mat=L+diags(vol/dt);lu=splu(mat.tocsc())
   bH=np.ones((3,B)) if uniform else np.stack([cin[:,j],np.zeros(B),np.zeros(B)])
   rhsH=volold[:,None]/dt*H+bc@bH+rhs_mem[0]
   rhsG=volold[:,None]/dt*G+bc@np.ones((3,B))+rhs_mem[1]
   ans=lu.solve(np.c_[rhsH,rhsG]);H=ans[:,:B];GT=ans[:,B:]
   if amp is not None:
    for k,state in enumerate([H,GT]):
     gradient=(state[b]-state[a])/dx[:,None]
     M[k]=dec[:,:,None]*M[k]-hh[:,:,None]*am[:,:,None]*gradient[:,None,:]
   rate=0. if uniform else RATE
   fac=np.ones((N,B))
   if factor is not None:fac[:ns]=factor[:,j].T
   G=GT*np.exp(-rate*H*fac*dt)
   Fn[:ns]=volold[:ns,None]/vol[:ns,None]*Fn[:ns]+GT[:ns]-G[:ns];Fn[-1]=0
   after=((G+Fn)*vol[:,None]).sum(0)
   balance=after-mass[-1]+dt*(L@GT-bc@np.ones((3,B))-rhs_mem[1]).sum(0)
   defects.append(float(abs(balance).max()/vol.sum()))
  out.append(np.stack([H[:ns],G[:ns],Fn[:ns]]).transpose(2,0,1))
  mass.append(((G+Fn)*vol[:,None]).sum(0));volume.append(vol[:ns]);volold=vol
  minimum=min(minimum,float(min(H.min(),G.min(),Fn.min())))
 return np.stack(out,-1),dict(seconds=time.perf_counter()-start,mass_balance_max=max(defects),minimum=minimum,uniform_max_error=float(abs(np.stack(out,-1)[:,:2]-1).max()) if uniform else None),np.stack(volume).T
def metrics(pred,truth,vol,t,ids):
 result={}
 for window in [.82,1.23]:
  mask=t<=window+1e-9;w=vol[:,mask]
  # Advanced indexing is applied in separate steps to keep case/species/cell/time axes.
  err=np.sqrt((((pred[ids]-truth[ids])[...,mask])**2*w[None,None]).sum((2,3))/((truth[ids][...,mask]**2)*w[None,None]).sum((2,3)))
  result[str(window)]=dict(mean_relative_l2=err.mean(0).tolist(),std_relative_l2=err.std(0,ddof=1).tolist(),per_case=err.tolist())
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 ar=arrays();m=Closure(ar).cuda();m.load_state_dict(torch.load(a.model/'model.pt',weights_only=True));m.eval()
 with torch.no_grad():amp,tau=[v.cpu().numpy() for v in m.coefficients()]
 truth=dict(np.load(a.reference));ty=truth['y'];tt=truth['t']
 if len(tt)!=len(ar['t']):
  assert len(tt)==2*len(ar['t'])-1;ty=ty[...,::2];tt=tt[::2]
 plain,mp,vol=solve()
 trans,mt,_=solve(amp,tau)
 def factors(y):
  feat=reaction_features(y,ar,.002)
  with torch.no_grad():return m.factor(torch.tensor(feat,device='cuda',dtype=torch.float32)).cpu().numpy()
 combined,mc,_=solve(amp,tau,factors(trans))
 reaction_only,mr,_=solve(factor=factors(plain))
 uniform,mu,_=solve(amp,tau,uniform=True)
 assert mu['uniform_max_error']<1e-10,mu
 reports={}
 for name,y,run in [('ordinary_1d',plain,mp),('neural_transport',trans,mt),('neural_reaction_only',reaction_only,mr),('neural_combined',combined,mc)]:
  reports[name]=dict(rollout=run,test=metrics(y,ty,vol,tt,np.arange(32,48)),validation=metrics(y,ty,vol,tt,np.arange(24,32)))
  np.savez_compressed(a.output/f'{name}.npz',y=y,t=tt,volume=vol)
 reports['uniform_check']=mu
 (a.output/'metrics.json').write_text(json.dumps(reports,indent=2))
 print(json.dumps({k:{'Fn':v['test']['0.82']['mean_relative_l2'][2],'Th':v['test']['0.82']['mean_relative_l2'][0],'minimum':v['rollout']['minimum']} for k,v in reports.items() if k!='uniform_check'}),flush=True)
if __name__=='__main__':main()
