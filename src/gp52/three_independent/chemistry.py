"""Batched CUDA conservative radial transport/reaction on independent CFD blood."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from gp52.three.reference import COUNTS,LENGTH,KNOTS,RATE,inlets,geometry
from .transfer import load_flow,operators,RUN

class CUDASolve:
    def __init__(self):self.history=[];self.checked=False
    def solve(self,A,b,x0):
        scale=1/A.diagonal();As=(diags(scale)@A).tocsr()
        aa=torch.sparse_csr_tensor(torch.as_tensor(As.indptr,device='cuda'),torch.as_tensor(As.indices,device='cuda'),
          torch.as_tensor(As.data,device='cuda'),size=As.shape,dtype=torch.float64)
        bb=b*torch.as_tensor(scale,device='cuda')[:,None];x=x0.clone()
        r=bb-torch.sparse.mm(aa,x);rh=r.clone();p=torch.zeros_like(r);v=torch.zeros_like(r)
        norm=torch.linalg.vector_norm(bb,dim=0).clamp_min(1e-20)
        tol=1e-11*norm;active=torch.linalg.vector_norm(r,dim=0)>tol
        one=torch.ones_like(norm);rho0=one.clone();alpha=one.clone();omega=one.clone()
        start=time.perf_counter()
        for i in range(400):
            rho=(rh*r).sum(0)
            beta=torch.where(active,rho/rho0*alpha/omega,0)
            p=r+beta[None]*(p-omega[None]*v)
            v=torch.sparse.mm(aa,p)
            den=(rh*v).sum(0)
            alpha=torch.where(active,rho/torch.where(den!=0,den,one),0)
            s=r-alpha[None]*v
            short=torch.linalg.vector_norm(s,dim=0)<=tol
            tt=torch.sparse.mm(aa,s);den2=(tt*tt).sum(0)
            omega=torch.where(active&~short,(tt*s).sum(0)/torch.where(den2!=0,den2,one),one)
            x=x+alpha[None]*p+torch.where(active&~short,omega,0)[None]*s
            r=torch.where(short[None],0,s-omega[None]*tt)
            active=active&(torch.linalg.vector_norm(r,dim=0)>tol)
            rho0=torch.where(active,rho,one)
            if i%4==3 and not bool(active.any()):break
        residual=float((torch.linalg.vector_norm(bb-torch.sparse.mm(aa,x),dim=0)/norm).max())
        if residual>3e-10:raise RuntimeError(f'CUDA sparse solve did not converge: {residual}, iterations={i+1}')
        parity=None
        if not self.checked:
            ref=splu(A.tocsc()).solve(b.cpu().numpy())
            parity=float(np.linalg.norm(x.cpu().numpy()-ref)/np.linalg.norm(ref))
            assert parity<3e-9,parity;self.checked=True
        self.history.append(dict(iterations=i+1,residual=residual,cpu_relative_difference=parity,seconds=time.perf_counter()-start))
        return x
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--blood',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--refine',type=int,default=4);ap.add_argument('--radial',type=int,default=24);ap.add_argument('--dt',type=float,default=.001)
    ap.add_argument('--indices',type=int,nargs='+');ap.add_argument('--uniform',action='store_true');ap.add_argument('--parabolic',action='store_true',help='Diagnostic only: replace solved radial profile with a parabola at the same CFD A/Q');a=ap.parse_args()
    torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'executed-source.py').write_text(Path(__file__).read_text())
    n=COUNTS*a.refine;nr=a.radial
    data,t,stats=load_flow(str(a.blood),tuple(n),nr,a.dt);g=geometry(n,nr)
    if a.parabolic:
        for d in data:d['qbins']=d['q'][:,:,None]*g['w'][None,None,:]*g['profile'][None,None,:]
    aa,split=inlets();ids=np.arange(48) if a.indices is None else np.array(a.indices);amps=aa[ids];B=len(ids)
    cin=np.stack([np.interp(t,KNOTS,z) for z in amps])
    h0=.2*np.exp(-((g['x']/.06-.25)/.1)**2)*(g['cell_v']==0)
    H=torch.tensor(np.broadcast_to(np.r_[h0,0][:,None],(g['ns']+1,B)).copy(),device='cuda')
    G=torch.ones_like(H);F=torch.zeros_like(H)
    if a.uniform:H.fill_(1)
    solver=CUDASolve();all_y=[];mass=[];defects=[];constant=[];allvol=[];start=time.perf_counter();volold=None
    log=(a.output/'gpu.csv').open('w');mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=log)
    try:
        for j,tt in enumerate(t):
            L,bc,vn,q,cond=operators(data,j,g,n,nr);vol=torch.as_tensor(vn,device='cuda');bct=torch.as_tensor(bc,device='cuda')
            if j:
                bh=np.ones((3,B)) if a.uniform else np.stack([cin[:,j],np.zeros(B),np.zeros(B)])
                rhsH=volold[:,None]/a.dt*H+bct@torch.as_tensor(bh,device='cuda')
                rhsG=volold[:,None]/a.dt*G+bct@torch.ones((3,B),device='cuda',dtype=torch.float64)
                ans=solver.solve(L+diags(vn/a.dt),torch.cat([rhsH,rhsG],1),torch.cat([H,G],1))
                H=ans[:,:B];gt=ans[:,B:];G=gt*torch.exp(-(0 if a.uniform else RATE)*H*a.dt)
                F[:-1]=volold[:-1,None]/vol[:-1,None]*F[:-1]+gt[:-1]-G[:-1]
                after=((G+F)*vol[:,None]).sum(0)
                # Sum internal edge fluxes cancels; only boundary rows contribute.
                gtc=gt.cpu().numpy()
                balance=after.cpu().numpy()-mass[-1]+a.dt*(L@gtc-bc@np.ones((3,B))).sum(0)
                defects.append(float(abs(balance).max()/vn.sum()))
                const=np.asarray(L.sum(1)).ravel()-bc.sum(1)+(vn-volold.cpu().numpy())/a.dt
                constant.append(float(abs(const).max()*a.dt/(vn.sum()/g['ns'])))
            av=[]
            for v,nv in enumerate(n):
                sl=slice(g['offset'][v],g['offset'][v+1]);w=torch.as_tensor(g['w'],device='cuda')
                raw=torch.stack([H[sl],G[sl],F[sl]]).reshape(3,nv,nr,B)
                radial=(raw*w[None,None,:,None]).sum(2).reshape(3,COUNTS[v],a.refine,B)
                vv=torch.as_tensor(data[v]['volume'][j].reshape(COUNTS[v],a.refine),device='cuda')
                av.append(((radial*vv[None,:,:,None]).sum(2)/vv.sum(1)[None,:,None]).permute(2,0,1))
            all_y.append(torch.cat(av,2).cpu().numpy());mass.append(((G+F)*vol[:,None]).sum(0).cpu().numpy())
            allvol.append(np.concatenate([d['volume'][j].reshape(COUNTS[v],a.refine).sum(1) for v,d in enumerate(data)]))
            volold=vol
            if j%50==0:
                print(json.dumps(dict(step=j,t=tt,seconds=time.perf_counter()-start,minimum=float(min(H.min(),G.min(),F.min())),linear_iterations=solver.history[-1]['iterations'] if j else 0)),flush=True)
        y=np.stack(all_y,-1)
        np.savez_compressed(a.output/'data.npz',y=y,t=t,amplitudes=amps,case_ids=ids,volume=np.stack(allvol))
        info=dict(seconds=time.perf_counter()-start,split=split,case_ids=ids.tolist(),flow=str(a.blood),
          axial_counts=n.tolist(),radial_cells=nr,dt=a.dt,flow_transfer=stats,
          mass_balance_max=max(defects),constant_preservation_residual_max=max(constant),
          uniform_max_error=float(abs(y[:,:2]-1).max()) if a.uniform else None,
          minimum_concentration=float(y.min()),linear_history=solver.history,
          physical_model=('Prescribed parabolic profile at CFD A/Q: diagnostic, not an independent velocity reference' if a.parabolic else 'Independent axisymmetric Navier-Stokes velocity in conservative cross-section-resolved reaction-transport; perfectly mixed zero-volume junction'))
        (a.output/'metrics.json').write_text(json.dumps(info,indent=2));print(json.dumps({k:v for k,v in info.items() if k!='linear_history'}),flush=True)
    finally:mon.terminate();mon.wait();log.close()
if __name__=='__main__':main()
