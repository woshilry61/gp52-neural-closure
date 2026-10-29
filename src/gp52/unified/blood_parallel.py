"""Parallel per-vessel assembly and sampling; same monolithic CFD equations."""
import numpy as np
from scipy.sparse import bmat,block_diag,csc_matrix
from scipy.sparse.linalg import spsolve
from skfem import FacetBasis,asm
from .blood_network import f,constraint,junction_flux,sample
from .topology import NETWORKS,junctions
STATES=None;NAME=None
def initialize_workers(states,name):
 global STATES,NAME
 STATES=states;NAME=name
def assemble(args):
 k,rad,oldrad,oldu,uv,dt,t=args;s=STATES[k];name=NAME;z=s['z']
 mesh=f.mesh_at(s['base'],z,rad);v,p=f.bases(mesh);wm=np.zeros(v.N);ri=v.split_indices()[1]
 wm[ri]=np.interp(v.doflocs[0,ri],z,(rad-oldrad)/dt)*v.doflocs[1,ri]/np.interp(v.doflocs[0,ri],z,rad)
 M=asm(f.mass,v);vf=v.interpolate(uv);adv=v.interpolate(uv-wm)
 A=M/dt+asm(f.viscous,v)+asm(f.advection,v,a=adv)+asm(f.advection_newton,v,velocity=vf)+asm(f.streamline,v,a=adv,dt=dt)
 rhs=M@oldu/dt+asm(f.advection_rhs,v,velocity=vf);B=asm(f.divergence,v,p)
 fb=FacetBasis(mesh,v.elem,facets=mesh.boundaries['wall'],intorder=6);xx=fb.global_coordinates()[0]
 a0,beta=s['mechanics'](xx*f.L0);r0=np.sqrt(a0/np.pi)/f.L0;elasticity=beta*np.sqrt(np.pi)*f.L0/(f.RHO*f.U0**2)
 A=A+asm(f.elastic,fb,k=elasticity,dt=dt);rhs=rhs+asm(f.elastic_rhs,fb,k=elasticity,dr=np.interp(xx,z,oldrad)-r0)
 K=bmat([[A,-B.T],[-B,None]],format='csr');rhs=np.r_[rhs,np.zeros(p.N)]
 C,bc=constraint(v,p,z,rad,{tag:float(wave(t%.82)) for tag,wave in s['waves'].items()})
 cols=[];nodes=NETWORKS[name]
 for node in junctions(name):
  flux=np.zeros(v.N+p.N)
  for tag,nid in [('inlet',nodes['inlet_nodes'][k]),('outlet',nodes['outlet_nodes'][k])]:
   if node==nid:
    jb=FacetBasis(mesh,v.elem,facets=mesh.boundaries[tag],intorder=6);flux[:v.N]+=asm(junction_flux,jb)
  cols.append(C.T@flux)
 return C.T@K@C,C.T@(rhs-K@bc),np.stack(cols,1),C,bc,B,v.N,v.nodal_dofs[1,s['wn']]
def step(pool,states,name,dt,t):
 oldrad=[s['rad'].copy() for s in states];oldu=[s['u'].copy() for s in states];radii=[r.copy() for r in oldrad];uv=[u.copy() for u in oldu];js=junctions(name)
 for it in range(40):
  blocks=list(pool.map(assemble,[(k,radii[k],oldrad[k],oldu[k],uv[k],dt,t) for k in range(len(states))]))
  K=block_diag([b[0] for b in blocks],format='csc');fv=np.concatenate([b[2] for b in blocks],0);fc=csc_matrix(fv)
  aug=bmat([[K,fc],[fc.T,None]],format='csc');rhs=np.r_[np.concatenate([b[1] for b in blocks]),np.zeros(len(js))]
  sol=spsolve(aug,rhs);cuts=np.r_[0,np.cumsum([len(b[1]) for b in blocks])];newu=[];newp=[];target=[]
  for k,b in enumerate(blocks):
   a=b[4]+b[3]@sol[cuts[k]:cuts[k+1]];newu.append(a[:b[6]]);newp.append(a[b[6]:]);target.append(oldrad[k]+dt*a[b[7]])
  dr=max(float(abs(a-b).max()) for a,b in zip(target,radii));du=max(float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-20)) for a,b in zip(newu,uv))
  if dr<1e-8 and du<1e-7:break
  radii=target;uv=newu
 if it==39:raise RuntimeError(f'Newton failed at {t}: dR={dr}, du={du}')
 for k,s in enumerate(states):s.update(rad=radii[k],u=newu[k],p=newp[k])
 return dict(iterations=it+1,radius_change=dr,velocity_change=du,
  junction_flux_abs_m3_s=(abs(fv.T@sol[:-len(js)])*2*np.pi*f.L0**2*f.U0).tolist(),
  weak_divergence=max(float(np.linalg.norm(b[5]@u)) for b,u in zip(blocks,newu)),
  linear_relative_residual=float(np.linalg.norm(aug@sol-rhs)/np.linalg.norm(rhs)))
def sample_task(args):
 k,rad,u,p,nx,nr=args;s=STATES[k].copy()
 v,pb=f.bases(f.mesh_at(s['base'],s['z'],rad));s.update(rad=rad,u=u,p=p,v=v,pbasis=pb)
 return sample(s,NAME,k,nx,nr)
