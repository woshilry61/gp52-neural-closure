"""Topology-driven independent ALE blood CFD, using raw numerical boundary data."""
from pathlib import Path
import argparse,json,time,hashlib,re
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import bmat,coo_matrix,block_diag,csc_matrix
from scipy.sparse.linalg import spsolve
from skfem import MeshTri,FacetBasis,asm
from gp52.raw import cfd_flow as f
from gp52.three_independent.blood import junction_flux,sample as sample_original
from .protocol import ROOT,RUN,LENGTH,COUNTS
from .topology import NETWORKS,junctions
def extract():
 out=RUN/'seven';out.mkdir(exist_ok=True);p=out/'raw-input.npz'
 if p.exists():return
 d=NETWORKS['seven'];fields={};sources={};r0s=[.018206,.015799,.008561,.015187,.004532,.007348,.004899]
 for v,file_id in enumerate(d['source_files']):
  src=ROOT/f'external/sen2024/PIGNN_7_artery/Snaps/sim_1_{file_id}.his'
  text=src.read_text();positions=[float(x) for x in re.findall(r'# Point \d+: x = ([\d.eE+-]+)',text)]
  a=np.loadtxt(src);a=a[(a[:,0]>=4.94)&(a[:,0]<=5.76)]
  raw=np.stack([a[a[:,6]==i+1][::2] for i in range(len(positions))])
  fields[f'v{v}_positions']=np.array(positions);fields[f'v{v}_area']=raw[:,:,4];fields[f'v{v}_velocity']=raw[:,:,2]
  fields['t']=raw[0,:,0]-raw[0,0,0]
  x=np.linspace(0,d['lengths'][v],10);radius=r0s[v]+(-.0033667*x if v==0 else np.zeros_like(x))
  area=3.14*radius**2;eh=.1*(3000000*np.exp(-900*radius)+489871.1841)*radius
  fields[f'v{v}_a0']=area;fields[f'v{v}_beta']=(4/3)*1.772*eh/area
  sources[str(src.relative_to(ROOT))]=hashlib.sha256(src.read_bytes()).hexdigest()
 np.savez_compressed(p,**fields)
 (out/'raw-provenance.json').write_text(json.dumps(dict(sources=sources,columns=dict(time=0,velocity=2,area=4,point_id=6),
   source='Unmodified upstream numerical 1D snapshots; no neural predictions',source_commit='bf0ec89f0aab8a639bdd2d363dc3c0cfae770c1d'),indent=2))
def constraint(v,p,z,rad,external):
 N=v.N+p.N;bc=np.zeros(N);D=[v.get_dofs('wall').all(['u^1']),v.get_dofs('axis').all(['u^2'])]
 for tag,mean in external.items():
  iz=v.get_dofs(tag).all(['u^1']);R=rad[0 if tag=='inlet' else -1];bc[iz]=2*mean*(1-(v.doflocs[1,iz]/R)**2);D.append(iz)
 D=np.unique(np.concatenate(D));target=np.arange(N);scale=np.ones(N);slaves=[]
 for tag in ['inlet','outlet']:
  ir=v.get_dofs(tag).all(['u^2']);master=ir[np.argmax(v.doflocs[1,ir])];ss=ir[ir!=master]
  target[ss]=master;scale[ss]=v.doflocs[1,ss]/rad[0 if tag=='inlet' else -1];slaves.extend(ss)
 facets=v.mesh.boundaries['wall'];mid=v.facet_dofs[1,facets]
 left=v.nodal_dofs[1,v.mesh.facets[0,facets]];right=v.nodal_dofs[1,v.mesh.facets[1,facets]]
 free=np.setdiff1d(np.arange(N),np.union1d(D,np.r_[slaves,mid]));lookup=np.full(N,-1);lookup[free]=np.arange(len(free))
 row=np.setdiff1d(np.arange(N),np.union1d(D,mid));row=row[lookup[target[row]]>=0]
 rows=[row];cols=[lookup[target[row]]];vals=[scale[row]]
 for master in [left,right]:
  active=(lookup[master]>=0)&(~np.isin(mid,D));rows.append(mid[active]);cols.append(lookup[master[active]]);vals.append(np.full(active.sum(),.5))
 return coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(N,len(free))).tocsr(),bc
def initialize(src,name,k,nx,nr):
 positions=src[f'v{k}_positions'];areas=src[f'v{k}_area'];vel=src[f'v{k}_velocity'];length=LENGTH[name][k]
 z=np.linspace(0,length/f.L0,nx+1);base=MeshTri.init_tensor(z,np.linspace(0,1,nr+1))
 base=base.with_boundaries({'inlet':lambda x:np.isclose(x[0],0),'outlet':lambda x:np.isclose(x[0],z[-1]),'axis':lambda x:np.isclose(x[1],0),'wall':lambda x:np.isclose(x[1],1)})
 R=PchipInterpolator(positions/f.L0,np.sqrt(areas[:,0]/np.pi)/f.L0)(z)
 wn=np.where(np.isclose(base.p[1],1))[0];wn=wn[np.argsort(base.p[0,wn])]
 v,p=f.bases(f.mesh_at(base,z,R));uf=PchipInterpolator(positions/f.L0,vel[:,0]/f.U0);rf=PchipInterpolator(z,R)
 zz,rr=v.doflocs;U=uf(zz);dU=uf.derivative()(zz);RR=rf(zz);dR=rf.derivative()(zz);u=np.zeros(v.N);iz,ir=v.split_indices()
 u[iz]=(2*U*(1-rr**2/RR**2))[iz];u[ir]=(-dU*rr+.5*dU*rr**3/RR**2-U*dR*rr**3/RR**3)[ir]
 sx=np.linspace(0,length,10)
 def mechanics(x):return np.interp(x,sx,src[f'v{k}_a0']),np.interp(x,sx,src[f'v{k}_beta'])
 a0,beta=mechanics(p.doflocs[0]*f.L0);pressure=beta*(np.sqrt(np.pi)*rf(p.doflocs[0])*f.L0-np.sqrt(a0))/(f.RHO*f.U0**2)
 waves={};nodes=NETWORKS[name];js=junctions(name)
 for tag,node,index in [('inlet',nodes['inlet_nodes'][k],0),('outlet',nodes['outlet_nodes'][k],-1)]:
  if node not in js:
   ext=vel[index].copy()/f.U0;ext[0]=ext[-1]=.5*(ext[0]+ext[-1]);waves[tag]=PchipInterpolator(src['t'],ext)
 return dict(base=base,z=z,rad=R,u=u,p=pressure,wn=wn,v=v,pbasis=p,mechanics=mechanics,waves=waves)
def step(states,name,dt,t):
 oldrad=[s['rad'].copy() for s in states];oldu=[s['u'].copy() for s in states];radii=[r.copy() for r in oldrad];uv=[u.copy() for u in oldu]
 js=junctions(name);nodes=NETWORKS[name]
 for it in range(40):
  KK=[];rr=[];ff=[];cs=[];bcs=[];bb=[];vp=[]
  for k,s in enumerate(states):
   mesh=f.mesh_at(s['base'],s['z'],radii[k]);v,p=f.bases(mesh);vp.append((v,p));wm=np.zeros(v.N);ri=v.split_indices()[1];z=s['z'];rad=radii[k]
   wm[ri]=np.interp(v.doflocs[0,ri],z,(rad-oldrad[k])/dt)*v.doflocs[1,ri]/np.interp(v.doflocs[0,ri],z,rad)
   M=asm(f.mass,v);vf=v.interpolate(uv[k]);adv=v.interpolate(uv[k]-wm)
   A=M/dt+asm(f.viscous,v)+asm(f.advection,v,a=adv)+asm(f.advection_newton,v,velocity=vf)+asm(f.streamline,v,a=adv,dt=dt)
   rhs=M@oldu[k]/dt+asm(f.advection_rhs,v,velocity=vf);B=asm(f.divergence,v,p);bb.append(B)
   fb=FacetBasis(mesh,v.elem,facets=mesh.boundaries['wall'],intorder=6);xx=fb.global_coordinates()[0]
   a0,beta=s['mechanics'](xx*f.L0);r0=np.sqrt(a0/np.pi)/f.L0;elasticity=beta*np.sqrt(np.pi)*f.L0/(f.RHO*f.U0**2)
   A=A+asm(f.elastic,fb,k=elasticity,dt=dt);rhs=rhs+asm(f.elastic_rhs,fb,k=elasticity,dr=np.interp(xx,z,oldrad[k])-r0)
   K=bmat([[A,-B.T],[-B,None]],format='csr');rhs=np.r_[rhs,np.zeros(p.N)]
   C,bc=constraint(v,p,z,rad,{tag:float(wave(t%.82)) for tag,wave in s['waves'].items()})
   cols=[]
   for node in js:
    flux=np.zeros(v.N+p.N)
    for tag,nid in [('inlet',nodes['inlet_nodes'][k]),('outlet',nodes['outlet_nodes'][k])]:
     if node==nid:
      jb=FacetBasis(mesh,v.elem,facets=mesh.boundaries[tag],intorder=6);flux[:v.N]+=asm(junction_flux,jb)
    cols.append(C.T@flux)
   KK.append(C.T@K@C);rr.append(C.T@(rhs-K@bc));ff.append(np.stack(cols,1));cs.append(C);bcs.append(bc)
  K=block_diag(KK,format='csc');fv=np.concatenate(ff,0);fc=csc_matrix(fv);aug=bmat([[K,fc],[fc.T,None]],format='csc');rhs=np.r_[np.concatenate(rr),np.zeros(len(js))]
  sol=spsolve(aug,rhs);cuts=np.r_[0,np.cumsum([len(r) for r in rr])];newu=[];newp=[];target=[]
  for k,s in enumerate(states):
   a=bcs[k]+cs[k]@sol[cuts[k]:cuts[k+1]];v,p=vp[k];newu.append(a[:v.N]);newp.append(a[v.N:]);target.append(oldrad[k]+dt*a[v.nodal_dofs[1,s['wn']]])
  dr=max(float(abs(a-b).max()) for a,b in zip(target,radii));du=max(float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-20)) for a,b in zip(newu,uv))
  if dr<1e-8 and du<1e-7:break
  radii=target;uv=newu
 if it==39:raise RuntimeError(f'Newton failed at t={t}, dR={dr}, du={du}')
 for k,s in enumerate(states):s.update(rad=radii[k],u=newu[k],p=newp[k],v=vp[k][0],pbasis=vp[k][1])
 return dict(iterations=it+1,radius_change=dr,velocity_change=du,
  junction_flux_abs_m3_s=(abs(fv.T@sol[:-len(js)])*2*np.pi*f.L0**2*f.U0).tolist(),
  weak_divergence=max(float(np.linalg.norm(B@u)) for B,u in zip(bb,newu)),linear_relative_residual=float(np.linalg.norm(aug@sol-rhs)/np.linalg.norm(rhs)))
def sample(s,name,k,nx,nr):
 # The original sampling algorithm is geometry-independent except for its lengths array.
 import gp52.three_independent.blood as old
 old_lengths=old.LENGTH;old.LENGTH=LENGTH[name]
 try:return sample_original(s,k,nx,nr)
 finally:old.LENGTH=old_lengths
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--nx',type=int,nargs=7,default=[24,8,14,16,56,14,38]);p.add_argument('--nr',type=int,default=10);p.add_argument('--dt',type=float,default=.001);p.add_argument('--duration',type=float,default=1.23);p.add_argument('--workers',type=int,default=7);a=p.parse_args()
 extract();a.output.mkdir(parents=True,exist_ok=False);(a.output/'executed-source.py').write_text(Path(__file__).read_text())
 src=dict(np.load(RUN/'seven/raw-input.npz'));name='seven';states=[initialize(src,name,k,a.nx[k],a.nr) for k in range(7)]
 from concurrent.futures import ProcessPoolExecutor
 import multiprocessing
 from . import blood_parallel as parallel
 pool=ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork'),initializer=parallel.initialize_workers,initargs=(states,name)) if a.workers>1 else None
 nout=COUNTS[name]*4;records=[[sample(s,name,k,nout[k],24)] for k,s in enumerate(states)]
 raw=[dict(u=[s['u'].copy()],p=[s['p'].copy()],r=[s['rad'].copy()]) for s in states];ts=np.arange(round(a.duration/a.dt)+1)*a.dt;history=[];start=time.perf_counter()
 for j,t in enumerate(ts[1:],1):
  info=(step(states,name,a.dt/f.T0,t) if pool is None else parallel.step(pool,states,name,a.dt/f.T0,t));history.append(info)
  samples=[sample(s,name,k,nout[k],24) for k,s in enumerate(states)] if pool is None else list(pool.map(parallel.sample_task,[(k,s['rad'],s['u'],s['p'],nout[k],24) for k,s in enumerate(states)]))
  for k,s in enumerate(states):
   records[k].append(samples[k]);raw[k]['u'].append(s['u'].copy());raw[k]['p'].append(s['p'].copy());raw[k]['r'].append(s['rad'].copy())
  if j%10==0:print(json.dumps(dict(step=j,t=t,seconds=time.perf_counter()-start,**info)),flush=True)
 if pool is not None:pool.shutdown()
 fields={}
 for k,s in enumerate(states):
  for key in records[k][0]:fields[f'v{k}_{key}']=records[k][0][key] if key in ['x','xf'] else np.stack([r[key] for r in records[k]])
  np.savez_compressed(a.output/f'finite-elements-{k}.npz',velocity=np.stack(raw[k]['u']),pressure=np.stack(raw[k]['p']),wall_radius=np.stack(raw[k]['r']),wall_x=s['z'],reference_points=s['base'].p,triangles=s['base'].t,t=ts)
 np.savez_compressed(a.output/'flow.npz',**fields,t=ts)
 report=dict(seconds=time.perf_counter()-start,workers=a.workers,nx=a.nx,nr=a.nr,dt=a.dt,duration=a.duration,
  source_sha256=hashlib.sha256((RUN/'seven/raw-input.npz').read_bytes()).hexdigest(),
  model='Seven independent axisymmetric ALE P2/P1 Navier-Stokes tubes; three common-traction constraints enforce junction flow conservation',
  boundary='Original numerical external velocity and initial area/velocity; parabolic initial/external profiles; periodic endpoint continuation; no learned blood',history=history)
 (a.output/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
if __name__=='__main__':main()
