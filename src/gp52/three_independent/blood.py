"""Independent three-tube ALE Navier--Stokes, monolithic junction flux constraint.
Only raw numerical boundary/initial values are read; no learned blood fields.
"""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import bmat,coo_matrix,block_diag,csc_matrix
from scipy.sparse.linalg import spsolve
from skfem import MeshTri,FacetBasis,LinearForm,asm
from gp52.raw import cfd_flow as f
from gp52.three.reference import ROOT,LENGTH
@LinearForm
def junction_flux(v,w):return v[0]*w.n[0]*w.x[1]
def constraint(v,p,z,rad,external,mean):
    N=v.N+p.N;bc=np.zeros(N)
    iz=v.get_dofs(external).all(['u^1']);R=rad[0 if external=='inlet' else -1]
    bc[iz]=2*mean*(1-(v.doflocs[1,iz]/R)**2)
    D=np.unique(np.r_[v.get_dofs('wall').all(['u^1']),v.get_dofs('axis').all(['u^2']),iz])
    target=np.arange(N);scale=np.ones(N);slaves=[]
    for tag in ['inlet','outlet']:
        ir=v.get_dofs(tag).all(['u^2']);master=ir[np.argmax(v.doflocs[1,ir])]
        ss=ir[ir!=master];target[ss]=master
        scale[ss]=v.doflocs[1,ss]/rad[0 if tag=='inlet' else -1];slaves.extend(ss)
    facets=v.mesh.boundaries['wall'];mid=v.facet_dofs[1,facets]
    left=v.nodal_dofs[1,v.mesh.facets[0,facets]];right=v.nodal_dofs[1,v.mesh.facets[1,facets]]
    free=np.setdiff1d(np.arange(N),np.union1d(D,np.r_[slaves,mid]));lookup=np.full(N,-1);lookup[free]=np.arange(len(free))
    row=np.setdiff1d(np.arange(N),np.union1d(D,mid));row=row[lookup[target[row]]>=0]
    rows=[row];cols=[lookup[target[row]]];vals=[scale[row]]
    for master in [left,right]:
        active=(lookup[master]>=0)&(~np.isin(mid,D))
        rows.append(mid[active]);cols.append(lookup[master[active]]);vals.append(np.full(active.sum(),.5))
    C=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(N,len(free))).tocsr()
    return C,bc
def initialize(src,k,nx,nr,grading=1.):
    positions=[[0,.03,.04,.06],[0,.01,.02],[0,.017,.034]][k]
    raw=np.stack([src[f'ref{k+1}_{i+1}'] for i in range(len(positions))])
    z=np.linspace(0,LENGTH[k]/f.L0,nx+1)
    base=MeshTri.init_tensor(z,1-(1-np.linspace(0,1,nr+1))**grading)
    base=base.with_boundaries({'inlet':lambda x:np.isclose(x[0],0),'outlet':lambda x:np.isclose(x[0],z[-1]),'axis':lambda x:np.isclose(x[1],0),'wall':lambda x:np.isclose(x[1],1)})
    R=PchipInterpolator(np.array(positions)/f.L0,np.sqrt(raw[:,0,5]/np.pi)/f.L0)(z)
    wn=np.where(np.isclose(base.p[1],1))[0];wn=wn[np.argsort(base.p[0,wn])]
    v,p=f.bases(f.mesh_at(base,z,R))
    uf=PchipInterpolator(np.array(positions)/f.L0,raw[:,0,3]/f.U0);rf=PchipInterpolator(z,R)
    zz,rr=v.doflocs;U=uf(zz);dU=uf.derivative()(zz);RR=rf(zz);dR=rf.derivative()(zz)
    u=np.zeros(v.N);iz,ir=v.split_indices()
    u[iz]=(2*U*(1-rr**2/RR**2))[iz]
    u[ir]=(-dU*rr+.5*dU*rr**3/RR**2-U*dR*rr**3/RR**3)[ir]
    sx=np.linspace(0,LENGTH[k],10)
    def mechanics(x):
        return np.interp(x,sx,src['a0'][k]),np.interp(x,sx,src['beta'][k])
    a0,beta=mechanics(p.doflocs[0]*f.L0)
    pressure=beta*(np.sqrt(np.pi)*rf(p.doflocs[0])*f.L0-np.sqrt(a0))/(f.RHO*f.U0**2)
    external='inlet' if k==0 else 'outlet'
    ext=raw[0 if k==0 else -1,:,3]/f.U0
    ext=ext.copy();ext[0]=ext[-1]=.5*(ext[0]+ext[-1])
    return dict(base=base,z=z,rad=R,u=u,p=pressure,wn=wn,v=v,pbasis=p,
                mechanics=mechanics,external=external,wave=PchipInterpolator(src['t'],ext))
def step(states,dt,t):
    oldrad=[s['rad'].copy() for s in states];oldu=[s['u'].copy() for s in states]
    radii=[r.copy() for r in oldrad];uv=[u.copy() for u in oldu]
    for it in range(35):
        KK=[];rr=[];ff=[];cs=[];bcs=[];bb=[];vp=[]
        for k,s in enumerate(states):
            mesh=f.mesh_at(s['base'],s['z'],radii[k]);v,p=f.bases(mesh);vp.append((v,p))
            wm=np.zeros(v.N);ri=v.split_indices()[1];z=s['z'];rad=radii[k]
            wm[ri]=np.interp(v.doflocs[0,ri],z,(rad-oldrad[k])/dt)*v.doflocs[1,ri]/np.interp(v.doflocs[0,ri],z,rad)
            M=asm(f.mass,v);vf=v.interpolate(uv[k]);adv=v.interpolate(uv[k]-wm)
            A=M/dt+asm(f.viscous,v)+asm(f.advection,v,a=adv)+asm(f.advection_newton,v,velocity=vf)+asm(f.streamline,v,a=adv,dt=dt)
            rhs=M@oldu[k]/dt+asm(f.advection_rhs,v,velocity=vf)
            B=asm(f.divergence,v,p);bb.append(B)
            fb=FacetBasis(mesh,v.elem,facets=mesh.boundaries['wall'],intorder=6);xx=fb.global_coordinates()[0]
            a0,beta=s['mechanics'](xx*f.L0);r0=np.sqrt(a0/np.pi)/f.L0;elasticity=beta*np.sqrt(np.pi)*f.L0/(f.RHO*f.U0**2)
            A=A+asm(f.elastic,fb,k=elasticity,dt=dt)
            rhs=rhs+asm(f.elastic_rhs,fb,k=elasticity,dr=np.interp(xx,z,oldrad[k])-r0)
            K=bmat([[A,-B.T],[-B,None]],format='csr');rhs=np.r_[rhs,np.zeros(p.N)]
            C,bc=constraint(v,p,z,rad,s['external'],float(s['wave'](t%.82)))
            jb=FacetBasis(mesh,v.elem,facets=mesh.boundaries['outlet' if k==0 else 'inlet'],intorder=6)
            flux=np.r_[asm(junction_flux,jb),np.zeros(p.N)]
            KK.append(C.T@K@C);rr.append(C.T@(rhs-K@bc));ff.append(C.T@flux)
            cs.append(C);bcs.append(bc)
        K=block_diag(KK,format='csc');fv=np.concatenate(ff);fc=csc_matrix(fv[:,None])
        aug=bmat([[K,fc],[fc.T,None]],format='csc')
        rhs=np.r_[np.concatenate(rr),0.]
        sol=spsolve(aug,rhs);cuts=np.r_[0,np.cumsum([len(r) for r in rr])]
        newu=[];newp=[];target=[]
        for k,s in enumerate(states):
            a=bcs[k]+cs[k]@sol[cuts[k]:cuts[k+1]];v,p=vp[k]
            newu.append(a[:v.N]);newp.append(a[v.N:])
            target.append(oldrad[k]+dt*a[v.nodal_dofs[1,s['wn']]])
        dr=max(float(abs(a-b).max()) for a,b in zip(target,radii))
        du=max(float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-20)) for a,b in zip(newu,uv))
        if dr<1e-8 and du<1e-7:break
        radii=target;uv=newu
    if it==34:raise RuntimeError(f'Coupled Newton failed at {t}: dR={dr}, du={du}')
    for k,s in enumerate(states):s.update(rad=radii[k],u=newu[k],p=newp[k],v=vp[k][0],pbasis=vp[k][1])
    return dict(iterations=it+1,radius_change=dr,velocity_change=du,
      junction_flux_abs_m3_s=float(abs(fv@sol[:-1])*2*np.pi*f.L0**2*f.U0),
      weak_divergence=max(float(np.linalg.norm(B@u)) for B,u in zip(bb,newu)),
      junction_traction_Pa=float(sol[-1]*f.RHO*f.U0**2+9986.),
      linear_relative_residual=float(np.linalg.norm(aug@sol-rhs)/max(np.linalg.norm(rhs),1e-30)))
def sample(s,k,nx,nr):
    v=s['v'];p=s['pbasis'];z=s['z'];rad=s['rad'];xf=np.linspace(0,LENGTH[k]/f.L0,nx+1)
    x=.5*(xf[:-1]+xf[1:]);xe=np.clip(xf,1e-10,z[-1]-1e-10);R=np.interp(xe,z,rad)
    # 6-point quadrature in each bin, including all P2 radial shape functions.
    q,w=np.polynomial.legendre.leggauss(6);se=np.linspace(0,1,nr+1)
    ss=.5*(se[:-1,None]+se[1:,None])+.5*np.diff(se)[:,None]*q
    ww=.5*np.diff(se)[:,None]*w*ss
    pts=np.stack([np.broadcast_to(xe[:,None,None],(len(xf),nr,6)).ravel(),(R[:,None,None]*ss[None]).ravel()])
    vals=v.interpolator(s['u'])(pts).reshape(2,len(xf),nr,6)*f.U0
    prof=(vals*ww[None,None]).sum(-1)/ww.sum(-1)[None,None]
    af=np.pi*(R*f.L0)**2
    # Exact integral of piecewise-linear radius squared across output cells.
    def primitive(xq):
        idx=np.minimum(np.searchsorted(z,xq,side='right')-1,len(z)-2);delta=xq-z[idx]
        slope=np.diff(rad)/np.diff(z)
        integ=np.r_[0,np.cumsum(np.diff(z)*(rad[:-1]**2+rad[:-1]*rad[1:]+rad[1:]**2)/3)]
        return (integ[idx]+rad[idx]**2*delta+rad[idx]*slope[idx]*delta**2+slope[idx]**2*delta**3/3)*np.pi*f.L0**3
    volume=np.diff(primitive(xf));area=volume/np.diff(xf*f.L0)
    mean=prof[0]@np.diff(se**2);qf=mean*af
    qp,qw=np.polynomial.legendre.leggauss(12);sp=.5*(qp+1);rc=np.interp(x,z,rad)
    pp=p.interpolator(s['p'])(np.stack([np.broadcast_to(x[:,None],(len(x),12)).ravel(),(rc[:,None]*sp).ravel()])).reshape(len(x),12)
    return dict(x=x*f.L0,xf=xf*f.L0,face_area=af,area=area,volume=volume,
                face_flow=qf,radial_axial_velocity=prof[0],radial_velocity=prof[1],
                pressure=pp@(qw*sp)*f.RHO*f.U0**2+9986.)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path,default=ROOT/'runs/sen-input/input.npz')
    ap.add_argument('--nx',type=int,nargs=3,default=[32,12,20]);ap.add_argument('--nr',type=int,default=12)
    ap.add_argument('--dt',type=float,default=.001);ap.add_argument('--duration',type=float,default=1.23);ap.add_argument('--grading',type=float,default=1.)
    ap.add_argument('--sample-refine',type=int,default=4);ap.add_argument('--sample-radial',type=int,default=24)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'executed-source.py').write_text(Path(__file__).read_text())
    srcpath=a.source;src=dict(np.load(srcpath))
    states=[initialize(src,k,a.nx[k],a.nr,a.grading) for k in range(3)]
    nout=np.array([30,10,17])*a.sample_refine
    records=[[sample(s,k,nout[k],a.sample_radial)] for k,s in enumerate(states)]
    raw=[dict(u=[s['u'].copy()],p=[s['p'].copy()],r=[s['rad'].copy()]) for s in states]
    ts=np.arange(round(a.duration/a.dt)+1)*a.dt;history=[];start=time.perf_counter()
    for j,t in enumerate(ts[1:],1):
        info=step(states,a.dt/f.T0,t);history.append(info)
        for k,s in enumerate(states):
            records[k].append(sample(s,k,nout[k],a.sample_radial))
            raw[k]['u'].append(s['u'].copy());raw[k]['p'].append(s['p'].copy());raw[k]['r'].append(s['rad'].copy())
        if j%10==0:print(json.dumps(dict(step=j,t=t,seconds=time.perf_counter()-start,**info)),flush=True)
    flow={}
    for k,s in enumerate(states):
        for key in records[k][0]:
            flow[f'v{k}_{key}']=records[k][0][key] if key in ['x','xf'] else np.stack([r[key] for r in records[k]],axis=0)
        np.savez_compressed(a.output/f'finite-elements-{k}.npz',velocity=np.stack(raw[k]['u']),pressure=np.stack(raw[k]['p']),
          wall_radius=np.stack(raw[k]['r']),wall_x=s['z'],reference_points=s['base'].p,triangles=s['base'].t,t=ts)
    np.savez_compressed(a.output/'flow.npz',**flow,t=ts)
    info=dict(seconds=time.perf_counter()-start,nx=a.nx,nr=a.nr,radial_grading=a.grading,dt=a.dt,duration=a.duration,
      source_sha256=hashlib.sha256(srcpath.read_bytes()).hexdigest(),
      model='Three axisymmetric ALE P2/P1 Navier-Stokes tubes with massless radial elastic walls; common junction normal traction is a Lagrange multiplier imposing exact net flux zero',
      boundary='Raw numerical initial A/u and external endpoint mean-velocity waveforms; parabolic external profiles; periodic continuation; no learned blood fields',
      junction='Zero-volume common normal traction, exact flow sum; no angle, vortex or explicit junction volume',
      history=history)
    (a.output/'metrics.json').write_text(json.dumps(info,indent=2));print(json.dumps({k:v for k,v in info.items() if k!='history'}),flush=True)
if __name__=='__main__':main()
