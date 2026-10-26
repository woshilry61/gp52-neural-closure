"""Axisymmetric P2/P1 ALE Navier-Stokes with radial elastic-wall traction."""
from pathlib import Path
import argparse,json,time,sys,shutil
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import bmat,coo_matrix
from scipy.sparse.linalg import spsolve
from skfem import MeshTri,Basis,FacetBasis,ElementVector,ElementTriP2,ElementTriP1,BilinearForm,LinearForm,asm
from .numerics import mechanics
CONVECTION=1.
STABILIZATION=1.
RELAXATION=1.
L0=.018206;U0=.2;T0=L0/U0;RHO=1060.;NU=.0025/RHO/(L0*U0)
@BilinearForm
def mass(u,v,w):return (u[0]*v[0]+u[1]*v[1])*w.x[1]
@BilinearForm
def viscous(u,v,w):
    r=w.x[1];ug=u.grad;vg=v.grad
    return 2*NU*(ug[0,0]*vg[0,0]+ug[1,1]*vg[1,1]+u[1]*v[1]/r**2+.5*(ug[0,1]+ug[1,0])*(vg[0,1]+vg[1,0]))*r
@BilinearForm
def divergence(u,q,w):return q*(u.grad[0,0]+u.grad[1,1]+u[1]/w.x[1])*w.x[1]
@BilinearForm
def advection(u,v,w):
    return ((w.a[0]*u.grad[0,0]+w.a[1]*u.grad[0,1])*v[0]+(w.a[0]*u.grad[1,0]+w.a[1]*u.grad[1,1])*v[1])*w.x[1]
@BilinearForm
def advection_newton(u,v,w):
    g=w.velocity.grad
    return ((u[0]*g[0,0]+u[1]*g[0,1])*v[0]+(u[0]*g[1,0]+u[1]*g[1,1])*v[1])*w.x[1]
@LinearForm
def advection_rhs(v,w):
    u=w.velocity;g=u.grad
    return ((u[0]*g[0,0]+u[1]*g[0,1])*v[0]+(u[0]*g[1,0]+u[1]*g[1,1])*v[1])*w.x[1]
@BilinearForm
def streamline(u,v,w):
    a=w.a;speed2=a[0]**2+a[1]**2
    tau=w.dt/np.sqrt(4+4*w.dt**2*speed2/w.h**2)
    du0=a[0]*u.grad[0,0]+a[1]*u.grad[0,1];du1=a[0]*u.grad[1,0]+a[1]*u.grad[1,1]
    dv0=a[0]*v.grad[0,0]+a[1]*v.grad[0,1];dv1=a[0]*v.grad[1,0]+a[1]*v.grad[1,1]
    return tau*(du0*dv0+du1*dv1)*w.x[1]
@BilinearForm
def elastic(u,v,w):return w.k*w.dt*u[1]*v[1]*w.n[1]*w.x[1]
@LinearForm
def elastic_rhs(v,w):return -w.k*w.dr*v[1]*w.n[1]*w.x[1]

def initial_geometry(nx,nr,source):
    z=np.linspace(0,.06/L0,nx+1);s=np.linspace(0,1,nr+1)
    base=MeshTri.init_tensor(z,s)
    base=base.with_boundaries({'inlet':lambda x:np.isclose(x[0],0),'outlet':lambda x:np.isclose(x[0],z[-1]),'axis':lambda x:np.isclose(x[1],0),'wall':lambda x:np.isclose(x[1],1)})
    rad=PchipInterpolator(source['x_m']/L0,np.sqrt(source['area_m2'][:,0]/np.pi)/L0)(z)
    wall_nodes=np.where(np.isclose(base.p[1],1))[0];wall_nodes=wall_nodes[np.argsort(base.p[0,wall_nodes])]
    return base,z,rad,wall_nodes
def mesh_at(base,z,rad):
    pts=base.p.copy();pts[1]*=np.interp(pts[0],z,rad)
    return MeshTri(pts,base.t,sort_t=False).with_boundaries(base.boundaries)
def bases(mesh):
    return Basis(mesh,ElementVector(ElementTriP2()),intorder=6),Basis(mesh,ElementTriP1(),intorder=6)
def matrix_constraint(v,p,z,rad,uin,uout,rigid=False):
    N=v.N+p.N;bc=np.zeros(N)
    comp=v.split_indices();uz,ur=comp
    wallz=v.get_dofs('wall').all(['u^1']);axisr=v.get_dofs('axis').all(['u^2'])
    ends=[]
    for tag,uval in [('inlet',uin),('outlet',uout)]:
        iz=v.get_dofs(tag).all(['u^1']);ir=v.get_dofs(tag).all(['u^2'])
        R=rad[0 if tag=='inlet' else -1];bc[iz]=2*uval*(1-(v.doflocs[1,iz]/R)**2)
        master=ir[np.argmax(v.doflocs[1,ir])];ends.append((ir,master,R))
    D=np.unique(np.r_[wallz,axisr,v.get_dofs('inlet').all(['u^1']),v.get_dofs('outlet').all(['u^1'])])
    if rigid:D=np.unique(np.r_[D,v.get_dofs('wall').all(['u^2']),v.N+p.get_dofs('outlet').all()[0]])
    target=np.arange(N);scale=np.ones(N);slaves=[]
    for ir,master,R in ends:
        ss=ir[ir!=master];target[ss]=master;scale[ss]=v.doflocs[1,ss]/R;slaves.extend(ss)
    # Radial wall velocity is P1, matching the moving P1 mesh geometry.
    facets=v.mesh.boundaries['wall'];wall_mid=v.facet_dofs[1,facets]
    wall_left=v.nodal_dofs[1,v.mesh.facets[0,facets]];wall_right=v.nodal_dofs[1,v.mesh.facets[1,facets]]
    free=np.setdiff1d(np.arange(N),np.union1d(D,np.r_[slaves,wall_mid]));lookup=np.full(N,-1);lookup[free]=np.arange(len(free))
    row=np.setdiff1d(np.arange(N),np.union1d(D,wall_mid));row=row[lookup[target[row]]>=0]
    rows=[row];cols=[lookup[target[row]]];vals=[scale[row]]
    for masters in [wall_left,wall_right]:
        active=(lookup[masters]>=0)&(~np.isin(wall_mid,D))
        rows.append(wall_mid[active]);cols.append(lookup[masters[active]]);vals.append(np.full(active.sum(),.5))
    C=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(N,len(free))).tocsr()
    return C,bc

def step(base,z,oldrad,old_u,oldguess,dt,uin,uout,wall_nodes,rigid=False,steady=False):
    rad=oldrad.copy();uv=oldguess.copy()
    for it in range(50):
        mesh=mesh_at(base,z,rad);v,p=bases(mesh)
        wm=np.zeros(v.N);ri=v.split_indices()[1]
        wm[ri]=np.interp(v.doflocs[0,ri],z,(rad-oldrad)/dt)*v.doflocs[1,ri]/np.interp(v.doflocs[0,ri],z,rad)
        M=asm(mass,v);vfield=v.interpolate(uv);adv=v.interpolate(uv-wm);A=asm(viscous,v)+CONVECTION*(asm(advection,v,a=adv)+asm(advection_newton,v,velocity=vfield)+STABILIZATION*asm(streamline,v,a=adv,dt=dt));B=asm(divergence,v,p)
        rhs=CONVECTION*asm(advection_rhs,v,velocity=vfield)
        if not steady:A=A+M/dt;rhs=rhs+M@old_u/dt
        if not rigid:
            fb=FacetBasis(mesh,v.elem,facets=mesh.boundaries['wall'],intorder=6);xx=fb.global_coordinates()[0]
            a0,beta=mechanics(xx*L0);r0=np.sqrt(a0/np.pi)/L0
            k=beta*np.sqrt(np.pi)*L0/(RHO*U0**2)
            dr=np.interp(xx,z,oldrad)-r0
            A=A+asm(elastic,fb,k=k,dt=dt);rhs=rhs+asm(elastic_rhs,fb,k=k,dr=dr)
        K=bmat([[A,-B.T],[-B,None]],format='csr');rhs=np.r_[rhs,np.zeros(p.N)]
        C,bc=matrix_constraint(v,p,z,rad,uin,uout,rigid)
        answer=bc+C@spsolve((C.T@K@C).tocsc(),C.T@(rhs-K@bc))
        newu=answer[:v.N];pressure=answer[v.N:]
        target=oldrad if rigid else oldrad+dt*newu[v.nodal_dofs[1,wall_nodes]]
        change=float(np.max(abs(target-rad)));uchange=float(np.linalg.norm(newu-uv)/max(np.linalg.norm(newu),1e-30))
        if change<1e-8 and uchange<1e-7:
            uv=newu
            break
        uv=RELAXATION*newu+(1-RELAXATION)*uv
        rad=RELAXATION*target+(1-RELAXATION)*rad
    residual=float(np.linalg.norm(B@newu))
    return rad,newu,pressure,dict(iterations=it+1,wall_iteration_change=change,velocity_iteration_change=uchange,weak_divergence=residual),v,p

def sample(v,p,uv,pressure,z,rad,nxout=41,nrout=32):
    xf=np.linspace(0,.06/L0,nxout+1);x=(xf[1:]+xf[:-1])/2
    radius=np.interp(xf,z,rad);rc=np.interp(x,z,rad)
    q,w=np.polynomial.legendre.leggauss(4);edges=np.linspace(0,1,nrout+1);ss=(edges[:-1,None]+edges[1:,None])/2+(edges[1:,None]-edges[:-1,None])*q/2
    ww=w[None,:]*(edges[1:,None]-edges[:-1,None])/2*ss
    zz=np.broadcast_to(xf[:,None,None],(len(xf),nrout,4));rr=radius[:,None,None]*ss[None,:,:]
    points=np.stack([zz.ravel(),rr.ravel()])
    vals=v.interpolator(uv)(points).reshape(2,len(xf),nrout,4)
    prof=(vals*ww[None,None,:,:]).sum(-1)/ww.sum(-1)[None,None,:]*U0
    mean=np.sum(prof[0]*np.diff(edges**2)[None,:],axis=1)
    area=np.pi*(radius*L0)**2;cellarea=np.pi*(rc*L0)**2
    qq,qw=np.polynomial.legendre.leggauss(24);ssq=.5*(qq+1);ppts=np.stack([np.broadcast_to(x[:,None],(len(x),24)).ravel(),(rc[:,None]*ssq[None,:]).ravel()]);pv=p.interpolator(pressure)(ppts).reshape(len(x),24);pp=(pv*(qw*ssq)[None,:]).sum(1)*RHO*U0**2+9986.
    return dict(face_area=area,area=cellarea,face_flow=area*mean,face_velocity=mean,radial_axial_velocity=prof[0],radial_velocity=prof[1],pressure=pp,x=x*L0,xf=xf*L0)

def poiseuille():
    source={'x_m':np.array([0.,.06]),'area_m2':np.full((2,1),np.pi*L0**2)}
    base,z,rad,wn=initial_geometry(12,6,source);mesh=mesh_at(base,z,rad);v,p=bases(mesh)
    u=np.zeros(v.N);ix=v.split_indices()[0];u[ix]=2*(1-v.doflocs[1,ix]**2)
    r,sol,pr,info,v,p=step(base,z,rad,u,u,1.,1.,1.,wn,rigid=True,steady=True)
    refp=8*NU*(z[-1]-p.doflocs[0])
    return dict(velocity_relative_l2=float(np.linalg.norm(sol-u)/np.linalg.norm(u)),pressure_relative_l2=float(np.linalg.norm(pr-refp)/np.linalg.norm(refp)),**info)

def main():
    global CONVECTION,STABILIZATION,RELAXATION
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--nx',type=int,default=24);ap.add_argument('--nr',type=int,default=8);ap.add_argument('--dt',type=float,default=.002);ap.add_argument('--duration',type=float,default=.82);ap.add_argument('--poiseuille',action='store_true');ap.add_argument('--stokes',action='store_true');ap.add_argument('--stabilization',type=float,default=1.);ap.add_argument('--relaxation',type=float,default=1.);ap.add_argument('--velocity-scale',type=float,default=1.);ap.add_argument('--harmonic',type=float,default=0.)
    a=ap.parse_args();CONVECTION=0. if a.stokes else 1.;STABILIZATION=a.stabilization;RELAXATION=a.relaxation;a.output.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,a.output/'executed-source.py')
    if a.poiseuille:
        r=poiseuille();print(json.dumps(r));(a.output/'poiseuille.json').write_text(json.dumps(r,indent=2));return
    src=dict(np.load(a.source));source_time=src['time_s']-src['time_s'][0];src['velocity_m_s']=a.velocity_scale*src['velocity_m_s']+a.harmonic*np.max(abs(src['velocity_m_s']))*np.sin(4*np.pi*source_time/source_time[-1])[None,:];base,z,rad,wn=initial_geometry(a.nx,a.nr,src)
    mesh=mesh_at(base,z,rad);v,p=bases(mesh)
    uf=PchipInterpolator(src['x_m']/L0,src['velocity_m_s'][:,0]/U0);rf=PchipInterpolator(z,rad)
    zz,rr=v.doflocs;U=uf(zz);dU=uf.derivative()(zz);RR=rf(zz);dR=rf.derivative()(zz)
    u=np.zeros(v.N);iz,ir=v.split_indices()
    u[iz]=(2*U*(1-rr**2/RR**2))[iz]
    u[ir]=(-dU*rr+.5*dU*rr**3/RR**2-U*dR*rr**3/RR**3)[ir]
    a0,beta=mechanics(p.doflocs[0]*L0);pressure=beta*(np.sqrt(np.pi)*rf(p.doflocs[0])*L0-np.sqrt(a0))/(RHO*U0**2)
    times=np.linspace(0,a.duration,round(a.duration/a.dt)+1);dt=(times[1]-times[0])/T0
    source_t=src['time_s']-src['time_s'][0];uin=PchipInterpolator(source_t,src['velocity_m_s'][0]/U0)(times);uout=PchipInterpolator(source_t,src['velocity_m_s'][-1]/U0)(times)
    records=[sample(v,p,u,pressure,z,rad)];raw_u=[u.copy()];raw_p=[pressure.copy()];raw_r=[rad.copy()];history=[];start=time.perf_counter()
    for j in range(1,len(times)):
        rad,u,pressure,info,v,p=step(base,z,rad,u,u,dt,uin[j],uout[j],wn)
        records.append(sample(v,p,u,pressure,z,rad));raw_u.append(u.copy());raw_p.append(pressure.copy());raw_r.append(rad.copy());history.append(info)
        if j%10==0:print(json.dumps(dict(step=j,t=float(times[j]),seconds=time.perf_counter()-start,radius_min=float(rad.min()*L0),radius_max=float(rad.max()*L0),**info)),flush=True)
    fields={k:np.stack([r[k] for r in records],axis=-1) for k in records[0] if k not in ['x','xf']}
    fields.update(x=records[0]['x'],xf=records[0]['xf'],t=times,dt=times[1]-times[0],dx=.06/41)
    np.savez_compressed(a.output/'flow.npz',**fields)
    np.savez_compressed(a.output/'finite-elements.npz',velocity=np.stack(raw_u,1),pressure=np.stack(raw_p,1),wall_radius=np.stack(raw_r,1),wall_x=z,reference_points=base.p,triangles=base.t,t=times)
    volume=np.sum(fields['area'],axis=0)*float(fields['dx']);net=fields['face_flow'][0]-fields['face_flow'][-1]
    defect=volume-volume[0]-np.r_[0,np.cumsum((times[1]-times[0])*net[1:])]
    report=dict(velocity_scale=a.velocity_scale,harmonic=a.harmonic,convection=bool(CONVECTION),streamline_stabilization=STABILIZATION if CONVECTION else 0.,newton_relaxation=RELAXATION,seconds=time.perf_counter()-start,nx=a.nx,nr=a.nr,dt=float(times[1]-times[0]),max_relative_volume_balance=float(np.max(abs(defect))/volume[0]),nonlinear_history=history,physical_model=('Axisymmetric Navier-Stokes' if CONVECTION else 'Axisymmetric unsteady Stokes')+', P2/P1 ALE, radial quasi-static elastic-wall traction matching the Sen pressure-area law; prescribed axial end means and radially dilating end constraints',reference_inputs='Independent numerical initial A,u at four positions and endpoint mean-velocity waveforms; no PIGNN outputs',reference_initialization='Interpolated initial area and velocity, parabolic initial axial profile, divergence-compatible initial radial velocity',limitations='Radial-only massless wall law; P1 moving mesh; input initial radial profile assumed from the 1D model')
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='nonlinear_history'}),flush=True)
if __name__=='__main__':main()
