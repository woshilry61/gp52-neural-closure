"""Same independent ALE finite elements, batched inlets and direct reaction moments.
No change to fluid field, mesh, or chemical equations; assemble shared operators once.
"""
from pathlib import Path
import argparse,json,time,sys,shutil
import numpy as np
from scipy.sparse import diags,coo_matrix,triu
from scipy.sparse.linalg import splu
from skfem import MeshTri,Basis,FacetBasis,ElementTriP1,ElementTriP2,ElementVector,BilinearForm,asm
from .cfd_flow import L0,U0,T0,mesh_at,bases
@BilinearForm
def scalar_mass(u,v,w):return u*v*w.x[1]
@BilinearForm
def diffusion(u,v,w):return w.D*(u.grad[0]*v.grad[0]+u.grad[1]*v.grad[1])*w.x[1]
@BilinearForm
def transport(u,v,w):return -u*(w.a[0]*v.grad[0]+w.a[1]*v.grad[1])*w.x[1]
@BilinearForm
def outflow(u,v,w):
    an=w.a[0]*w.n[0]+w.a[1]*w.n[1]
    return np.maximum(an,0)*u*v*w.x[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--flow',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--refine',type=int,default=1);ap.add_argument('--diffusion',type=float,default=2e-4);ap.add_argument('--uniform',action='store_true');ap.add_argument('--inlets',type=Path,nargs='+',required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,a.output/'executed-source.py')
    raw=dict(np.load(a.flow/'finite-elements.npz'));fl=dict(np.load(a.flow/'flow.npz'))
    base=MeshTri(raw['reference_points'],raw['triangles'],sort_t=False)
    base=base.with_boundaries({'inlet':lambda x:np.isclose(x[0],0),'outlet':lambda x:np.isclose(x[0],.06/L0),'axis':lambda x:np.isclose(x[1],0),'wall':lambda x:np.isclose(x[1],1)})
    z=raw['wall_x'];times=raw['t'];dt=(times[1]-times[0])/T0;D=a.diffusion/(U0*L0);rate=0. if a.uniform else 2.*T0
    custom=[dict(np.load(p)) for p in a.inlets]; B=len(custom)
    state=None;massold=None;coordold=None;average=[];direct=[];discrete=[];variance=[];masshistory=[];consistency=[];projection=None;flux_correction=[];constant_defects=[];boundary_projection=[]
    xout=fl['x']/L0;q,qw=np.polynomial.legendre.leggauss(32);ss=.5*(q+1);avw=qw*ss
    start=time.perf_counter()
    for j,t in enumerate(times):
        mesh=mesh_at(base,z,raw['wall_radius'][:,j]);vb,_=bases(mesh)
        fine=mesh.refined(a.refine);b=Basis(fine,ElementTriP1(),intorder=6);vf=Basis(fine,ElementVector(ElementTriP2()),intorder=6)
        vi=vf.split_indices();points=vf.doflocs[:,vi[0]]
        if projection is None:
            coarse_ref=Basis(base,ElementVector(ElementTriP2()))
            fine_ref=Basis(base.refined(a.refine),ElementVector(ElementTriP2()))
            projection=coarse_ref.probes(fine_ref.doflocs[:,fine_ref.split_indices()[0]])
        val=(projection@raw['velocity'][:,j]).reshape(2,len(vi[0]));uv=np.zeros(vf.N);uv[vi[0]]=val[0];uv[vi[1]]=val[1]
        mass=np.asarray(asm(scalar_mass,b).sum(1)).ravel()
        if state is None:
            h0=.2*np.exp(-((b.doflocs[0]/(.06/L0)-.25)/.1)**2);state=np.broadcast_to(np.stack([np.zeros(b.N),h0,np.ones(b.N),np.zeros(b.N)]),(B,4,b.N)).copy()
            if a.uniform:state[:]=1.
        else:
            wm=np.zeros(vf.N);wm[vi[0]]=(points[0]-coordold[0])/dt;wm[vi[1]]=(points[1]-coordold[1])/dt
            adv=vf.interpolate(uv-wm);op=asm(diffusion,b,D=D)+asm(transport,b,a=adv)
            ends=np.r_[fine.boundaries['inlet'],fine.boundaries['outlet']]
            fb=FacetBasis(fine,b.elem,facets=ends,intorder=6);fv=FacetBasis(fine,vf.elem,facets=ends,intorder=6)
            op=op+asm(outflow,fb,a=fv.interpolate(uv-wm))
            inlet=b.get_dofs('inlet').all()  # Persistent upstream concentration reservoir
            outlet=b.get_dofs('outlet').all() if fl['face_velocity'][-1,j]<0 else np.array([],dtype=int)
            fixed=np.union1d(inlet,outlet);free=np.setdiff1d(np.arange(b.N),fixed)
            # Conservative H(div)-compatible algebraic flux projection.
            # Coarse Taylor-Hood incompressibility does not hold against refined
            # scalar tests. Enforce Mdot + L*1 = 0 by internal antisymmetric edge
            # fluxes, with inflow nodes as the fixed reservoir. Unlike a diagonal
            # concentration correction, each edge contributes equal/opposite mass.
            defect=np.asarray(op.sum(1)).ravel()+(mass-massold)/dt
            projection_free=free
            # During brief two-ended outflow there is no Dirichlet reservoir.
            # Remove the global BE moving-volume quadrature defect through the
            # open ends, then fix one potential gauge for the Neumann projection.
            net_adjustment=0.
            if len(fixed)==0:
                endnodes=np.union1d(b.get_dofs('inlet').all(),b.get_dofs('outlet').all())
                delta=np.zeros(b.N);net_adjustment=float(defect.sum())
                delta[endnodes]=-net_adjustment*mass[endnodes]/mass[endnodes].sum()
                op=op+diags(delta);defect=defect+delta
                projection_free=np.arange(1,b.N)
            boundary_projection.append(abs(net_adjustment)*dt/float(mass.sum()))
            offd=asm(diffusion,b,D=1.).tocsr();offd.setdiag(0)
            weights=abs(offd);lap=diags(np.asarray(weights.sum(1)).ravel())-weights
            potential=np.zeros(b.N)
            potential[projection_free]=splu(lap[projection_free][:,projection_free].tocsc()).solve(-defect[projection_free])
            edges=triu(weights,1).tocoo();i,k=edges.row,edges.col
            flux=edges.data*(potential[i]-potential[k])
            correction=coo_matrix((np.r_[flux/2,flux/2,-flux/2,-flux/2],(np.r_[i,i,k,k],np.r_[i,k,i,k])),shape=op.shape).tocsr()
            rawflux=np.asarray(op[i,k]-op[k,i]).ravel()
            flux_correction.append([float(flux@flux),float(rawflux@rawflux)])
            op=op+correction
            constant_defects.append(float(np.max(abs(np.asarray(op.sum(1)).ravel()[free]+(mass-massold)[free]/dt))/np.mean(mass)*dt))
            off=op.copy().tocsr();off.setdiag(0);upwind=off.maximum(off.T).maximum(0);upwind.setdiag(0)
            op=op-upwind+diags(np.asarray(upwind.sum(1)).ravel())
            mat=op+diags(mass/dt)
            left=1. if a.uniform else min(t/.2,1.)
            new=state.copy()
            lu=splu(mat[free][:,free].tocsc());coupling=mat[free][:,fixed]
            for k in [0,1]:
                bc=np.zeros((b.N,B))
                bc[inlet]=left if k==0 else np.array([float(np.interp(t,c['t'],c['boundary'])) for c in custom])
                if a.uniform:bc[:]=1.
                new[:,k,fixed]=bc[fixed].T
                rhs=(massold[None]*state[:,k]/dt).T
                new[:,k,free]=lu.solve(rhs[free]-coupling@bc[fixed]).T
            gbc=np.ones(b.N)
            for case in range(B):
                mg=mat+diags(mass*rate*new[case,1]);rhs=massold*state[case,2]/dt
                new[case,2,fixed]=1.
                new[case,2,free]=splu(mg[free][:,free].tocsc()).solve(rhs[free]-mg[free][:,fixed]@gbc[fixed])
            new[:,3]=massold/mass*state[:,3]+dt*rate*new[:,1]*new[:,2]
            residual=mass*(new[:,2]+new[:,3])-massold*(state[:,2]+state[:,3])+dt*(op@new[:,2].T).T
            consistency.append(float(np.max(abs(residual[:,free]))/np.mean(mass)))
            state=new
        radius=np.interp(xout,z,raw['wall_radius'][:,j]);pts=np.stack([np.broadcast_to(xout[:,None],(len(xout),32)).ravel(),(radius[:,None]*ss[None,:]).ravel()])
        probes=b.probes(pts)
        radial=(probes@state.reshape(B*4,-1).T).T.reshape(B,4,len(xout),32)
        av=radial@avw
        # Physical quadratic moment: evaluate fields, multiply, then integrate.
        direct.append(2*(radial[:,1]*radial[:,2])@avw)
        # The lumped FE update uses nodal products; interpolation of that source
        # differs from the product of interpolants and is saved separately.
        discrete.append((probes@(2*state[:,1]*state[:,2]).T).T.reshape(B,len(xout),32)@avw)
        variance.append(np.stack([(radial[:,1]**2)@avw-av[:,1]**2,(radial[:,2]**2)@avw-av[:,2]**2],1))
        average.append(av);masshistory.append((state[:,2]+state[:,3])@mass);massold=mass;coordold=points.copy()
        if j and j%100==0:print(json.dumps(dict(step=j,time=float(t),seconds=time.perf_counter()-start,minimum=float(state.min()))),flush=True)
    y=np.stack(average,-1);rp=np.stack(direct,-1);rd=np.stack(discrete,-1)
    vv=np.stack(variance,-1);mh=np.stack(masshistory,-1)
    for i,p in enumerate(a.inlets):
        seed=int(p.stem.split('-')[-1]);dest=a.output/f'chemical-{seed}';dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/'fields.npz',y=y[i],x=fl['x'],t=times,area=fl['area'],mass_fg_fn=mh[i],reaction_physical=rp[i],reaction_discrete=rd[i],variance=vv[i],covariance=rp[i]/2-y[i,1]*y[i,2])
    summary=dict(seconds=time.perf_counter()-start,batch=B,nodes=int(b.N),diffusivity=a.diffusion,conservation_interior_residual_max=max(consistency),constant_preservation_residual_max=max(constant_defects),minimum_concentration=float(state.min()),inlets=[str(p) for p in a.inlets])
    (a.output/'batch-metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
