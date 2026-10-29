"""Conservative transfer of independent CFD to resolved/reference and 1-D grids."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import coo_matrix,diags
from gp52.three.reference import ROOT,COUNTS,LENGTH,D,geometry
RUN=ROOT/'runs/independent-cfd-v1'
def interval_average(z,k):
    if k==1:return z.copy()
    return np.concatenate([z[:1],z[1:].reshape((len(z)-1)//k,k,*z.shape[1:]).mean(1)])
def interpolate_axis(z,x,xnew):
    return np.moveaxis(np.stack([np.interp(xnew,x,r) for r in np.moveaxis(z,1,-1).reshape(-1,len(x))]).reshape(len(z),*z.shape[2:],len(xnew)),-1,1)
@lru_cache(maxsize=6)
def load_flow(path,n_tuple=(30,10,17),nr=1,dt=.002):
    raw=np.load(Path(path)/'flow.npz');st=raw['t'];sdt=float(st[1]-st[0]);k=round(dt/sdt)
    assert k>=1 and abs(k*sdt-dt)<1e-10
    t=st[::k];n=np.array(n_tuple);native=[];targets=[];derivatives=[]
    for v in range(3):
        xf=raw[f'v{v}_xf'];vol=raw[f'v{v}_volume'];af=raw[f'v{v}_face_area']
        qraw=raw[f'v{v}_face_flow'];prof=raw[f'v{v}_radial_axial_velocity']
        dv=np.r_[((vol[1]-vol[0])/sdt)[None],np.diff(vol,axis=0)/sdt]
        cum=np.c_[np.zeros(len(st)),dv.cumsum(1)]
        targets.append((qraw+cum).mean(1));derivatives.append(dv.sum(1))
        native.append(dict(xf=xf,vol=vol,af=af,qraw=qraw,prof=prof,dv=dv,cum=cum))
    # Least-squares projection with equal weight per tube; junction is one constraint.
    targets=np.stack(targets,1);sign=np.array([1,-1,-1])
    delta=(targets@sign-derivatives[0])/3
    qin=targets-delta[:,None]*sign[None]
    data=[];projection=[]
    for v,d in enumerate(native):
        qproj=qin[:,v,None]-d['cum']
        projection.append([float(np.sum((qproj-d['qraw'])**2)),float(np.sum(d['qraw']**2))])
        se=np.linspace(0,1,d['prof'].shape[-1]+1);w=np.diff(se**2)
        qb=d['af'][:,:,None]*d['prof']*w[None,None]
        qb+=(qproj-qb.sum(2))[:,:,None]*w[None,None]
        # Preserve radial integrated flow while changing radial bins.
        sn=np.linspace(0,1,nr+1)**2
        qc=np.concatenate([np.zeros((*qb.shape[:2],1)),qb.cumsum(2)],2)
        newc=np.stack([np.interp(sn,se**2,z) for z in qc.reshape(-1,len(se))]).reshape(*qb.shape[:2],nr+1)
        qb=np.diff(newc,axis=2)
        xf=np.linspace(0,LENGTH[v],n[v]+1);x=.5*(xf[:-1]+xf[1:])
        primitive=np.c_[np.zeros(len(st)),d['vol'].cumsum(1)]
        vol=np.diff(interpolate_axis(primitive,d['xf'],xf),axis=1)[::k]
        af=interpolate_axis(d['af'],d['xf'],xf)[::k]
        q=interval_average(interpolate_axis(qproj,d['xf'],xf),k)
        qbins=interval_average(interpolate_axis(qb,d['xf'],xf),k)
        qr=interval_average(interpolate_axis(d['qraw'],d['xf'],xf),k)
        dv=np.r_[((vol[1]-vol[0])/dt)[None],np.diff(vol,axis=0)/dt]
        data.append(dict(x=x,xf=xf,area=vol/(LENGTH[v]/n[v]),af=af,volume=vol,dv=dv,
             q=q,qraw=qr,qbins=qbins,cumulative=np.c_[np.zeros(len(t)),dv.cumsum(1)]))
    stats=dict(projection_relative_l2_by_vessel=[np.sqrt(a/b) for a,b in projection],
      continuity_max_m3_s=max(float(abs(d['dv'][1:]+np.diff(d['q'][1:],axis=1)).max()) for d in data),
      junction_max_m3_s=float(abs(data[0]['q'][1:,-1]-data[1]['q'][1:,0]-data[2]['q'][1:,0]).max()))
    return data,t,stats
def operators(data,j,g,n,nr,diffusion=D):
    w=g['w'];se=g['se']
    radial=[]
    for d in data:
        res=d['dv'][j,:,None]*w+np.diff(d['qbins'][j],axis=0)
        radial.append(-res.cumsum(1)[:,:-1])
    tags=g['tags'];qs=[];gs=[]
    for kind,v,i,k in tags:
        d=data[v];dx=LENGTH[v]/n[v]
        if kind=='radial':
            qs.append(radial[v][i,k]);gs.append(diffusion*2*np.pi*se[k+1]*dx*nr)
        else:
            qs.append(d['qbins'][j,i,k]);gs.append(diffusion*d['af'][j,i]*w[k]/(dx*(.5 if kind=='junction' else 1.)))
    q=np.array(qs);cond=np.array(gs);a,b=g['edges'].T
    plus=q.clip(0)+cond;minus=q.clip(max=0)-cond;size=g['ns']+1
    L=coo_matrix((np.r_[plus,minus,-plus,-minus],(np.r_[a,a,b,b],np.r_[a,b,a,b])),shape=(size,size)).tocsr()
    bc=np.zeros((size,3));diag=np.zeros(size)
    for v,d in enumerate(data):
        ix=0 if v==0 else -1;out=(-1 if v==0 else 1)*d['qbins'][j,ix]
        gd=diffusion*d['af'][j,ix]*w/(.5*LENGTH[v]/n[v])
        if v:gd=gd*(out<0)
        ii=g['bnodes'][v*nr:(v+1)*nr]
        diag[ii]+=out.clip(0)+gd;bc[ii,v]=(-out).clip(0)+gd
    L=L+diags(diag)
    vol=np.r_[np.concatenate([(d['volume'][j,:,None]*w).ravel() for d in data]),0.]
    return L,bc,vol,q,cond
