"""Conservative three-vessel geometry and cross-section-resolved transport reference.
The junction is a zero-volume, perfectly mixed node. This is not a 3-D bifurcation CFD solver.
"""
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
import argparse,json,time
ROOT=Path(__file__).resolve().parents[3]
LENGTH=np.array([.06,.02,.034]); COUNTS=np.array([30,10,17]); D=2e-4; RATE=2.
KNOTS=np.array([0,.06,.14,.24,.4,.6,.82])
def flow(n=COUNTS,dt=.002,duration=1.23):
 src=dict(np.load(ROOT/'runs/sen-three-gpu-v2/fields.npz'));s=int(src['selected'])
 A=src['area'][s];Q=A*src['velocity'][s];st=src['t'];period=float(st[-1]);t=np.arange(round(duration/dt)+1)*dt
 data=[];targets=[]
 for v,nv in enumerate(n):
  xf=np.linspace(0,LENGTH[v],nv+1);x=.5*(xf[:-1]+xf[1:]);sx=np.linspace(0,LENGTH[v],10)
  def interp(y,xx):
   y=y.copy();y[:,0]=y[:,-1]=.5*(y[:,0]+y[:,-1])
   waves=CubicSpline(st,y,axis=1,bc_type='periodic')(t%period)
   return np.stack([np.interp(xx,sx,waves[:,j]) for j in range(len(t))])
  af=interp(A[v],xf);qr=interp(Q[v],xf)
  # Integrate the same piecewise-linear Sen area exactly on every mesh.
  ag=interp(A[v],sx);slopes=np.diff(ag,axis=1)/np.diff(sx)[None]
  integ=np.c_[np.zeros(len(t)),(.5*(ag[:,1:]+ag[:,:-1])*np.diff(sx)[None]).cumsum(1)]
  loc=np.minimum(np.searchsorted(sx,xf,side='right')-1,8);delta=xf-sx[loc]
  F=integ[:,loc]+ag[:,loc]*delta[None]+.5*slopes[:,loc]*delta[None]**2
  ac=np.diff(F,axis=1)/(LENGTH[v]/nv)
  vol=ac*(LENGTH[v]/nv);dv=np.r_[((vol[1]-vol[0])/dt)[None],np.diff(vol,axis=0)/dt]
  cumulative=np.c_[np.zeros(len(t)),dv.cumsum(1)]
  data.append(dict(x=x,xf=xf,area=ac,af=af,volume=vol,dv=dv,cumulative=cumulative,qraw=qr))
  targets.append(interp(Q[v],sx).mean(1)+.5*dv.sum(1))
 parent_in=data[0]['qraw'][:,0];parent_end=parent_in-data[0]['cumulative'][:,-1]
 # Equal absolute least-squares fit of daughter fluxes over all faces, with exact junction sum.
 w1,w2=1,1
 inlet1=(w1*targets[1]+w2*(parent_end-targets[2]))/(w1+w2);inlet2=parent_end-inlet1
 for d,qi in zip(data,[parent_in,inlet1,inlet2]):d['q']=qi[:,None]-d['cumulative']
 return data,t
def geometry(n,nr):
 ns=int(sum(n)*nr);junction=ns
 offset=np.r_[0,np.cumsum(n*nr)]
 cell_v=np.concatenate([np.full(nv*nr,v) for v,nv in enumerate(n)])
 x=np.concatenate([np.repeat((np.arange(nv)+.5)*LENGTH[v]/nv,nr) for v,nv in enumerate(n)])
 se=np.linspace(0,1,nr+1);w=np.diff(se**2);profile=2-se[1:]**2-se[:-1]**2
 # Directed physical connections: internal axial, radial, parent->junction, junction->daughters.
 edges=[];tags=[]
 for v,nv in enumerate(n):
  for i in range(nv-1):
   for k in range(nr):
    edges.append((offset[v]+i*nr+k,offset[v]+(i+1)*nr+k));tags.append(('axial',v,i+1,k))
  for i in range(nv):
   for k in range(nr-1):
    edges.append((offset[v]+i*nr+k,offset[v]+i*nr+k+1));tags.append(('radial',v,i,k))
  for k in range(nr):
   edges.append((offset[v]+(nv-1)*nr+k,junction) if v==0 else (junction,offset[v]+k))
   tags.append(('junction',v,nv if v==0 else 0,k))
 bnodes=np.r_[offset[0]+np.arange(nr),offset[1]+(n[1]-1)*nr+np.arange(nr),offset[2]+(n[2]-1)*nr+np.arange(nr)].astype(int)
 return dict(ns=ns,junction=junction,offset=offset,w=w,profile=profile,se=se,x=x,cell_v=cell_v,edges=np.asarray(edges,int),tags=tags,bnodes=bnodes)
def operators(data,j,g,n,nr,diffusion=D):
 w=g['w'];pr=g['profile'];se=g['se'];qs=[];gs=[]
 radial=[]
 for v,nv in enumerate(n):
  d=data[v];res=d['dv'][j,:,None]*w+(d['q'][j,1:]-d['q'][j,:-1])[:,None]*w*pr
  radial.append(-res.cumsum(1)[:,:-1])
 for kind,v,i,k in g['tags']:
  d=data[v];dx=LENGTH[v]/n[v]
  if kind=='radial':
   qs.append(radial[v][i,k]);gs.append(diffusion*2*np.pi*se[k+1]*dx*nr)
  else:
   qs.append(d['q'][j,i]*w[k]*pr[k])
   gs.append(diffusion*d['af'][j,i]*w[k]/(dx*(.5 if kind=='junction' else 1.)))
 q=np.asarray(qs);cond=np.asarray(gs);a,b=g['edges'].T
 plus=q.clip(0)+cond;minus=q.clip(max=0)-cond
 rows=np.r_[a,a,b,b];cols=np.r_[a,b,a,b];vals=np.r_[plus,minus,-plus,-minus]
 size=g['ns']+1; L=coo_matrix((vals,(rows,cols)),shape=(size,size)).tocsr()
 bc=np.zeros((size,3));diag=np.zeros(size)
 for v in range(3):
  d=data[v];ix=0 if v==0 else -1;flowout=(-1 if v==0 else 1)*d['q'][j,ix]*w*pr
  gd=diffusion*d['af'][j,ix]*w/(.5*LENGTH[v]/n[v])
  if v:gd=gd*(flowout<0)
  ii=g['bnodes'][v*nr:(v+1)*nr]
  diag[ii]+=flowout.clip(0)+gd;bc[ii,v]=(-flowout).clip(0)+gd
 L=L+diags(diag)
 volume=np.r_[np.concatenate([(d['volume'][j,:,None]*w).ravel() for d in data]),0.]
 return L,bc,volume,q,cond
def inlets():
 rng=np.random.default_rng(2026102801)
 amps=rng.uniform(.05,1.15,size=(48,7))
 return amps,dict(train=list(range(24)),validation=list(range(24,32)),test=list(range(32,48)),seed=2026102801)
def run(output,refine=2,nr=12,dt=.002,duration=1.23,indices=None,uniform=False):
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 (output/'executed-source.py').write_text(Path(__file__).read_text())
 n=COUNTS*refine;data,t=flow(n,dt,duration);g=geometry(n,nr);amps,split=inlets()
 ids=np.arange(len(amps)) if indices is None else np.asarray(indices);amps=amps[ids];B=len(amps)
 cin=np.stack([np.interp(t,KNOTS,a) for a in amps])
 h0=.2*np.exp(-((g['x']/.06-.25)/.1)**2)*(g['cell_v']==0)
 H=np.broadcast_to(np.r_[h0,0.][:,None],(g['ns']+1,B)).copy();G=np.ones_like(H);Fn=np.zeros_like(H)
 if uniform:H[:]=1.;Fn[:]=0
 avs=[];rates=[];fluxes=[];mass=[];defects=[];minimum=1.;start=time.perf_counter();volold=None
 for j,tt in enumerate(t):
  L,bc,vol,q,cond=operators(data,j,g,n,nr)
  if j:
   mat=L+diags(vol/dt);lu=splu(mat.tocsc())
   bH=np.ones((3,B)) if uniform else np.stack([cin[:,j],np.zeros(B),np.zeros(B)])
   rhsH=volold[:,None]/dt*H+bc@bH
   rhsG=volold[:,None]/dt*G+bc@np.ones((3,B))
   ans=lu.solve(np.c_[rhsH,rhsG]);H=ans[:,:B];Gtransport=ans[:,B:]
   rate=0. if uniform else RATE
   G=Gtransport*np.exp(-rate*H*dt)
   Fn[:g['ns']]=volold[:g['ns'],None]/vol[:g['ns'],None]*Fn[:g['ns']]+Gtransport[:g['ns']]-G[:g['ns']]
   Fn[-1]=0
   before=mass[-1];after=((G+Fn)*vol[:,None]).sum(0)
   balance=after-before+dt*((L@Gtransport-bc@np.ones((3,B))).sum(0))
   defects.append(float(abs(balance).max()/vol.sum()))
  volold=vol
  av=[];rr=[];jf=[]
  for v,nv in enumerate(n):
   sl=slice(g['offset'][v],g['offset'][v+1]);h=H[sl].reshape(nv,nr,B);gg=G[sl].reshape(nv,nr,B);f=Fn[sl].reshape(nv,nr,B)
   aa=np.stack([h,gg,f]).transpose(3,0,1,2)
   vv=data[v]['volume'][j].reshape(COUNTS[v],refine)
   vals=(aa@g['w']).reshape(B,3,COUNTS[v],refine)
   av.append((vals*vv[None,None]).sum(-1)/vv.sum(-1)[None,None])
   rate_phys=(RATE*h*gg).transpose(2,0,1)@g['w']
   rr.append((rate_phys.reshape(B,COUNTS[v],refine)*vv[None]).sum(-1)/vv.sum(-1)[None])
   faces=np.arange(1,COUNTS[v])*refine
   # Axial advective covariance at coarse physical faces; use consistent upwind reference state.
   up=np.where(data[v]['q'][j,faces]>=0,faces-1,faces)
   hc=h[up]
   cov=(hc*g['w'][None,:,None]*(g['profile'][None,:,None]-1)).sum(1).T
   jf.append(cov*data[v]['q'][j,faces][None])
  avs.append(np.concatenate(av,axis=2));rates.append(np.concatenate(rr,axis=1));fluxes.append(np.concatenate(jf,axis=1));mass.append(((G+Fn)*vol[:,None]).sum(0));minimum=min(minimum,float(min(H.min(),G.min(),Fn.min())))
  if j and j%100==0:print(json.dumps(dict(step=j,seconds=time.perf_counter()-start,minimum=minimum)),flush=True)
 y=np.stack(avs,-1);rp=np.stack(rates,-1);fl=np.stack(fluxes,1)
 np.savez_compressed(output/'data.npz',y=y,reaction=rp,flux=fl,t=t,amplitudes=amps,case_ids=ids)
 qc=np.concatenate([d['qraw'] for d in data],1);qp=np.concatenate([d['q'] for d in data],1)
 info=dict(seconds=time.perf_counter()-start,axial_counts=n.tolist(),radial_cells=nr,dt=dt,diffusion=D,reaction_rate=RATE,split=split,case_ids=ids.tolist(),reference='Axisymmetric prescribed-flow reaction-transport on three tubes with a perfectly mixed zero-volume junction; not 3D bifurcation Navier-Stokes CFD',flow='Sen GPU prediction, projected onto discrete volume and junction conservation; shared by every method',flow_projection_relative_l2=float(np.linalg.norm(qp-qc)/np.linalg.norm(qc)),mass_balance_max=max(defects),minimum=minimum,uniform_max_error=float(abs(y[:,:2]-1).max()) if uniform else None)
 (output/'metrics.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--refine',type=int,default=2);ap.add_argument('--radial',type=int,default=12);ap.add_argument('--dt',type=float,default=.002);ap.add_argument('--indices',type=int,nargs='+');ap.add_argument('--uniform',action='store_true')
 a=ap.parse_args();run(a.output,a.refine,a.radial,a.dt,indices=a.indices,uniform=a.uniform)
