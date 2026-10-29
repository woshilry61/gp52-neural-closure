"""Shared resolved biochemical generator on independent CFD radial velocities."""
import argparse,json,time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import coo_matrix,diags
from gp52.three_independent.chemistry import CUDASolve
from gp52.three_independent.transfer import interpolate_axis,interval_average
from .protocol import ROOT,RUN,COUNTS,LENGTH,KNOTS,inputs
from .topology import NETWORKS,junctions,constraints,endpoints
D=2e-4;RATE=2.
def canonical_blood():
 for name in ['single','three','seven']:
  out=RUN/name;out.mkdir(exist_ok=True)
  if (out/'blood.npz').exists():continue
  if name=='seven':
   source=RUN/'seven/blood/flow.npz'
   if not source.exists():continue
   z=np.load(source);fields={k:z[k] for k in z.files}
  elif name=='three':
   source=RUN/'three/blood/flow.npz'
   z=np.load(source);fields={k:z[k] for k in z.files}
  else:
   source=RUN/'single/blood/flow.npz'
   z=np.load(source);fields={'t':z['t'],'v0_xf':z['xf'],
    'v0_volume':z['area'].T*float(z['dx']),'v0_face_area':z['face_area'].T,
    'v0_face_flow':z['face_flow'].T,'v0_radial_axial_velocity':np.moveaxis(z['radial_axial_velocity'],-1,0)}
  np.savez_compressed(out/'blood.npz',**fields)
  import hashlib
  (out/'blood-provenance.json').write_text(json.dumps(dict(source=str(source),
   sha256=hashlib.sha256(source.read_bytes()).hexdigest(),model='Independent axisymmetric compliant-wall ALE Navier-Stokes; no Sen prediction'),indent=2))
@lru_cache(maxsize=8)
def flow(name,n_tuple,nr=1,dt=.002):
 raw=np.load(RUN/name/'blood.npz');st=raw['t'];sdt=float(st[1]-st[0]);k=round(dt/sdt)
 assert k>=1 and abs(k*sdt-dt)<1e-10
 n=np.array(n_tuple);t=st[::k];native=[];targets=[];derivatives=[]
 for v in range(len(n)):
  xf=raw[f'v{v}_xf'];vol=raw[f'v{v}_volume'];af=raw[f'v{v}_face_area']
  qr=raw[f'v{v}_face_flow'];prof=raw[f'v{v}_radial_axial_velocity']
  dv=np.r_[((vol[1]-vol[0])/sdt)[None],np.diff(vol,axis=0)/sdt]
  cum=np.c_[np.zeros(len(st)),dv.cumsum(1)]
  targets.append((qr+cum).mean(1));derivatives.append(dv.sum(1))
  native.append(dict(xf=xf,vol=vol,af=af,qr=qr,prof=prof,dv=dv,cum=cum))
 qin=np.stack(targets,1)
 C=constraints(name)
 if len(C):
  rhs=np.stack([sum(derivatives[v] for v,node in enumerate(NETWORKS[name]['outlet_nodes']) if node==j) for j in junctions(name)],1)
  qin=qin-(np.linalg.solve(C@C.T,(qin@C.T-rhs).T).T@C)
 data=[];proj=[]
 for v,d in enumerate(native):
  qp=qin[:,v,None]-d['cum'];proj.append(float(np.linalg.norm(qp-d['qr'])/np.linalg.norm(d['qr'])))
  se=np.linspace(0,1,d['prof'].shape[-1]+1);w=np.diff(se**2)
  qb=d['af'][:,:,None]*d['prof']*w[None,None]
  qb+=(qp-qb.sum(2))[:,:,None]*w[None,None]
  qc=np.concatenate([np.zeros((*qb.shape[:2],1)),qb.cumsum(2)],2)
  newc=np.stack([np.interp(np.linspace(0,1,nr+1)**2,se**2,z) for z in qc.reshape(-1,len(se))]).reshape(*qb.shape[:2],nr+1)
  xf=np.linspace(0,LENGTH[name][v],n[v]+1);primitive=np.c_[np.zeros(len(st)),d['vol'].cumsum(1)]
  vol=np.diff(interpolate_axis(primitive,d['xf'],xf),axis=1)[::k]
  dv=np.r_[((vol[1]-vol[0])/dt)[None],np.diff(vol,axis=0)/dt]
  data.append(dict(xf=xf,x=.5*(xf[1:]+xf[:-1]),volume=vol,area=vol/(LENGTH[name][v]/n[v]),
   af=interpolate_axis(d['af'],d['xf'],xf)[::k],dv=dv,
   q=interval_average(interpolate_axis(qp,d['xf'],xf),k),
   qbins=interval_average(interpolate_axis(np.diff(newc,axis=2),d['xf'],xf),k)))
 stats=dict(projection_relative_l2=proj,continuity_max=max(float(abs(d['dv'][1:]+np.diff(d['q'][1:],axis=1)).max()) for d in data))
 if len(C):
  stats['junction_max']=max(float(abs(sum(data[v]['q'][:,-1] for v,node in enumerate(NETWORKS[name]['outlet_nodes']) if node==j)-sum(data[v]['q'][:,0] for v,node in enumerate(NETWORKS[name]['inlet_nodes']) if node==j)).max()) for j in junctions(name))
 return data,t,stats
def geometry(name,n,nr):
 lengths=LENGTH[name];ns=int(sum(n)*nr);js=junctions(name);jm={j:ns+k for k,j in enumerate(js)}
 off=np.r_[0,np.cumsum(n*nr)];se=np.linspace(0,1,nr+1);edges=[];tags=[]
 for v,nv in enumerate(n):
  for i in range(nv-1):
   for k in range(nr):edges.append((off[v]+i*nr+k,off[v]+(i+1)*nr+k));tags.append(('axial',v,i+1,k))
  for i in range(nv):
   for k in range(nr-1):edges.append((off[v]+i*nr+k,off[v]+i*nr+k+1));tags.append(('radial',v,i,k))
  node_in=NETWORKS[name]['inlet_nodes'][v];node_out=NETWORKS[name]['outlet_nodes'][v]
  if node_in in jm:
   for k in range(nr):edges.append((jm[node_in],off[v]+k));tags.append(('junction',v,0,k))
  if node_out in jm:
   for k in range(nr):edges.append((off[v]+(nv-1)*nr+k,jm[node_out]));tags.append(('junction',v,nv,k))
 ends=endpoints(name,n)
 bnodes=[off[v]+(0 if face==0 else n[v]-1)*nr+np.arange(nr) for v,face,sg in ends]
 x=np.concatenate([np.repeat((np.arange(nv)+.5)*lengths[v]/nv,nr) for v,nv in enumerate(n)])
 return dict(ns=ns,size=ns+len(js),offset=off,w=np.diff(se**2),se=se,
  edges=np.array(edges),tags=tags,ends=ends,bnodes=bnodes,x=x,
  cell_v=np.concatenate([np.full(nv*nr,v) for v,nv in enumerate(n)]))
def operators_serial(name,data,j,g,n,nr):
 radial=[-(d['dv'][j,:,None]*g['w']+np.diff(d['qbins'][j],axis=0)).cumsum(1)[:,:-1] for d in data]
 qs=[];gs=[];w=g['w'];se=g['se'];lengths=LENGTH[name]
 for kind,v,i,k in g['tags']:
  d=data[v];dx=lengths[v]/n[v]
  qs.append(radial[v][i,k] if kind=='radial' else d['qbins'][j,i,k])
  gs.append(D*2*np.pi*se[k+1]*dx*nr if kind=='radial' else D*d['af'][j,i]*w[k]/(dx*(.5 if kind=='junction' else 1.)))
 q=np.array(qs);cond=np.array(gs);a,b=g['edges'].T
 plus=q.clip(0)+cond;minus=q.clip(max=0)-cond
 L=coo_matrix((np.r_[plus,minus,-plus,-minus],(np.r_[a,a,b,b],np.r_[a,b,a,b])),shape=(g['size'],g['size'])).tocsr()
 bc=np.zeros((g['size'],len(g['ends'])));diag=np.zeros(g['size'])
 for end,(v,face,sign) in enumerate(g['ends']):
  d=data[v];out=sign*d['qbins'][j,face];gd=D*d['af'][j,face]*w/(.5*lengths[v]/n[v])
  if end:gd=gd*(out<0)
  ii=g['bnodes'][end];diag[ii]+=out.clip(0)+gd;bc[ii,end]=(-out).clip(0)+gd
 vol=np.concatenate([(d['volume'][j,:,None]*w).ravel() for d in data])
 if g['size']>g['ns']:vol=np.r_[vol,np.zeros(g['size']-g['ns'])]
 return L+diags(diag),bc,vol
def operators(name,data,j,g,n,nr):
 # Index geometry once, then assemble each variable-flow step in vectorized arrays.
 if '_assembly_groups' not in g:
  tags=g['tags'];groups=[]
  for v,nv in enumerate(n):
   for kind in ['axial','radial','junction']:
    idx=np.array([e for e,z in enumerate(tags) if z[0]==kind and z[1]==v],dtype=int)
    if len(idx):
     ii=np.array([tags[e][2] for e in idx]);kk=np.array([tags[e][3] for e in idx])
     groups.append((v,kind,idx,ii,kk))
  g['_assembly_groups']=groups
 radial=[-(d['dv'][j,:,None]*g['w']+np.diff(d['qbins'][j],axis=0)).cumsum(1)[:,:-1] for d in data]
 q=np.empty(len(g['tags']));cond=np.empty_like(q);w=g['w'];se=g['se'];lengths=LENGTH[name]
 for v,kind,idx,ii,kk in g['_assembly_groups']:
  d=data[v];dx=lengths[v]/n[v]
  q[idx]=radial[v][ii,kk] if kind=='radial' else d['qbins'][j,ii,kk]
  cond[idx]=D*2*np.pi*se[kk+1]*dx*nr if kind=='radial' else D*d['af'][j,ii]*w[kk]/(dx*(.5 if kind=='junction' else 1.))
 a,b=g['edges'].T;plus=q.clip(0)+cond;minus=q.clip(max=0)-cond
 L=coo_matrix((np.r_[plus,minus,-plus,-minus],(np.r_[a,a,b,b],np.r_[a,b,a,b])),shape=(g['size'],g['size'])).tocsr()
 bc=np.zeros((g['size'],len(g['ends'])));diag=np.zeros(g['size'])
 for end,(v,face,sign) in enumerate(g['ends']):
  d=data[v];out=sign*d['qbins'][j,face];gd=D*d['af'][j,face]*w/(.5*lengths[v]/n[v])
  if end:gd=gd*(out<0)
  ii=g['bnodes'][end];diag[ii]+=out.clip(0)+gd;bc[ii,end]=(-out).clip(0)+gd
 vol=np.concatenate([(d['volume'][j,:,None]*w).ravel() for d in data])
 if g['size']>g['ns']:vol=np.r_[vol,np.zeros(g['size']-g['ns'])]
 return L+diags(diag),bc,vol

def run(name,refine=4,nr=24,dt=.001,ids=None,label='reference',uniform=False):
 out=RUN/name/label;out.mkdir(exist_ok=False);n=COUNTS[name]*refine
 data,t,stats=flow(name,tuple(n),nr,dt);g=geometry(name,n,nr)
 ids=np.arange(112) if ids is None else np.array(ids);amps=inputs()[ids];B=len(ids)
 cin=np.stack([np.interp(t,KNOTS,a) for a in amps]);h0=.2*np.exp(-((g['x']/.06-.25)/.1)**2)*(g['cell_v']==0)
 if g['size']>g['ns']:h0=np.r_[h0,np.zeros(g['size']-g['ns'])]
 H=torch.tensor(np.broadcast_to(h0[:,None],(g['size'],B)).copy(),device='cuda')
 G=torch.ones_like(H);F=torch.zeros_like(H)
 if uniform:H.fill_(1)
 solver=CUDASolve();ys=[];rates=[];fluxes=[];vols=[];defects=[];consts=[];start=time.perf_counter()
 for j,tt in enumerate(t):
  L,bc,vn=operators(name,data,j,g,n,nr);vol=torch.as_tensor(vn,device='cuda');bct=torch.as_tensor(bc,device='cuda')
  if j:
   bh=np.ones((len(g['ends']),B)) if uniform else np.r_[cin[None,:,j],np.zeros((len(g['ends'])-1,B))]
   rhsH=vo[:,None]/dt*H+bct@torch.as_tensor(bh,device='cuda')
   rhsG=vo[:,None]/dt*G+bct@torch.ones((len(g['ends']),B),device='cuda',dtype=torch.float64)
   ans=solver.solve(L+diags(vn/dt),torch.cat([rhsH,rhsG],1),torch.cat([H,G],1))
   H=ans[:,:B];gt=ans[:,B:];G=gt*torch.exp(-(0 if uniform else RATE)*H*dt)
   F[:g['ns']]=vo[:g['ns'],None]/vol[:g['ns'],None]*F[:g['ns']]+gt[:g['ns']]-G[:g['ns']]
   after=((G+F)*vol[:,None]).sum(0)
   bal=after.cpu().numpy()-mass+dt*(L@gt.cpu().numpy()-bc@np.ones((len(g['ends']),B))).sum(0)
   defects.append(float(abs(bal).max()/vn.sum()))
   constant=np.asarray(L.sum(1)).ravel()-bc.sum(1)+(vn-vo.cpu().numpy())/dt
   consts.append(float(abs(constant).max()*dt/(vn.sum()/g['ns'])))
  av=[];rr=[];jf=[]
  for v,nv in enumerate(n):
   sl=slice(g['offset'][v],g['offset'][v+1]);w=torch.as_tensor(g['w'],device='cuda')
   h=H[sl].reshape(nv,nr,B);gg=G[sl].reshape(nv,nr,B)
   vv=torch.as_tensor(data[v]['volume'][j].reshape(COUNTS[name][v],refine),device='cuda')
   mean=(torch.stack([H[sl],G[sl],F[sl]]).reshape(3,nv,nr,B)*w[None,None,:,None]).sum(2).reshape(3,COUNTS[name][v],refine,B)
   av.append(((mean*vv[None,:,:,None]).sum(2)/vv.sum(1)[None,:,None]).permute(2,0,1))
   r=(RATE*h*gg*w[None,:,None]).sum(1).reshape(COUNTS[name][v],refine,B)
   rr.append(((r*vv[:,:,None]).sum(1)/vv.sum(1)[:,None]).T)
   faces=np.arange(1,COUNTS[name][v])*refine
   qb=torch.as_tensor(data[v]['qbins'][j,faces],device='cuda')
   hup=torch.where(qb[:,:,None]>=0,h[faces-1],h[faces])
   jf.append(((qb-qb.sum(1)[:,None]*w[None])[:,:,None]*hup).sum(1).T)
  ys.append(torch.cat(av,2).cpu().numpy());rates.append(torch.cat(rr,1).cpu().numpy());fluxes.append(torch.cat(jf,1).cpu().numpy())
  vols.append(np.concatenate([d['volume'][j].reshape(COUNTS[name][v],refine).sum(1) for v,d in enumerate(data)]))
  mass=((G+F)*vol[:,None]).sum(0).cpu().numpy();vo=vol
  if j%100==0:print(json.dumps(dict(network=name,phase=label,step=j,seconds=time.perf_counter()-start)),flush=True)
 y=np.stack(ys,-1)
 np.savez_compressed(out/'data.npz',y=y,reaction=np.stack(rates,-1),flux=np.stack(fluxes,1),volume=np.stack(vols),t=t,amplitudes=amps,case_ids=ids)
 report=dict(seconds=time.perf_counter()-start,cases=ids.tolist(),refine=refine,nr=nr,dt=dt,transfer=stats,mass_balance_max=max(defects),
  constant_operator_max=max(consts),minimum=float(y.min()),uniform_error=float(abs(y[:,:2]-1).max()) if uniform else None,
  linear_residual_max=max(r['residual'] for r in solver.history),cpu_parity=solver.history[0]['cpu_relative_difference'])
 (out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('network',choices=list(NETWORKS));p.add_argument('--refine',type=int,default=4);p.add_argument('--radial',type=int,default=24);p.add_argument('--dt',type=float,default=.001);p.add_argument('--ids',type=int,nargs='+');p.add_argument('--label',default='reference');p.add_argument('--uniform',action='store_true');a=p.parse_args()
 canonical_blood();run(a.network,a.refine,a.radial,a.dt,a.ids,a.label,a.uniform)
