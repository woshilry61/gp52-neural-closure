"""Reference convergence and topology conservation checks, without training/test tuning."""
import json
import numpy as np
from .protocol import RUN,ROOT,COUNTS,LENGTH
from .topology import validate
from .reference import geometry,operators,operators_serial,flow
from gp52.three_independent.transfer import load_flow
def main():
 records={'topologies':validate()}
 a,t,_=flow('three',tuple(COUNTS['three']),1,.002)
 b,_,_=load_flow(str(RUN/'three/blood'),tuple(COUNTS['three']),1,.002)
 records['three_transfer_parity']=max(float(abs(aa[k]-bb[k]).max()) for aa,bb in zip(a,b) for k in ['volume','af','q','qbins'])
 assert records['three_transfer_parity']<1e-15
 n=COUNTS['seven'];nr=2;g=geometry('seven',n,nr);q=np.array([4,2,2,1,1,1,1])*1e-6;ds=[]
 for v,nv in enumerate(n):
  area=1e-4;vol=np.full((2,nv),area*LENGTH['seven'][v]/nv)
  ds.append(dict(volume=vol,dv=np.zeros_like(vol),af=np.full((2,nv+1),area),qbins=np.broadcast_to(q[v]*g['w'],(2,nv+1,nr))))
 L,bc,vol=operators('seven',ds,1,g,n,nr);r=L@np.ones(g['size'])-bc@np.ones(5)
 records['seven_manufactured_uniform_residual']=float(abs(r).max());assert abs(r).max()<1e-16
 records['seven_scope']='Independent seven-tube CFD and three-junction coupling, followed by common biochemical generation and training.'
 records['seven_blood']=json.loads((RUN/'seven/blood/metrics.json').read_text())
 records['assembly_parity']={}
 for name in ['single','three','seven']:
  nn=COUNTS[name]*4;data,tt,_=flow(name,tuple(nn),24,.001);gg=geometry(name,nn,24)
  original=operators_serial(name,data,123,gg,nn,24);fast=operators(name,data,123,gg,nn,24)
  assert np.array_equal(original[0].data,fast[0].data) and np.array_equal(original[0].indices,fast[0].indices)
  assert np.array_equal(original[1],fast[1]) and np.array_equal(original[2],fast[2])
  records['assembly_parity'][name]='bitwise equal'
 records['refinement']={}
 for name in ['single','three','seven']:
  a=np.load(RUN/name/'reference/data.npz');b=np.load(RUN/name/'refined/data.npz');ids=b['case_ids'];w=a['volume'].T.copy()
  w[:,[0,-1]]*=.5
  e=np.sqrt(np.sum((a['y'][ids]-b['y'])**2*w[None,None],(-2,-1))/np.sum(b['y']**2*w[None,None],(-2,-1)))
  records['refinement'][name]=dict(case_ids=ids.tolist(),species=['Th','Fg','Fn'],errors=e.tolist(),max_by_species=e.max(0).tolist())
  u=json.loads((RUN/name/'uniform/metrics.json').read_text())
  assert u['uniform_error']<1e-9 and u['mass_balance_max']<1e-8
  records[name+'_uniform']=u
 (RUN/'reference-validation.json').write_text(json.dumps(records,indent=2)+'\n')
 print(json.dumps({k:v['max_by_species'] for k,v in records['refinement'].items()}),flush=True)
if __name__=='__main__':main()
