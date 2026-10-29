"""Declarative vessel topology shared by 1-, 3-, and 7-vessel experiments."""
import numpy as np
NETWORKS={
 'single':dict(inlet_nodes=[0],outlet_nodes=[1],lengths=[.06],counts=[41],source_files=[1],
  sensors=[[0,.03],[0,.006],[0,.054]],status='active'),
 'three':dict(inlet_nodes=[0,1,1],outlet_nodes=[1,2,3],lengths=[.06,.02,.034],counts=[30,10,17],source_files=[1,2,3],
  sensors=[[0,.03],[1,.01],[2,.017]],status='active'),
 'seven':dict(inlet_nodes=[0,1,1,2,2,3,3],outlet_nodes=[1,2,3,4,5,6,7],
  lengths=[.06,.02,.034,.039,.139,.034,.094],counts=[30,10,17,20,70,17,47],source_files=[1,2,3,6,7,4,5],
  sensors=[[0,.03],[3,.0195],[6,.047]],status='supported')
}
def junctions(name):
 d=NETWORKS[name];return sorted(set(d['inlet_nodes'])&set(d['outlet_nodes']))
def constraints(name):
 d=NETWORKS[name];ii=np.array(d['inlet_nodes']);oo=np.array(d['outlet_nodes']);js=junctions(name)
 return np.array([(oo==j).astype(float)-(ii==j).astype(float) for j in js]).reshape(len(js),len(ii))
def endpoints(name,n):
 d=NETWORKS[name];js=junctions(name);ends=[]
 for v,(a,b) in enumerate(zip(d['inlet_nodes'],d['outlet_nodes'])):
  if a not in js:ends.append((v,0,-1))
  if b not in js:ends.append((v,int(n[v]),1))
 return sorted(ends,key=lambda z:(z[2],d['inlet_nodes'][z[0]] if z[2]<0 else d['outlet_nodes'][z[0]]))
def validate():
 for name,d in NETWORKS.items():
  ii=d['inlet_nodes'];oo=d['outlet_nodes'];assert len(ii)==len(oo)==len(d['lengths'])==len(d['counts'])
  assert all(a!=b for a,b in zip(ii,oo))
  C=constraints(name);assert np.linalg.matrix_rank(C)==len(C) if len(C) else True
  ends=endpoints(name,np.array(d['counts']));assert sum(e[2]<0 for e in ends)==1
  assert len(ends)==len(ii)-len(junctions(name))+1
 return {k:dict(vessels=len(v['counts']),junctions=len(junctions(k)),external_ends=len(endpoints(k,v['counts']))) for k,v in NETWORKS.items()}
