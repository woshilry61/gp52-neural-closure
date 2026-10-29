"""Numerical parity checks before test-set evaluation."""
import argparse,json
import numpy as np
import torch
from scipy.sparse import diags
from scipy.sparse.linalg import splu,spsolve
from .protocol import RUN,COUNTS,inputs
from .reference import flow,geometry,operators
from .models import model
from .evaluate import configure
from gp52.three import inverse as inv
def cpu_c0(name,amps):
 blood,t,_=flow(name,tuple(COUNTS[name]),1,.002);n=COUNTS[name];g=geometry(name,n,1)
 h=.2*np.exp(-((g['x']/.06-.25)/.1)**2)*(g['cell_v']==0)
 h=np.broadcast_to(h[:,None],(g['ns'],len(amps))).copy()
 if g['size']>g['ns']:h=np.r_[h,np.zeros((g['size']-g['ns'],len(amps)))]
 fg=np.ones_like(h);fn=np.zeros_like(h);ys=[];vo=None
 from .protocol import KNOTS
 cin=np.stack([np.interp(t,KNOTS,a) for a in amps])
 for j in range(len(t)):
  L,bc,vol=operators(name,blood,j,g,n,1)
  if j:
   matrix=L+diags(vol/.002);h=splu(matrix.tocsc()).solve(vo[:,None]/.002*h+bc[:,0,None]*cin[:,j])
   rhs=vo[:,None]/.002*fg+bc@np.ones((len(g['ends']),len(amps)))
   if name=='single':
    fg=np.stack([spsolve(matrix+diags(2*h[:,i]*vol),rhs[:,i]) for i in range(len(amps))],1)
    fn=vo[:,None]/vol[:,None]*fn+.002*2*h*fg
   else:
    gt=splu(matrix.tocsc()).solve(rhs);fg=gt*np.exp(-2*h*.002)
    fn[:g['ns']]=vo[:g['ns'],None]/vol[:g['ns'],None]*fn[:g['ns']]+gt[:g['ns']]-fg[:g['ns']]
  ys.append(np.stack([h[:g['ns']],fg[:g['ns']],fn[:g['ns']]]).transpose(2,0,1));vo=vol
 return np.stack(ys,-1)
def main():
 p=argparse.ArgumentParser();p.add_argument('network',choices=['single','three','seven']);p.add_argument('--kind',choices=['C0','new','old'],required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 configure();m=model(a.network,a.kind,a.seed);amps=inputs()[[0,31,63]];actual=m.predict(amps,True)
 if a.kind=='C0':expected=cpu_c0(a.network,amps)
 elif a.network=='single':
  from gp52.reaction import Coupled
  from gp52.runtime import ASSETS
  root=RUN/'single'/f'train-{a.seed}'
  mm=Coupled(transport_weights=root/'transport.pt' if a.kind=='new' else None)
  mm.reaction.load_state_dict(torch.load(root/'reaction.pt' if a.kind=='new' else ASSETS/'reaction-GG.pt',weights_only=True,map_location='cuda'))
  with torch.no_grad():expected=mm(torch.tensor(amps,device='cuda',dtype=torch.float32))[0].cpu().numpy()
 else:
  if a.network=='seven':
   from . import seven_cpu as old
   from .network_closure import reaction_features
  else:
   from gp52.three import evaluate as old
   from gp52.three.closure import reaction_features
  old.flow=lambda *args,**kwargs:(m.blood,m.t);old.inlets=lambda:(amps,{})
  with torch.no_grad():am,ta=[z.cpu().numpy().astype(float) for z in m.model.coefficients()]
  y=old.solve(am,ta)[0];feat=reaction_features(y,m.ar,.002)
  with torch.no_grad():fac=m.model.factor(torch.tensor(feat,device='cuda',dtype=m.model.reaction.input.weight.dtype)).cpu().numpy()
  expected=old.solve(am,ta,fac)[0]
 e=abs(actual-expected).max((0,2,3));assert e.max()<3e-5,e
 scale=np.ones((3,144))/12;j1=inv.jacobian(m,amps,scale,0,step=.002);j2=inv.jacobian(m,amps,scale,0,step=.001)
 gap=float(np.linalg.norm(j1-j2)/np.linalg.norm(j1));assert gap<.002,gap
 r=dict(max_absolute_by_species=e.tolist(),jacobian_step_halving=gap,minimum=float(actual.min()),cases=[0,31,63])
 out=RUN/a.network/'evaluation';out.mkdir(exist_ok=True);(out/f'verification-{m.name}.json').write_text(json.dumps(r,indent=2))
 print(json.dumps(r),flush=True)
if __name__=='__main__':main()
