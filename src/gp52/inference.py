"""Fast predictor-corrector inference and sparse inverse interface."""
import torch
from .reaction import Coupled,features
from .runtime import ASSETS
from .corrector import Corrector
from .predictor import Deployed

class Solver:
    def __init__(self,variant='GG',weights=None,transport_weights=None):
        kind={'GG':'graph','GL':'local','LG':'graph','LL':'local','polynomial':'polynomial'}[variant]
        transport_kind='local' if variant in ['LG','LL'] else 'graph'
        self.model=Coupled(kind,transport_kind=transport_kind,transport_weights=transport_weights);m=self.model
        m.reaction.load_state_dict(torch.load(weights or ASSETS/f'reaction-{variant}.pt',map_location='cuda',weights_only=True))
        m.eval().requires_grad_(False)
        self.predictor=Deployed(m.transport);self.corrector=Corrector(m.transport)
        m.reaction.double()
        self.N=m.transport.N;self.T=m.transport.T;self.x=m.transport.x;self.t=m.transport.t
        self.tensor=self.predictor.tensor
    @torch.inference_mode()
    def solve(self,amplitudes,theta=None,flow_ids=None):
        a=self.tensor(amplitudes);outs=[];diags=[]
        limit=128 if self.model.reaction.kind=='graph' else 64
        for start in range(0,len(a),limit):
            aa=a[start:start+limit];base,_=self.predictor.solve(aa)
            m=self.model
            xf=features(base,*[v.double() for v in m.blood()],m.transport.dx,m.transport.dt)
            factor=m.reaction(xf)
            y,diag=self.corrector.solve(aa,factor)
            outs.append(y);diags.append(diag)
        return torch.cat(outs),torch.cat(diags)


def get_solver(name='GG'):
    if name=='C1':
        from .baseline import Baseline
        return Baseline()
    if name=='transport':
        from .reaction import frozen_transport
        return Deployed(frozen_transport())
    return Solver(name)
