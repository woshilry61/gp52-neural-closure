"""Causal reaction predictor-corrector and physical source features."""
import torch
from torch import nn
from .transport import Rollout,prefix_scan
from .networks import LocalGraph,LocalMLP
from .runtime import ASSETS

def history(x,dt,tau):
    # Parallel causal exponential filter along dimension 1, with h(0)=x(0).
    d=x.new_tensor(-dt/tau).exp()
    h=torch.cat([x[:,:1],(1-d)*x[:,1:]],1);stride=1
    while stride<x.shape[1]:
        h=torch.cat([h[:,:stride],h[:,stride:]+d.pow(stride)*h[:,:-stride]],1)
        stride*=2
    return h

def features(y,area,q,af,dx,dt):
    # y [batch,3,cell,time], blood [time,cell/face], include initial time.
    h=y[:,0].transpose(1,2);g=y[:,1].transpose(1,2)
    def grad(v):
        return torch.cat([(v[...,1:2]-v[...,:1])/dx,(v[...,2:]-v[...,:-2])/(2*dx),(v[...,-1:]-v[...,-2:-1])/dx],-1)
    radius=(area/torch.pi).sqrt()[None]
    gh=grad(h)*radius;gg=grad(g)*radius
    u=.5*(q[:,:-1]/af[:,:-1]+q[:,1:]/af[:,1:])
    cross=gh*gg/(h*g+1e-4)
    old=[h,g,h*g,gh,gg,(area/.001-1)[None].expand_as(h),(u/.2)[None].expand_as(h),cross.tanh()]
    # Two scalar temporal descriptors, not radial bins or radial concentrations.
    for tau in [.02,.1]:
        old.append((history(gh,dt,tau)*history(gg,dt,tau)/(h*g+1e-4)).tanh())
    return torch.stack(old,-1)[:,1:]

class ReactionHead(nn.Module):
    def __init__(self,kind='graph'):
        super().__init__();self.kind=kind
        if kind=='graph':self.net=LocalGraph(10,1,32)
        elif kind=='local':self.net=LocalMLP(10,1)
        elif kind=='polynomial':
            self.net=nn.Linear(495,1);nn.init.zeros_(self.net.weight);nn.init.zeros_(self.net.bias)
            self.register_buffer('pairs',torch.triu_indices(30,30))
        else:raise ValueError(kind)
    def forward(self,x):
        shape=x.shape[:-1];z=x.reshape(-1,x.shape[-2],10)
        if self.kind in ['graph','local']:v=self.net(z)[...,0]
        else:
            def neighbor(a):return .5*(torch.cat([a[:,:1],a[:,:-1]],1)+torch.cat([a[:,1:],a[:,-1:]],1))
            first=neighbor(z);a=torch.cat([z,first,neighbor(first)],-1)
            phi=torch.cat([a,a[...,self.pairs[0]]*a[...,self.pairs[1]]],-1)
            # bias supplies the constant term: 30+465 features.
            v=self.net(phi)[...,0]
        return torch.exp(.25*torch.tanh(v.reshape(shape)))

def frozen_transport(dtype=torch.float32,kind='graph',weights=None):
    m=Rollout(kind=kind,channels=2,dtype=dtype)
    path=weights or ASSETS/('transport-graph.pt' if kind=='graph' else 'transport-local.pt')
    state=torch.load(path,map_location='cuda',weights_only=True)
    if 'state' in state:state={k[4:]:v for k,v in state['state'].items() if k.startswith('net.')}
    m.net.load_state_dict(state);m.eval().requires_grad_(False)
    return m

class Coupled(nn.Module):
    def __init__(self,kind='graph',dtype=torch.float32,transport_kind='graph',transport_weights=None):
        super().__init__();self.transport=frozen_transport(dtype,transport_kind,transport_weights);self.reaction=ReactionHead(kind).to(device='cuda',dtype=dtype)
    def blood(self):
        m=self.transport
        return m.area,torch.cat([m.q[:1],m.q],0),torch.cat([m.af[:1],m.af],0)
    def predictor(self,amplitudes):
        m=self.transport;y,detail=m(amplitudes)
        return dict(y=y,coeff=(detail['amp'],detail['tau']),features=features(y,*self.blood(),m.dx,m.dt))
    def correct(self,cache,factor=None):
        m=self.transport;th=cache['y'][:,0];B=len(th)
        factor=self.reaction(cache['features']) if factor is None else factor
        F,l,r=m.operators(*cache['coeff'],reaction=2*th[:,:,1:].transpose(1,2)*factor)
        z0=torch.nn.functional.pad(m.initial[1],(0,m.channels*m.E))[None,None,:,None]
        z=prefix_scan(F,l+r,z0)
        fg=torch.cat([m.initial[1][None,:,None].expand(B,-1,-1),z[...,:m.N,0].transpose(1,2)],-1)
        rate=2*th[:,:,1:]*fg[:,:,1:]*factor.transpose(1,2)
        mass=(m.dt*m.area[None,1:].transpose(1,2)*rate).cumsum(-1)
        fn=torch.cat([torch.zeros_like(fg[:,:,:1]),mass/m.area[None,1:].transpose(1,2)],-1)
        return torch.stack([th,fg,fn],1),factor
    def forward(self,amplitudes):return self.correct(self.predictor(amplitudes))
