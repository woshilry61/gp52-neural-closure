"""Differentiable conservative rollout; no radial concentration bins."""
import numpy as np
import torch
from torch import nn
from .runtime import FlowPack,ASSETS,tensor
from .networks import LocalGraph,LocalMLP

THETA=[.7873817261695291,-.09295961516979014,-.05423938854938835,-1.2230866117405637,-.07015216488900271]

def prefix_scan(F,b,initial):
    """F [...,time,S,S], b [...,time,S,cases]; returns all states."""
    b=torch.cat([b[...,:1,:,:]+F[...,:1,:,:]@initial,b[...,1:,:,:]],-3)
    d=1;T=F.shape[-3]
    while d<T:
        b=torch.cat([b[...,:d,:,:],b[...,d:,:,:]+F[...,d:,:,:]@b[...,:-d,:,:]],-3)
        if 2*d<T:
            F=torch.cat([F[...,:d,:,:],F[...,d:,:,:]@F[...,:-d,:,:]],-3)
        d*=2
    return b

def tridiagonal_solve(lower,diagonal,upper,rhs):
    """Parallel cyclic reduction: differentiable and CUDA-graph capturable."""
    n=diagonal.shape[-1];idx=torch.arange(n,device=diagonal.device)
    lo=torch.nn.functional.pad(lower,(1,0));up=torch.nn.functional.pad(upper,(0,1))
    d=diagonal;b=rhs;stride=1
    while stride<n:
        li=(idx-stride).clamp_min(0);ri=(idx+stride).clamp_max(n-1)
        al=torch.where(idx>=stride,-lo/d.index_select(-1,li),0.)
        ar=torch.where(idx+stride<n,-up/d.index_select(-1,ri),0.)
        b=b+al[...,None]*b.index_select(-2,li)+ar[...,None]*b.index_select(-2,ri)
        d=d+al*up.index_select(-1,li)+ar*lo.index_select(-1,ri)
        lo=al*lo.index_select(-1,li);up=ar*up.index_select(-1,ri)
        stride*=2
    return b/d[...,None]

class Rollout(nn.Module):
    def __init__(self,channels=2,kind="graph",dtype=torch.float32):
        super().__init__()
        if channels!=2:raise ValueError("Published checkpoints use two flux-memory channels.")
        self.channels=channels;self.kind=kind
        b=FlowPack()
        self.N=b.N;self.E=b.N+1;self.T=b.T;self.dt=b.dt;self.dx=b.dx
        self.S=self.N+self.channels*self.E;self.x=b.x;self.t=b.t;self.basetau=b.tau
        with np.load(ASSETS/"flow.npz") as z:
            for key in z.files:
                if key.startswith("buffer_"):
                    self.register_buffer(key[7:],tensor(z[key],dtype))
        if kind in ["graph","local"]:
            self.net=LocalGraph(8,2*channels,32).to(device="cuda",dtype=dtype)
            if kind=="local":
                rng=torch.get_rng_state()
                self.net=LocalMLP(8,2*channels).to(device="cuda",dtype=dtype)
                torch.set_rng_state(rng)
        elif kind=="linear":
            self.net=nn.Linear(8,2*channels).to(device="cuda",dtype=dtype)
            nn.init.zeros_(self.net.weight);nn.init.zeros_(self.net.bias)
        else:raise ValueError(kind)
    def coefficients(self):
        c=self.net(self.features)
        amps=[self.baseamp+.2*torch.tanh(c[...,0])]
        taus=[self.basetau*torch.exp(2*torch.tanh(c[...,1]))]
        for k in range(1,self.channels):
            amps.append(.2*torch.tanh(c[...,2*k]+.02))
            taus.append(.02*torch.exp(2*torch.tanh(c[...,2*k+1])))
        return torch.stack(amps,-1),torch.stack(taus,-1)
    def operators(self,amp,tau,reaction=None):
        N,E=self.N,self.E;dt,dx=self.dt,self.dx
        decay=torch.exp(-dt/tau);h=tau*(-torch.expm1(-dt/tau))
        diff=2e-4+(h*amp.square()).sum(-1)*self.inside
        plus=self.q.clamp_min(0)+self.af*diff/dx
        minus=self.q.clamp_max(0)-self.af*diff/dx
        d=self.area[1:]/dt
        pad=torch.nn.functional.pad
        d=d+pad(plus[:,1:-1]/dx,(0,1))-pad(minus[:,1:-1]/dx,(1,0))
        kl=2*self.af[:,0]*2e-4/dx;kr=2*self.af[:,-1]*2e-4/dx
        d=d+pad(((-self.q[:,0].clamp_max(0)+kl)/dx)[:,None],(0,N-1))
        d=d+pad(((self.q[:,-1].clamp_min(0)+torch.where(self.q[:,-1]<0,kr,0.))/dx)[:,None],(N-1,0))
        lower=-plus[:,1:-1]/dx;upper=minus[:,1:-1]/dx
        left=pad(((self.q[:,0].clamp_min(0)+kl)/dx)[:,None],(0,N-1))
        right=pad((torch.where(self.q[:,-1]<0,(kr-self.q[:,-1])/dx,0.))[:,None],(N-1,0))
        memrhs=[];memdecay=[];memgrad=[]
        for k in range(self.channels):
            memrhs.append(-self.divergence@((self.af*amp[...,k]*decay[...,k])[...,None]*self.advection))
            memdecay.append(decay[...,k,None]*self.advection)
            memgrad.append((h[...,k]*amp[...,k])[...,None]*self.gradient)
        rhs=torch.cat([torch.diag_embed(self.area[:-1]/dt),*memrhs,left[...,None],right[...,None]],-1)
        if reaction is not None:
            d=d[None]+reaction*self.area[None,1:]
            rhs=rhs[None].expand(reaction.shape[0],-1,-1,-1)
        top=tridiagonal_solve(lower,d,upper,rhs)
        bottoms=[]
        for k in range(self.channels):
            row=-memgrad[k]@top
            direct=torch.nn.functional.pad(memdecay[k],(N+k*E,(self.channels-k-1)*E+2))
            bottoms.append(row+direct)
        trans=torch.cat([top,*bottoms],-2)
        return trans[...,:self.S],trans[...,self.S:self.S+1],trans[...,self.S+1:self.S+2]
    def passive(self,amplitudes,coeff=None):
        amp,tau=self.coefficients() if coeff is None else coeff
        F,l,r=self.operators(amp,tau)
        inlet=amplitudes@self.basis
        b=l*inlet.T[1:,None,:]
        z0=torch.nn.functional.pad(self.initial[0],(0,self.channels*self.E))[None,:,None]
        z=prefix_scan(F,b,z0)
        th=torch.cat([self.initial[0][None,:,None].expand(len(amplitudes),-1,-1),z[:,:self.N].permute(2,1,0)],-1)
        return th,(amp,tau),z
    def forward(self,amplitudes):
        th,coeff,zth=self.passive(amplitudes)
        F,l,r=self.operators(*coeff,reaction=2*th[:,:,1:].transpose(1,2))
        z0=torch.nn.functional.pad(self.initial[1],(0,self.channels*self.E))[None,None,:,None]
        z=prefix_scan(F,l+r,z0)
        fg=torch.cat([self.initial[1][None,:,None].expand(len(amplitudes),-1,-1),z[...,:self.N,0].transpose(1,2)],-1)
        mass=(self.dt*self.area[None,1:].transpose(1,2)*2*th[:,:,1:]*fg[:,:,1:]).cumsum(-1)
        fn=torch.cat([torch.zeros_like(fg[:,:,:1]),mass/self.area[None,1:].transpose(1,2)],-1)
        return torch.stack([th,fg,fn],1),dict(th_state=zth,fg_state=z,amp=coeff[0],tau=coeff[1])
