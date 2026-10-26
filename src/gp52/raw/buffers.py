"""Construct the exact original float32 transport buffers from new blood fields."""
import numpy as np
import torch
from torch import nn
from ..runtime import tensor
from ..transport import THETA

class Buffers(nn.Module):
    def __init__(self,old,dtype=torch.float32):
        super().__init__();self.channels=2
        self.N=old.N;self.E=old.N+1;self.T=old.T;self.dt=old.dt;self.dx=old.dx
        self.S=self.N+self.channels*self.E;self.x=old.x;self.t=old.t
        pack=old.F[0].to(dtype);N=self.N;E=self.E
        def buf(k,v):self.register_buffer(k,v.to(dtype).contiguous())
        buf("area",pack[:,0,:N]);buf("af",pack[1:,1,:E]);buf("q",pack[1:,2,:E])
        buf("basis",old.basis);buf("initial",old.IC)
        xf=tensor(old.f["xf"],dtype);u=pack[:,4,:E]*.2
        x=2*xf/.06-1
        th=tensor(THETA,dtype)
        amp=pack[:,3,:E]*torch.exp(.5*torch.tanh(th[0]+th[1]*x+th[2]*pack[:,4,:E]+th[3]*pack[:,5,:E]))
        buf("baseamp",amp[1:]);self.basetau=old.tau
        hist=[]
        for tau in [.01,.05,.2]:
            h=[u[0]];d=np.exp(-self.dt/tau)
            for j in range(1,self.T):h.append(d*h[-1]+(1-d)*u[j])
            hist.append(torch.stack(h))
        du=torch.cat([u[:,1:2]-u[:,:1],.5*(u[:,2:]-u[:,:-2]),u[:,-1:]-u[:,-2:-1]],1)
        feat=torch.stack([x.expand_as(u),pack[:,1,:E]/.001-1,u/.2,
                          hist[0]/.2,hist[1]/.2,hist[2]/.2,du/.2,
                          torch.sqrt(pack[:,1,:E]/torch.pi)/.06],-1)
        buf("features",feat[1:])
        ix=old.IX[0,1:,:E].long()
        p=torch.zeros((self.T-1,E,E),device="cuda",dtype=dtype)
        p.scatter_add_(2,ix[...,None],old.W0[0,1:,:E,None].to(dtype))
        p.scatter_add_(2,(ix+1)[...,None],old.W1[0,1:,:E,None].to(dtype))
        mask=torch.cat([torch.zeros_like(self.q[:,:1]),torch.ones_like(self.q[:,1:-1]),(self.q[:,-1:]>=0).to(dtype)],1)
        buf("advection",p*mask[...,None])
        G=torch.zeros((E,N),device="cuda",dtype=dtype)
        i=torch.arange(N-1,device="cuda");G[i+1,i]=-1/self.dx;G[i+1,i+1]=1/self.dx
        V=torch.zeros((N,E),device="cuda",dtype=dtype)
        i=torch.arange(N,device="cuda");V[i,i]=-1/self.dx;V[i,i+1]=1/self.dx
        buf("gradient",G);buf("divergence",V)
        buf("inside",torch.cat([tensor([0],dtype),torch.ones(E-2,device="cuda",dtype=dtype),tensor([0],dtype)]))
