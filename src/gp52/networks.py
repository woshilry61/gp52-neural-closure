"""Shared graph and equal-parameter pointwise closure heads."""
import torch
from torch import nn

class LocalGraph(nn.Module):
    def __init__(self,inputs,outputs,width=32):
        super().__init__()
        self.input=nn.Linear(inputs,width)
        self.center=nn.ModuleList(nn.Linear(width,width)for _ in range(2))
        self.neighbor=nn.ModuleList(nn.Linear(width,width,bias=False)for _ in range(2))
        self.head=nn.Linear(width,outputs)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x,use_graph=True):
        z=torch.tanh(self.input(x))
        for local,message in zip(self.center,self.neighbor):
            left=torch.cat([z[:,:1],z[:,:-1]],1);right=torch.cat([z[:,1:],z[:,-1:]],1)
            msg=message(.5*(left+right)) if use_graph else 0.
            z=torch.tanh(z+.25*(local(z)+msg))
        return self.head(z)

class LocalMLP(nn.Module):
    """Shared pointwise MLP. No cross-node reads, dummy weights or tied branches.
    Hidden sizes 32,64,32 and no input bias give the exact original parameter count:
    d*32 + (32*64+64) + (64*32+32) + (32*k+k).
    """
    def __init__(self,inputs,outputs):
        super().__init__()
        self.input=nn.Linear(inputs,32,bias=False)
        self.hidden1=nn.Linear(32,64);self.hidden2=nn.Linear(64,32)
        self.head=nn.Linear(32,outputs)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        return self.head(torch.tanh(self.hidden2(torch.tanh(self.hidden1(torch.tanh(self.input(x)))))))
