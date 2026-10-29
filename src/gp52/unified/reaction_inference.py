"""FP64 evaluation of the unchanged reaction head with its fixed sparse adjacency."""
import torch,math
@torch.no_grad()
def _factor(head,x):
 if not hasattr(head,'_neighbor_index'):
  adj=head.adj;deg=int((adj!=0).sum(1).max())
  idx=torch.zeros((adj.shape[0],deg),device=adj.device,dtype=torch.long)
  weights=torch.zeros((adj.shape[0],deg),device=adj.device,dtype=adj.dtype)
  for i in range(len(adj)):
   js=torch.nonzero(adj[i],as_tuple=True)[0]
   idx[i,:len(js)]=js;weights[i,:len(js)]=adj[i,js]
  head._neighbor_index=idx;head._neighbor_weight=weights
 z=torch.tanh(head.input(x));idx=head._neighbor_index;w=head._neighbor_weight
 for local,message in zip(head.center,head.neighbor):
  mean=torch.zeros_like(z)
  for j in range(idx.shape[1]):mean.addcmul_(z.index_select(-2,idx[:,j]),w[:,j,None])
  z=torch.tanh(z+.25*(local(z)+message(mean)))
 return torch.exp(.25*torch.tanh(head.head(z)[...,0]))

@torch.no_grad()
def factor(head,x):
 # Bound the large FP64 hidden-state temporaries, independently of rollout batch size.
 block=max(1,4_000_000//math.prod(x.shape[1:-1]))
 if len(x)<=block:return _factor(head,x)
 return torch.cat([_factor(head,x[i:i+block]) for i in range(0,len(x),block)],0)
