"""Verify sparse FP64 adjacency evaluation against the unchanged dense head."""
import json,torch
from .protocol import RUN
from .models import Three
from .network_closure import Head,graph,COUNTS
from .reaction_inference import factor
def main():
 torch.manual_seed(52703104)
 m=Three('new',42)
 seven=Head(10,1,graph(COUNTS)).cuda().double()
 with torch.no_grad():
  seven.head.weight.normal_(0,.1);seven.head.bias.normal_(0,.1)
 records={}
 for name,head in [('three',m.model.reaction),('seven',seven)]:
  x=torch.randn((2,3,len(head.adj),10),device='cuda',dtype=torch.float64)
  with torch.no_grad():
   dense=torch.exp(.25*torch.tanh(head(x)[...,0]))
   sparse=factor(head,x)
  error=float(abs(dense-sparse).max());assert error<1e-12,error
  records[name]=dict(max_absolute=error,nonzero_output_weights=True)
 (RUN/'reaction-inference-parity.json').write_text(json.dumps(records,indent=2));print(records)
if __name__=='__main__':main()
