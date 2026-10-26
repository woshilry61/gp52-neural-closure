"""Axial FV boundary choices; old flow-switched and persistent upstream reservoir."""
import numpy as np
from .fv_operators import assemble as flow_switched_assemble,step
def assemble(flow,D=2e-4,left=1.,right=0.):
 p=flow_switched_assemble(flow,D,left,right)
 if bool(flow.get("inlet_reservoir",False)):
  # Upstream diffusive reservoir remains connected during zero/reverse flow.
  # Advection retains its conservative upwind direction.
  q=flow["face_flow"][0,1:];dx=float(flow["dx"]);d=np.broadcast_to(D,flow["face_area"].shape)
  conduct=2*flow["face_area"][0,1:]*d[0,1:]/dx
  extra=np.where(q<0,conduct,0.)/dx
  p["diag"][:,0]+=extra;p["rhs"][:,0]+=extra*np.broadcast_to(left,flow["t"].shape)[1:]
 return p
