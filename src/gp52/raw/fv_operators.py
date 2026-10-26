"""Conservative BE/FV operators with spatially/time-varying axial diffusivity."""
import numpy as np
from scipy.linalg import solve_banded
def assemble(flow,D=2e-4,left=1.,right=0.):
    A=flow['area'];Af=flow['face_area'];q=flow['face_flow'];t=flow['t'];dx=float(flow['dx']);dt=float(flow['dt']);n,tt=A.shape
    D=np.broadcast_to(D,Af.shape);left=np.broadcast_to(left,t.shape);right=np.broadcast_to(right,t.shape)
    lo=np.zeros((tt-1,n));up=lo.copy();di=A[:,1:].T/dt;hi=A[:,:-1].T/dt;b=np.zeros_like(di)
    flux=q[1:-1,1:].T;conduct=(Af[1:-1,1:]*D[1:-1,1:]/dx).T
    lp=np.maximum(flux,0)+conduct;rp=np.minimum(flux,0)-conduct
    di[:,:-1]+=lp/dx;up[:,:-1]+=rp/dx;lo[:,1:]-=lp/dx;di[:,1:]-=rp/dx
    ql=q[0,1:];qr=q[-1,1:];kl=2*Af[0,1:]*D[0,1:]/dx;kr=2*Af[-1,1:]*D[-1,1:]/dx
    il=ql>=0;ir=qr<0
    di[:,0]+=np.where(il,kl,-ql)/dx;b[:,0]+=np.where(il,ql+kl,0)*left[1:]/dx
    di[:,-1]+=np.where(ir,kr,qr)/dx;b[:,-1]+=np.where(ir,kr-qr,0)*right[1:]/dx
    return dict(**flow,lower=lo,diag=di,upper=up,hist=hi,rhs=b,D=D)
def step(p,j,rhs,extra=0.):
    ab=np.zeros((3,len(p['x'])));ab[0,1:]=p['upper'][j,:-1];ab[1]=p['diag'][j]+extra;ab[2,:-1]=p['lower'][j,1:]
    return solve_banded((1,1),ab,rhs)
