"""Strict 1D spatially developing causal flux closure. One mean concentration per species/section.

The face convolution is a derived functional of axial gradients. It has no radial
grid, concentration profile, or CFD input. Recurrence caches evaluate the
Volterra integral; they are not independently predicted concentration states.
"""
from pathlib import Path
import sys,numpy as np
from .axial_operators import assemble,step
def shear(flow,pars):
    alpha,beta,tau_u,tau_c,chi=pars[:5]
    u=flow["face_flow"]/flow["face_area"];s=np.empty_like(u);s[:,0]=u[:,0]
    dt=float(flow["dt"]);decay=np.exp(-dt/tau_u)
    for j in range(1,u.shape[1]):s[:,j]=decay*s[:,j-1]+(1-decay)*u[:,j]
    amp=alpha*u+beta*s
    if len(pars)>5:
        z=2*flow["xf"]/flow["xf"][-1]-1
        amp=amp*np.exp(np.polynomial.legendre.legval(z,np.r_[0.,pars[5:]]))[:,None]
    return amp,tau_c,chi
def solve(flow,pars=(0.,0.,.1,.1,0.),inlet=None,uniform=False):
    A=flow["area"];Af=flow["face_area"];Q=flow["face_flow"]
    t=flow["t"];x=flow["x"];dt=float(flow["dt"]);dx=float(flow["dx"]);n=len(x);reservoir=bool(flow.get("inlet_reservoir",False))
    amp,tau,chi=shear(flow,pars)
    decay=np.exp(-dt/tau);h=tau*(1-decay)
    d=2e-4+h*amp**2
    d[[0,-1]]=2e-4
    ramp=np.minimum(t/.2,1.);left=np.stack([ramp,.8*ramp if inlet is None else inlet,np.ones(len(t))])
    right=np.zeros_like(left);right[2]=1.
    y=np.zeros((4,n,len(t)));y[1,:,0]=.2*np.exp(-((x/.06-.25)/.1)**2);y[2,:,0]=1.
    if uniform:y[:3,:,0]=1.;left[:]=1.;right[:]=1.
    ops=[assemble(flow,d,left[i],right[i])for i in range(3)]
    memory=np.zeros((3,n+1));fluxes=np.zeros((3,n+1,len(t)));rates=np.zeros((n,len(t)))
    for j in range(1,len(t)):
        departure=flow["xf"]-(Q[:,j]/Af[:,j])*dt
        advected=np.stack([np.interp(departure,flow["xf"],v,left=0.,right=0.) for v in memory])
        histflux=Af[:,j][None,:]*amp[:,j][None,:]*decay*advected
        # Inflow has zero covariance; outgoing characteristics carry flux history.
        histflux[:,0]*=(not reservoir and Q[0,j]<0) if len(pars)>5 else False;histflux[:,-1]*=(Q[-1,j]>=0) if len(pars)>5 else False
        rhs=[ops[i]["hist"][j-1]*y[i,:,j-1]+ops[i]["rhs"][j-1]-np.diff(histflux[i])/dx for i in range(3)]
        cp=step(ops[0],j-1,rhs[0]);th=step(ops[1],j-1,rhs[1])
        fg=step(ops[2],j-1,rhs[2],2*th*A[:,j] if not uniform else 0.)
        factor=np.ones(n)
        if chi and not uniform:
            for _ in range(3):
                cross=np.gradient(th,dx)*np.gradient(fg,dx)*A[:,j]/np.pi
                factor=np.exp(chi*np.tanh(cross/(th*fg+1e-4)))
                fg=step(ops[2],j-1,rhs[2],2*th*factor*A[:,j])
        reaction=2*th*fg*factor if not uniform else np.zeros(n)
        fn=A[:,j-1]/A[:,j]*y[3,:,j-1]+dt*reaction
        y[:,:,j]=[cp,th,fg,fn];rates[:,j]=reaction
        grad=np.zeros((3,n+1))
        grad[:,1:-1]=np.diff(y[:3,:,j],axis=1)/dx
        grad[:,0]=0.
        grad[:,-1]=0.
        memory=decay*advected-h*amp[:,j][None,:]*grad
        memory[:,0]*=(not reservoir and Q[0,j]<0) if len(pars)>5 else False;memory[:,-1]*=(Q[-1,j]>=0) if len(pars)>5 else False
        fluxes[:,:,j]=Af[:,j][None,:]*amp[:,j][None,:]*memory
    return dict(y=y,flux=fluxes,reaction=rates,x=x,t=t,area=A)
