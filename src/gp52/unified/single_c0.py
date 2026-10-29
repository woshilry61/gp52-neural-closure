"""Plain single-vessel C0; no memory flux or reaction correction.

The published C1 and frozen neural implementations are left unchanged.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from gp52.runtime import FlowPack
from gp52.baseline import band_solve

@triton.jit
def c0_kernel(F, INLET, INITIAL, OUT, DIAG,
              N:tl.constexpr,T:tl.constexpr,K:tl.constexpr,
              DT:tl.constexpr,DX:tl.constexpr,RATE:tl.constexpr):
    b=tl.program_id(0);r=tl.arange(0,K)
    h=tl.load(INITIAL+r,mask=r<N,other=0.)
    g=tl.load(INITIAL+N+r,mask=r<N,other=0.)
    fn=tl.load(INITIAL+2*N+r,mask=r<N,other=0.)
    aold=tl.load(F+r,mask=r<N,other=1.)
    v0=tl.sum(tl.where(r<N,aold,0.))*DX
    massh0=tl.sum(tl.where(r<N,aold*h,0.))*DX
    massgf0=tl.sum(tl.where(r<N,aold*(g+fn),0.))*DX
    cumh=massh0*0.;cumg=massh0*0.;massdef=massh0*0.
    minimum=tl.minimum(tl.min(h),tl.minimum(tl.min(g),tl.min(fn)))
    tl.store(OUT+((b*3)*N+r)*T,h,mask=r<N)
    tl.store(OUT+((b*3+1)*N+r)*T,g,mask=r<N)
    tl.store(OUT+((b*3+2)*N+r)*T,fn,mask=r<N)
    for j in range(1,T):
        off=j*6*K
        area=tl.load(F+off+r,mask=r<N,other=1.)
        af=tl.load(F+off+K+r,mask=r<N+1,other=0.)
        q=tl.load(F+off+2*K+r,mask=r<N+1,other=0.)
        plus=tl.maximum(q,0.)+af*2.e-4/DX
        minus=tl.minimum(q,0.)-af*2.e-4/DX
        rr=tl.minimum(r+1,K-1)
        qp=tl.gather(q,rr,0);ap=tl.gather(af,rr,0)
        pr=tl.gather(plus,rr,0);mr=tl.gather(minus,rr,0)
        lo=tl.where((r>0)&(r<N),-plus/DX,0.)
        up=tl.where(r<N-1,mr/DX,0.)
        di=area/DT+tl.where(r<N-1,pr/DX,0.)-tl.where(r>0,minus/DX,0.)
        kl=2*af*2.e-4/DX;kr=2*ap*2.e-4/DX
        di+=tl.where(r==0,(-tl.minimum(q,0.)+kl)/DX,0.)
        di+=tl.where(r==N-1,(tl.maximum(qp,0.)+tl.where(qp<0,kr,0.))/DX,0.)
        bc_left=tl.where(r==0,(tl.maximum(q,0.)+kl)/DX,0.)
        bc_right=tl.where((r==N-1)&(qp<0),(kr-qp)/DX,0.)
        cin=tl.load(INLET+b*T+j)
        rh=tl.where(r<N,aold/DT*h+bc_left*cin,0.)
        rg=tl.where(r<N,aold/DT*g+bc_left+bc_right,0.)
        di=tl.where(r<N,di,1.)
        h=band_solve(lo,di,up,rh,r,K)
        g=band_solve(lo,di+RATE*h*area,up,rg,r,K)
        fn=tl.where(r<N,aold/area*fn+DT*RATE*h*g,0.)
        hright=tl.sum(tl.where(r==N-1,h,0.))
        gright=tl.sum(tl.where(r==N-1,g,0.))
        hl=tl.sum(tl.where(r==0,tl.maximum(q,0.)*cin+tl.minimum(q,0.)*h-kl*(h-cin),0.))
        gl=tl.sum(tl.where(r==0,q-kl*(g-1.)+tl.minimum(q,0.)*(g-1.),0.))
        hr=tl.sum(tl.where(r==N,tl.maximum(q,0.)*hright+tl.where(q<0,2*af*2.e-4/DX*hright,0.),0.))
        gr=tl.sum(tl.where(r==N,tl.maximum(q,0.)*gright+tl.minimum(q,0.)-tl.where(q<0,2*af*2.e-4/DX*(1.-gright),0.),0.))
        cumh+=DT*(hl-hr);cumg+=DT*(gl-gr)
        errh=(tl.sum(tl.where(r<N,area*h,0.))*DX-massh0-cumh)/v0
        errg=(tl.sum(tl.where(r<N,area*(g+fn),0.))*DX-massgf0-cumg)/v0
        massdef=tl.maximum(massdef,tl.maximum(tl.abs(errh),tl.abs(errg)))
        minimum=tl.minimum(minimum,tl.minimum(tl.min(h),tl.minimum(tl.min(g),tl.min(fn))))
        tl.store(OUT+((b*3)*N+r)*T+j,h,mask=r<N)
        tl.store(OUT+((b*3+1)*N+r)*T+j,g,mask=r<N)
        tl.store(OUT+((b*3+2)*N+r)*T+j,fn,mask=r<N)
        aold=area
    tl.store(DIAG+b*2,massdef);tl.store(DIAG+b*2+1,minimum)

class C0(FlowPack):
    """single-c0-fv-v1: implicit FV, D=2e-4, rate=2 Th Fg."""
    def solve(self,amplitudes,theta=None,flow_ids=None,rate=2.,initial=None):
        if theta is not None and np.size(theta):
            raise ValueError("C0 has no closure parameters")
        if flow_ids is not None and np.any(np.asarray(flow_ids)!=0):
            raise ValueError("This frozen evaluation uses one blood-flow field")
        a=self.tensor(amplitudes);B=len(a)
        inlet=(a@self.basis).contiguous()
        out=torch.empty((B,3,self.N,self.T),device="cuda",dtype=torch.float64)
        diag=torch.empty((B,2),device="cuda",dtype=torch.float64)
        c0_kernel[(B,)](self.F,inlet,self.IC if initial is None else self.tensor(initial),out,diag,
            self.N,self.T,self.K,self.dt,self.dx,rate,num_warps=2,enable_fp_fusion=False)
        return out,diag
