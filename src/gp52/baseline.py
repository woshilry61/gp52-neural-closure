import numpy as np
import torch,triton,triton.language as tl
from triton.language.extra.cuda import libdevice
from .runtime import FlowPack

from .transport import THETA

@triton.jit
def band_solve(lo,di,up,rhs,r,K:tl.constexpr):
    for level in tl.static_range(0,6):
        stride=1<<level
        li=tl.maximum(r-stride,0);ri=tl.minimum(r+stride,K-1)
        dl=tl.gather(di,li,0);dr=tl.gather(di,ri,0)
        al=tl.where(r>=stride,-lo/dl,0.)
        ar=tl.where(r+stride<K,-up/dr,0.)
        rhs=rhs+al*tl.gather(rhs,li,0)+ar*tl.gather(rhs,ri,0)
        di=di+al*tl.gather(up,li,0)+ar*tl.gather(lo,ri,0)
        lo=al*tl.gather(lo,li,0);up=ar*tl.gather(up,ri,0)
    return rhs/di

@triton.jit
def run_kernel(F,IX,W0,W1,INLET,THETA,FLOWID,INITIAL,OUT,DIAG,
               N:tl.constexpr,T:tl.constexpr,K:tl.constexpr,
               DT:tl.constexpr,DX:tl.constexpr,DECAY:tl.constexpr,H:tl.constexpr,
               CHI:tl.constexpr,RATE:tl.constexpr):
    b=tl.program_id(0);fid=tl.load(FLOWID+b);r=tl.arange(0,K)
    h=tl.load(INITIAL+r,mask=r<N,other=0.)
    g=tl.load(INITIAL+N+r,mask=r<N,other=0.)
    fn=tl.load(INITIAL+2*N+r,mask=r<N,other=0.)
    mh=tl.full((K,),0,tl.float64);mg=tl.full((K,),0,tl.float64)
    aold=tl.load(F+fid*T*6*K+r,mask=r<N,other=1.)
    v0=tl.sum(tl.where(r<N,aold,0))*DX
    massh0=tl.sum(tl.where(r<N,aold*h,0))*DX
    massgf0=tl.sum(tl.where(r<N,aold*(g+fn),0))*DX
    cumh=massh0*0.;cumg=massh0*0.;massdef=massh0*0.;minimum=tl.minimum(tl.min(h),tl.min(g))
    p0=tl.load(THETA+b*5);p1=tl.load(THETA+b*5+1)
    p2=tl.load(THETA+b*5+2);p3=tl.load(THETA+b*5+3)
    chi=CHI+.25*libdevice.tanh(tl.load(THETA+b*5+4))
    tl.store(OUT+((b*3)*N+r)*T,h,mask=r<N)
    tl.store(OUT+((b*3+1)*N+r)*T,g,mask=r<N)
    tl.store(OUT+((b*3+2)*N+r)*T,fn,mask=r<N)
    for j in range(1,T):
        off=(fid*T+j)*6*K
        area=tl.load(F+off+r,mask=r<N,other=1.)
        af=tl.load(F+off+K+r,mask=r<N+1,other=0.)
        q=tl.load(F+off+2*K+r,mask=r<N+1,other=0.)
        amp0=tl.load(F+off+3*K+r,mask=r<N+1,other=0.)
        uf=tl.load(F+off+4*K+r,mask=r<N+1,other=0.)
        sf=tl.load(F+off+5*K+r,mask=r<N+1,other=0.)
        amp=amp0*libdevice.exp(.5*libdevice.tanh(p0+p1*(2.*r/N-1.)+p2*uf+p3*sf))
        diffusion=2.e-4+tl.where((r>0)&(r<N),H*amp*amp,0.)
        ix=tl.load(IX+(fid*T+j)*K+r,mask=r<N+1,other=0)
        w0=tl.load(W0+(fid*T+j)*K+r,mask=r<N+1,other=0.)
        w1=tl.load(W1+(fid*T+j)*K+r,mask=r<N+1,other=0.)
        ah=w0*tl.gather(mh,ix,0)+w1*tl.gather(mh,ix+1,0)
        ag=w0*tl.gather(mg,ix,0)+w1*tl.gather(mg,ix+1,0)
        valid=(r>0)&((r<N)|(q>=0))
        ah=tl.where(valid,ah,0.);ag=tl.where(valid,ag,0.)
        hist_h=af*amp*DECAY*ah;hist_g=af*amp*DECAY*ag
        plus=tl.maximum(q,0.)+af*diffusion/DX
        minus=tl.minimum(q,0.)-af*diffusion/DX
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
        rh=aold/DT*h+bc_left*cin-(tl.gather(hist_h,rr,0)-hist_h)/DX
        rg=aold/DT*g+bc_left+bc_right-(tl.gather(hist_g,rr,0)-hist_g)/DX
        di=tl.where(r<N,di,1.);rh=tl.where(r<N,rh,0.);rg=tl.where(r<N,rg,0.)
        h=band_solve(lo,di,up,rh,r,K)
        factor=tl.full((K,),1.,tl.float64)
        g=band_solve(lo,di+RATE*h*area,up,rg,r,K)
        il=tl.maximum(r-1,0);ir=tl.minimum(r+1,N-1)
        gh=(tl.gather(h,ir,0)-tl.gather(h,il,0))/(tl.maximum(ir-il,1)*DX)
        for repeat in tl.static_range(3):
            gg=(tl.gather(g,ir,0)-tl.gather(g,il,0))/(tl.maximum(ir-il,1)*DX)
            cross=gh*gg*area/3.14159265358979323846
            factor=libdevice.exp(chi*libdevice.tanh(cross/(h*g+1.e-4)))
            g=band_solve(lo,di+RATE*h*factor*area,up,rg,r,K)
        reaction=RATE*h*g*factor
        fn=tl.where(r<N,aold/area*fn+DT*reaction,0.)
        # Each physical face has one flux, entering its adjacent cells with
        # opposite signs. Fg consumption and Fn generation use the same rate.
        grad_h=(h-tl.gather(h,il,0))/DX
        grad_g=(g-tl.gather(g,il,0))/DX
        mh=tl.where(valid,DECAY*ah-H*amp*tl.where(r<N,grad_h,0.),0.)
        mg=tl.where(valid,DECAY*ag-H*amp*tl.where(r<N,grad_g,0.),0.)
        hright=tl.sum(tl.where(r==N-1,h,0.));gright=tl.sum(tl.where(r==N-1,g,0.))
        hl=tl.sum(tl.where(r==0,tl.maximum(q,0.)*cin+tl.minimum(q,0.)*h-kl*(h-cin),0.))
        gl=tl.sum(tl.where(r==0,q-kl*(g-1.)+tl.minimum(q,0.)*(g-1.),0.))
        hr=tl.sum(tl.where(r==N,tl.maximum(q,0.)*hright+af*amp*mh+tl.where(q<0,2*af*2.e-4/DX*hright,0.),0.))
        gr=tl.sum(tl.where(r==N,tl.maximum(q,0.)*gright+tl.minimum(q,0.)+af*amp*mg-tl.where(q<0,2*af*2.e-4/DX*(1.-gright),0.),0.))
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

class Baseline(FlowPack):
    def solve(self,amplitudes,theta=None,flow_ids=None,rate=2.,initial=None):
        a=self.tensor(amplitudes);B=len(a)
        if theta is None:theta=THETA
        theta=self.tensor(theta)
        if theta.ndim==1:theta=theta.expand(B,-1).contiguous()
        fi=self.tensor(np.zeros(B,np.int32) if flow_ids is None else flow_ids,torch.int32)
        inlet=(a@self.basis).contiguous();out=torch.empty((B,3,self.N,self.T),device="cuda",dtype=torch.float64)
        diag=torch.empty((B,2),device="cuda",dtype=torch.float64)
        run_kernel[(B,)](self.F,self.IX,self.W0,self.W1,inlet,theta,fi,self.IC if initial is None else self.tensor(initial),out,diag,
            self.N,self.T,self.K,self.dt,self.dx,self.decay,self.h,self.pars[4],rate,num_warps=2,enable_fp_fusion=False)
        return out,diag
