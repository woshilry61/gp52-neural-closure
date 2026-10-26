import numpy as np
import torch,triton,triton.language as tl
from triton.language.extra.cuda import libdevice
from .runtime import FlowPack

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
def run_kernel(F,IX,W0,W1,INLET,THETA,FLOWID,INITIAL,OUT,DIAG,COEFF,FACTOR,
               N:tl.constexpr,T:tl.constexpr,K:tl.constexpr,
               DT:tl.constexpr,DX:tl.constexpr,DECAY:tl.constexpr,H:tl.constexpr,
               CHI:tl.constexpr,RATE:tl.constexpr):
    b=tl.program_id(0);fid=tl.load(FLOWID+b);r=tl.arange(0,K)
    h=tl.load(INITIAL+r,mask=r<N,other=0.)
    g=tl.load(INITIAL+N+r,mask=r<N,other=0.)
    fn=tl.load(INITIAL+2*N+r,mask=r<N,other=0.)
    mh=tl.full((K,),0,tl.float64);mg=tl.full((K,),0,tl.float64)
    nh=tl.full((K,),0,tl.float64);ng=tl.full((K,),0,tl.float64)
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
        amp=tl.load(COEFF+j*4*K+r,mask=r<N+1,other=0.)
        tau=tl.load(COEFF+j*4*K+K+r,mask=r<N+1,other=1.)
        amp2=tl.load(COEFF+j*4*K+2*K+r,mask=r<N+1,other=0.)
        tau2=tl.load(COEFF+j*4*K+3*K+r,mask=r<N+1,other=1.)
        decay=libdevice.exp(-DT/tau);hh=tau*(1.-decay)
        decay2=libdevice.exp(-DT/tau2);hh2=tau2*(1.-decay2)
        diffusion=2.e-4+tl.where((r>0)&(r<N),hh*amp*amp+hh2*amp2*amp2,0.)
        ix=tl.load(IX+(fid*T+j)*K+r,mask=r<N+1,other=0)
        w0=tl.load(W0+(fid*T+j)*K+r,mask=r<N+1,other=0.)
        w1=tl.load(W1+(fid*T+j)*K+r,mask=r<N+1,other=0.)
        ah=w0*tl.gather(mh,ix,0)+w1*tl.gather(mh,ix+1,0)
        ag=w0*tl.gather(mg,ix,0)+w1*tl.gather(mg,ix+1,0)
        bh=w0*tl.gather(nh,ix,0)+w1*tl.gather(nh,ix+1,0)
        bg=w0*tl.gather(ng,ix,0)+w1*tl.gather(ng,ix+1,0)
        valid=(r>0)&((r<N)|(q>=0))
        ah=tl.where(valid,ah,0.);ag=tl.where(valid,ag,0.)
        bh=tl.where(valid,bh,0.);bg=tl.where(valid,bg,0.)
        hist_h=af*(amp*decay*ah+amp2*decay2*bh);hist_g=af*(amp*decay*ag+amp2*decay2*bg)
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
        factor=tl.load(FACTOR+(b*(T-1)+j-1)*N+r,mask=r<N,other=1.).to(tl.float64)
        g=band_solve(lo,di+RATE*h*factor*area,up,rg,r,K)
        il=tl.maximum(r-1,0)
        reaction=RATE*h*g*factor
        fn=tl.where(r<N,aold/area*fn+DT*reaction,0.)
        # Each physical face has one flux, entering its adjacent cells with
        # opposite signs. Fg consumption and Fn generation use the same rate.
        grad_h=(h-tl.gather(h,il,0))/DX
        grad_g=(g-tl.gather(g,il,0))/DX
        mh=tl.where(valid,decay*ah-hh*amp*tl.where(r<N,grad_h,0.),0.)
        mg=tl.where(valid,decay*ag-hh*amp*tl.where(r<N,grad_g,0.),0.)
        nh=tl.where(valid,decay2*bh-hh2*amp2*tl.where(r<N,grad_h,0.),0.)
        ng=tl.where(valid,decay2*bg-hh2*amp2*tl.where(r<N,grad_g,0.),0.)
        hright=tl.sum(tl.where(r==N-1,h,0.));gright=tl.sum(tl.where(r==N-1,g,0.))
        hl=tl.sum(tl.where(r==0,tl.maximum(q,0.)*cin+tl.minimum(q,0.)*h-kl*(h-cin),0.))
        gl=tl.sum(tl.where(r==0,q-kl*(g-1.)+tl.minimum(q,0.)*(g-1.),0.))
        hr=tl.sum(tl.where(r==N,tl.maximum(q,0.)*hright+af*(amp*mh+amp2*nh)+tl.where(q<0,2*af*2.e-4/DX*hright,0.),0.))
        gr=tl.sum(tl.where(r==N,tl.maximum(q,0.)*gright+tl.minimum(q,0.)+af*(amp*mg+amp2*ng)-tl.where(q<0,2*af*2.e-4/DX*(1.-gright),0.),0.))
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


ConservativeGraph = FlowPack

class Corrector:
    """Frozen shared edge map, two flux-memory channels, GPU finite volumes."""
    def __init__(self,model):
        self.base=ConservativeGraph()
        self.tensor=self.base.tensor
        b=self.base
        self.N,self.T,self.x,self.t,self.basis=b.N,b.T,b.x,b.t,b.basis
        with torch.no_grad():amp,tau=model.coefficients()
        self.coefficients=torch.zeros(b.T,4,b.K,device="cuda",dtype=torch.float64)
        values=torch.stack([amp[...,0],tau[...,0],amp[...,1],tau[...,1]],1)
        self.coefficients[1:,:,:b.N+1]=values.double()
    def solve(self,amplitudes,factor):
        b=self.base;a=b.tensor(amplitudes);B=len(a)
        params=b.tensor(np.zeros((B,5)))
        fi=b.tensor(np.zeros(B,np.int32),torch.int32)
        inlet=(a@b.basis).contiguous()
        out=torch.empty((B,3,b.N,b.T),device="cuda",dtype=torch.float64)
        diag=torch.empty((B,2),device="cuda",dtype=torch.float64)
        run_kernel[(B,)](b.F,b.IX,b.W0,b.W1,inlet,params,fi,b.IC,out,diag,self.coefficients,factor.contiguous(),
            b.N,b.T,b.K,b.dt,b.dx,b.decay,b.h,0.,2.,num_warps=2,enable_fp_fusion=False)
        return out,diag
