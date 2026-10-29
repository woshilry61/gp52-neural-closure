"""Batched CUDA evaluation of the frozen three-vessel forward models.

The time integrator, memory update and predictor/corrector are identical to
evaluate.solve. No model fitting occurs here. Th is affine in inlet amplitudes.
"""
from pathlib import Path
import json,time
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
@triton.jit
def integrate(H,INV,OLD,BC,VOLR,DEC,HA,AFAD,ELEFT,ERIGHT,EA,EB,DX,FACTOR,OUTG,OUTF,
              T:tl.constexpr,K:tl.constexpr,N:tl.constexpr,NEURAL:tl.constexpr,WITHFACTOR:tl.constexpr):
 c=tl.program_id(0);i=tl.arange(0,K)
 gg=tl.full((K,),1.,INV.dtype.element_ty);fn=tl.full((K,),0.,INV.dtype.element_ty)
 m0=tl.full((K,),0.,INV.dtype.element_ty);m1=tl.full((K,),0.,INV.dtype.element_ty)
 el=tl.load(ELEFT+i);er=tl.load(ERIGHT+i)
 ea=tl.load(EA+i);eb=tl.load(EB+i);dx=tl.load(DX+i)
 tl.store(OUTG+c*T*K+i,gg);tl.store(OUTF+c*T*K+i,fn)
 for j in range(1,T):
  hh=tl.load(H+c*T*K+j*K+i)
  rhs=tl.load(OLD+j*K+i)*gg+tl.load(BC+j*K+i)
  if NEURAL:
   d0=tl.load(DEC+j*K*2+i*2);d1=tl.load(DEC+j*K*2+i*2+1)
   mf=tl.load(AFAD+j*K*2+i*2)*m0+tl.load(AFAD+j*K*2+i*2+1)*m1
   rhs+=tl.where(el>=0,tl.gather(mf,tl.maximum(el,0),axis=0),0.)
   rhs-=tl.where(er>=0,tl.gather(mf,tl.maximum(er,0),axis=0),0.)
  gt=tl.full((K,),0.,INV.dtype.element_ty)
  rows=tl.arange(0,64)
  for block in range(K//64):
   mat=tl.load(INV+j*K*K+(rows[:,None]+block*64)*K+i[None,:])
   part=tl.sum(mat*rhs[None,:],1)
   gt=tl.where((i>=block*64)&(i<(block+1)*64),tl.gather(part,i%64,axis=0),gt)
  if NEURAL:
   grad=(tl.gather(gt,eb,axis=0)-tl.gather(gt,ea,axis=0))/dx
   m0=d0*m0-tl.load(HA+j*K*2+i*2)*grad
   m1=d1*m1-tl.load(HA+j*K*2+i*2+1)*grad
  fac=tl.load(FACTOR+c*T*K+j*K+i) if WITHFACTOR else 1.
  if INV.dtype.element_ty == tl.float64:
   gg=gt*libdevice.exp(-.004*hh*fac)
  else:
   gg=gt*tl.exp(-.004*hh*fac)
  fn=tl.where(i<N,tl.load(VOLR+j*K+i)*fn+gt-gg,0.)
  tl.store(OUTG+c*T*K+j*K+i,gg);tl.store(OUTF+c*T*K+j*K+i,fn)

@triton.jit
def features_kernel(H,G,R,AR,UV,LEFT,RIGHT,GDEN,OUT,T:tl.constexpr,K:tl.constexpr,N:tl.constexpr):
 c=tl.program_id(0);i=tl.arange(0,K);l=tl.load(LEFT+i);r=tl.load(RIGHT+i);denx=tl.load(GDEN+i)
 h02=tl.full((K,),0.,H.dtype.element_ty);g02=h02;h10=h02;g10=h02
 for j in range(T):
  h=tl.load(H+c*T*K+j*K+i);g=tl.load(G+c*T*K+j*K+i)
  rr=tl.load(R+j*K+i)
  gh=(tl.gather(h,r,0)-tl.gather(h,l,0))/denx*rr
  gg=(tl.gather(g,r,0)-tl.gather(g,l,0))/denx*rr
  h02=tl.where(j==0,gh,.9048374180359595*h02+.09516258196404043*gh)
  g02=tl.where(j==0,gg,.9048374180359595*g02+.09516258196404043*gg)
  h10=tl.where(j==0,gh,.9801986733067553*h10+.019801326693244747*gh)
  g10=tl.where(j==0,gg,.9801986733067553*g10+.019801326693244747*gg)
  p=(c*T*N+j*N+i)*10;mask=i<N;hg=h*g;den=hg+1e-4
  tl.store(OUT+p,h,mask);tl.store(OUT+p+1,g,mask);tl.store(OUT+p+2,hg,mask)
  tl.store(OUT+p+3,gh,mask);tl.store(OUT+p+4,gg,mask)
  tl.store(OUT+p+5,tl.load(AR+j*K+i)/.001-1,mask)
  tl.store(OUT+p+6,tl.load(UV+j*K+i)/.2,mask)
  x=gh*gg/den
  val=libdevice.tanh(x) if H.dtype.element_ty==tl.float64 else 2/(1+tl.exp(-2*x))-1
  tl.store(OUT+p+7,val,mask)
  x=h02*g02/den
  val=libdevice.tanh(x) if H.dtype.element_ty==tl.float64 else 2/(1+tl.exp(-2*x))-1
  tl.store(OUT+p+8,val,mask)
  x=h10*g10/den
  val=libdevice.tanh(x) if H.dtype.element_ty==tl.float64 else 2/(1+tl.exp(-2*x))-1
  tl.store(OUT+p+9,val,mask)
