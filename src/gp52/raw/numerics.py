"""Shared units, metrics and grid maps for the Project 52 verification suite."""
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
def mechanics(x):
    R=-.0033667*np.asarray(x)+.018206
    a0=3.14*R**2
    Eh=.1*(3000000*np.exp(-900*R)+489871.1841)*R
    return a0,(4/3)*1.772*Eh/a0
def metrics(pred,ref,weight=None,mask=None):
    p=np.asarray(pred);r=np.asarray(ref)
    w=np.ones_like(r) if weight is None else np.broadcast_to(weight,r.shape).copy()
    if mask is not None:w=w*np.broadcast_to(mask,r.shape)
    den=w.sum()
    if den==0:raise ValueError('Empty evaluation mask')
    d=p-r
    return dict(relative_l2=float(np.sqrt(np.sum(w*d*d)/max(np.sum(w*r*r),1e-30))),
      rmse=float(np.sqrt(np.sum(w*d*d)/den)),max_absolute=float(np.max(abs(d)[w>0])),
      reference_rms=float(np.sqrt(np.sum(w*r*r)/den)),count=int(np.count_nonzero(w)))
def time_weights(t):
    w=np.empty_like(t);w[1:-1]=(t[2:]-t[:-2])/2;w[0]=(t[1]-t[0])/2;w[-1]=(t[-1]-t[-2])/2
    return w
def map_field(y,x,t,xnew,tnew):
    yt=PchipInterpolator(t,y,axis=-1)(tnew)
    shape=yt.shape[:-2];flat=yt.reshape(-1,len(x),len(tnew))
    out=np.array([[np.interp(xnew,x,z[:,j]) for j in range(len(tnew))] for z in flat]).transpose(0,2,1)
    return out.reshape(shape+(len(xnew),len(tnew)))

def cell_average_map(y,x,t,xnew,tnew,length,area=None):
    """Conservatively average uniform FV cells onto another uniform partition."""
    yt=PchipInterpolator(t,y,axis=-1)(tnew)
    ar=np.ones((len(x),len(tnew))) if area is None else PchipInterpolator(t,area,axis=-1)(tnew)
    ef=np.linspace(0,length,len(x)+1);ec=np.linspace(0,length,len(xnew)+1)
    overlap=np.maximum(0.,np.minimum(ec[1:,None],ef[None,1:])-np.maximum(ec[:-1,None],ef[None,:-1]))
    return np.einsum('ij,...jt,jt->...it',overlap,yt,ar)/(overlap@ar)
