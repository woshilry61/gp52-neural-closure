"""Sen-architecture conservative PIGNN with BOTH velocity boundaries unobserved.
Random initialization; training reads only sparse interior observations. No full-BC
pretrained weights, reference fields or original endpoint-training inputs are loaded.
"""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import BSpline
import sys,shutil
from .numerics import mechanics
from .optimization import capture_backward
torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision('highest')
def spline(t,k):
    knots=np.r_[np.repeat(0.,4),np.linspace(0,t[-1],k-2)[1:-1],np.repeat(t[-1],4)]
    return BSpline.design_matrix(t,knots,3).toarray()
class Model(torch.nn.Module):
    def __init__(self,cases,observations,meta,dtype=torch.float32,n=41,dt=.002):
        super().__init__();self.cases=cases;self.n=n;self.dx=.06/n
        self.t=np.linspace(0,meta[0]["duration"],round(meta[0]["duration"]/dt)+1);self.dt=float(self.t[1]);self.x=(np.arange(n)+.5)*self.dx;self.xf=np.linspace(0,.06,n+1)
        B=len(cases);gx=np.linspace(0,.06,10);k=96;hidden=16
        nobs=max(len(observations[c["name"]+"_t"])for c in cases);nloc=max(len(observations[c["name"]+"_x"])for c in cases)
        a0,beta=mechanics(self.x);a0f,bf=mechanics(self.xf)
        adj=np.eye(10,k=1)+np.eye(10,k=-1);deg=adj.sum(1);G=-adj/np.sqrt(deg[:,None]*deg[None,:])
        feat=np.column_stack([gx/.06,np.ones(10),(gx/.06)**2])
        ic=np.stack([np.interp(self.x,gx,r) for r in np.eye(10)],1)
        iff=np.stack([np.interp(self.xf,gx,r) for r in np.eye(10)],1)
        def buf(name,a):self.register_buffer(name,torch.tensor(a,device='cuda',dtype=dtype))
        buf('features',np.broadcast_to(feat,(B,10,3)).copy());buf('graph',np.broadcast_to(G,(B,10,10)).copy())
        buf('ic',np.broadcast_to(ic,(B,n,10)).copy());buf('iff',np.broadcast_to(iff,(B,n+1,10)).copy());buf('basis',spline(self.t,k).T)
        buf('a0',a0[None,:,None]);buf('a0f',a0f[None,:,None]);buf('beta',beta[None,:,None])
        arrays=[[] for _ in range(12)];uin=[];obsU=[];obsA=[];maskU=[];maskA=[];scales=[];oic=[];oif=[];oit=[]
        for ci,c in enumerate(cases):
            rng=np.random.default_rng(c['seed']);name=c['name'];info=next(v for v in meta if v['name']==name)
            for j in range(12):
                inp=3 if j<2 else hidden;out=k if j>=10 else hidden
                arrays[j].append(rng.normal(size=(inp,out))*(.25/np.sqrt(inp) if j<10 else .02))
            v=observations[name+'_u'];ar=observations[name+'_area'];ox=observations[name+'_x'];ot=observations[name+'_t']
            uin.append(np.full(k,v.mean())+.001*rng.normal(size=k))
            # Pad to shared observed-time count; both endpoint velocities remain hidden.
            ix=np.r_[ox,np.repeat(ox[-1],nloc-len(ox))];it=np.r_[ot,np.repeat(ot[-1],nobs-len(ot))]
            u=np.zeros((nloc,nobs));a=np.zeros_like(u);mask=np.zeros_like(u)
            u[:len(ox),:len(ot)]=v;a[:len(ox),:len(ot)]=ar;mask[:len(ox),:len(ot)]=1
            obsU.append(u);obsA.append(a);maskU.append(mask);maskA.append(mask*info['area_observed']);scales.append([info['u_scale'],info['area_scale']])
            oif.append(np.stack([np.interp(ix,self.xf,r) for r in np.eye(n+1)],1))
            oit.append(np.stack([np.interp(it,self.t,r) for r in np.eye(len(self.t))],1).T)
        self.ws=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(np.stack(a),device='cuda',dtype=dtype).contiguous()) for a in arrays])
        self.ucoeff=torch.nn.Parameter(torch.tensor(np.stack(uin),device='cuda',dtype=dtype))
        for name,a in [('obs_u',obsU),('obs_a',obsA),('mask_u',maskU),('mask_a',maskA),('scales',scales),('obs_if',oif),('obs_it',oit)]:buf(name,np.array(a))
    def forward(self):
        z=self.features
        for i in range(5):
            z=torch.tanh(torch.nn.functional.leaky_relu(torch.bmm(self.graph,torch.bmm(z,self.ws[2*i]))+torch.bmm(z,self.ws[2*i+1]),float(np.float32(.3))))
        coeff=torch.bmm(self.graph,torch.bmm(z,self.ws[10]))+torch.bmm(z,self.ws[11])
        lognodes=.3*torch.tanh(coeff@self.basis)
        A=self.a0*torch.exp(torch.bmm(self.ic,lognodes));Af=self.a0f*torch.exp(torch.bmm(self.iff,lognodes))
        scale=(self.a0f[:,0,0]*1.02/Af[:,0,0])[:,None,None];A=A*scale;Af=Af*scale
        vin=self.ucoeff@self.basis
        qin=Af[:,:1,1:]*vin[:,None,1:]
        delta=(A[:,:,1:]-A[:,:,:-1])/self.dt
        Q=torch.cat([qin,qin-self.dx*delta.cumsum(1)],1)
        Q0=2*Q[:,:,:1]-Q[:,:,1:2];Q0=torch.cat([Af[:,:1,:1]*vin[:,None,:1],Q0[:,1:]],1)
        Q=torch.cat([Q0,Q],2);u=Q/Af;p=9986+self.beta*(torch.sqrt(A)-torch.sqrt(self.a0))
        return A,Q,u,p,Af
    def terms(self):
        A,Q,u,p,Af=self();vs=self.scales[:,0,None,None]
        kr=-2*1.33*.0025*np.pi/(1060*(1.33-1))
        v=u[:,1:-1,1:];ux=(u[:,2:,1:]-u[:,:-2,1:])/(2*self.dx);px=(p[:,1:,1:]-p[:,:-1,1:])/self.dx
        r=(v-u[:,1:-1,:-1]+self.dt*(1.33*v*ux+px/1060-kr*v/Af[:,1:-1,1:]))/vs
        physics=100*(r*r).mean((1,2))+(r.cumsum(2)**2).mean((1,2))
        ou=torch.bmm(torch.bmm(self.obs_if,u),self.obs_it);oa=torch.bmm(torch.bmm(self.obs_if,Af),self.obs_it)
        du=((ou-self.obs_u)/vs)**2*self.mask_u
        da=((oa-self.obs_a)/self.scales[:,1,None,None])**2*self.mask_a
        data=du.sum((1,2))/self.mask_u.sum((1,2))+da.sum((1,2))/self.mask_a.sum((1,2)).clamp_min(1)
        smooth=(torch.diff(self.ucoeff,n=2,dim=1)**2).mean(1)/(self.scales[:,0]**2)
        smooth=smooth+10*(torch.diff(torch.log(A),n=2,dim=2)**2).mean((1,2))*(.02/self.dt)**4
        return physics,data,smooth
    def losses(self):
        p,d,s=self.terms();return p+10*d+.01*s
    def loss(self):return self.losses().sum()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--adam',type=int,default=20000);ap.add_argument('--lbfgs',type=int,default=6000);ap.add_argument('--seeds',type=int,default=3)
    a=ap.parse_args();a.output.mkdir(exist_ok=False);shutil.copy2(__file__,a.output/"executed-source.py")
    obs=dict(np.load(a.data/'blood-observations.npz'));meta=json.loads((a.data/'blood-observations.json').read_text())
    cases=[dict(name=x['name'],seed=s) for x in meta for s in range(42,42+a.seeds)]
    m=Model(cases,obs,meta);lr=torch.tensor(.001,device="cuda");opt=torch.optim.Adam(m.parameters(),lr=lr,fused=True,capturable=True)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            opt.zero_grad(set_to_none=True);v=m.loss();v.backward();opt.step()
    torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        opt.zero_grad(set_to_none=True);v=m.loss();v.backward();opt.step()
    gpu=(a.output/'gpu.csv').open('w');mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=gpu)
    start=time.perf_counter()
    def log(stage,it):
        with torch.no_grad():vv=m.losses().cpu().numpy()
        z=dict(stage=stage,iteration=it,seconds=time.perf_counter()-start,min_loss=float(vv.min()),max_loss=float(vv.max()))
        print(json.dumps(z),flush=True)
        with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(z)+'\n')
    try:
        for j in range(a.adam):
            if j in [a.adam//2,3*a.adam//4]:
                lr.mul_(.2)
            graph.replay()
            if (j+1)%2000==0:log('adam',j+1)
        torch.cuda.synchronize();del graph,opt
        d=Model(cases,obs,meta,dtype=torch.float64)
        with torch.no_grad():
            for dst,src in zip(d.parameters(),m.parameters()):dst.copy_(src.double())
        m=d;opt=torch.optim.LBFGS(m.parameters(),max_iter=50,max_eval=70,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-11,tolerance_change=1e-16)
        closure=capture_backward(m)
        for j in range((a.lbfgs+49)//50):
            opt.step(closure)
            if (j+1)%10==0:log('lbfgs',(j+1)*50)
        torch.save(m.state_dict(),a.output/'weights.pt')
        with torch.no_grad():
            vals=[v.cpu().numpy() for v in m()];terms=[v.cpu().numpy() for v in m.terms()];loss=m.losses().cpu().numpy()
        np.savez_compressed(a.output/'fields.npz',area=vals[0],face_flow=vals[1],velocity=vals[2],pressure=vals[3],face_area=vals[4],x=m.x,xf=m.xf,t=m.t,dx=m.dx,dt=m.dt)
        report=dict(cases=[dict(**c,loss=float(loss[i]),physics=float(terms[0][i]),data=float(terms[1][i]),regularization=float(terms[2][i])) for i,c in enumerate(cases)],
          selected={x['name']:int(min([i for i,c in enumerate(cases) if c['name']==x['name']],key=lambda i:loss[i])) for x in meta},
          seconds=time.perf_counter()-start,device=torch.cuda.get_device_name(0),initialization='Random, independent of hidden boundary waveforms',
          training_files=['blood-observations.npz','blood-observations.json'],known_initial_information='A(inlet,t=0)=1.02*A0, geometry and wall law',selection='Training physics/data objective only')
        (a.output/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:mon.terminate();mon.wait();gpu.close()
if __name__=='__main__':main()
