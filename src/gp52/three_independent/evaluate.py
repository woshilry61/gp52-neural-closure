"""Frozen forward and 1/2/3-site inverse, with paired blood substitution."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from gp52.three import inverse as inv
from gp52.three.reference import COUNTS,LENGTH,inlets,ROOT
from .frozen import FrozenModel
from .transfer import RUN,load_flow
def reference():
    z=dict(np.load(RUN/'chem-main/data.npz'));k=round(.002/(z['t'][1]-z['t'][0]))
    return dict(y=z['y'][...,::k],t=z['t'][::k],volume=z['volume'][::k],amplitudes=z['amplitudes'],case_ids=z['case_ids'])
def observations(m):
    path=RUN/'observations.npz'
    if path.exists():return dict(np.load(path))
    raw=reference();assert np.array_equal(raw['case_ids'],np.arange(48))
    z=np.pad(raw['y'][:,:2].transpose(0,1,3,2),((0,0),(0,0),(0,0),(0,7)))
    truth=(z.reshape(96,-1)@m.obsw.cpu().numpy()).reshape(48,2,3,24)
    rng=np.random.default_rng(2026102802);obs=np.repeat(truth[:,None],4,axis=1)
    obs[:,1:]+=rng.normal(0,.02,obs[:,1:].shape)
    np.savez_compressed(path,observations=obs,truth=truth,amplitudes=raw['amplitudes'],sample_t=m.sample_t,sensors=m.sensors)
    return dict(np.load(path))
def validate(m):
    """Check original CPU solver with explicit blood, and finite-difference stability."""
    from gp52.three import evaluate as old
    from gp52.three.closure import reaction_features
    rng=np.random.default_rng(2026102804);amps=rng.uniform(.05,1.15,(4,7))
    of=old.flow;oi=old.inlets
    old.flow=lambda *args,**kwargs:(m.blood,m.t);old.inlets=lambda:(amps,{})
    try:
        if m.seed is None:y=old.solve()[0]
        else:
            with torch.no_grad():am,ta=[z.cpu().numpy() for z in m.model.coefficients()]
            pre=old.solve(am,ta)[0];feat=reaction_features(pre,m.ar,.002)
            with torch.no_grad():fac=m.model.factor(torch.tensor(feat,device='cuda',dtype=torch.float32)).cpu().numpy()
            y=old.solve(am,ta,fac)[0]
    finally:old.flow=of;old.inlets=oi
    actual=m.predict(amps,True);err=abs(actual-y).max((0,2,3))
    assert err.max()<3e-5,err
    scale=np.ones((4,144))/12
    j1=inv.jacobian(m,amps,scale,0,step=.002);j2=inv.jacobian(m,amps,scale,0,step=.001)
    e=float(np.linalg.norm(j1-j2)/np.linalg.norm(j1));assert e<.002,e
    return dict(cpu_max_absolute_by_species=err.tolist(),jacobian_step_halving_relative_change=e)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--blood',choices=['cfd','sen'],required=True);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--stage',choices=['verify','forward','inverse','all'],default='all');a=ap.parse_args()
    out=RUN/f'evaluation-{a.blood}';out.mkdir(exist_ok=True)
    m=FrozenModel(a.seed or None,cache=RUN/f'cache-{a.blood}',blood_path=RUN/'blood-main' if a.blood=='cfd' else None)
    batch_size=json.loads((RUN/'gpu-batch-benchmark-large.json').read_text())['selected']
    def batched_predict(model,amplitudes):
        y=model.predict(amplitudes,batch_size=batch_size).astype(float)
        y[:,0]=(np.c_[np.ones(len(amplitudes)),amplitudes]@model.obs_hbasis).reshape(-1,3,24)
        return y.reshape(len(amplitudes),-1)
    inv.predict=batched_predict
    log=(out/f'gpu-{m.name}.csv').open('w');mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=log)
    try:
        if a.stage in ['verify','all']:
            proof=validate(m);(out/f'verification-{m.name}.json').write_text(json.dumps(proof,indent=2));print('VERIFIED',m.name,proof,flush=True)
        if a.stage in ['forward','all']:
            amps,_=inlets();start=time.perf_counter();y=m.predict(amps[32:48],True)
            np.savez_compressed(out/f'forward-{m.name}.npz',y=y,t=m.t,volume=m.data['volume'],case_ids=np.arange(32,48),seconds=time.perf_counter()-start)
        if a.stage in ['inverse','all']:
            d=observations(m);p=inv.problems(d,range(32,48));protocol=json.loads((RUN/'protocol.json').read_text())
            lam=protocol['inverse']['regularization'][m.name]
            inv.MAXITER=100;inv.PGTOL=1e-8
            path=out/f'test-{m.name}.npz'
            if path.exists():r=dict(np.load(path))
            else:
                r=inv.solve(m,p,lam,f'independent-{a.blood}');np.savez_compressed(path,**r)
            y=m.predict(r['amplitudes'],True,batch_size=256)
            np.savez_compressed(out/f'prediction-{m.name}.npz',y=y,t=m.t,volume=m.data['volume'],cases=r['cases'],reps=r['reps'],layouts=r['layouts'])
    finally:mon.terminate();mon.wait();log.close()
if __name__=='__main__':main()
