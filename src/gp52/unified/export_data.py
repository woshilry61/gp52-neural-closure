"""Export a common model-independent train/validation/test array schema."""
import argparse,json,hashlib
import numpy as np
from .protocol import RUN,OUT,COUNTS,LENGTH,TRAIN,VALID,TEST,plan
from .reference import flow
from .evaluate import observations
def main():
 p=argparse.ArgumentParser();p.add_argument('network',choices=list(COUNTS));a=p.parse_args();name=a.network
 z=np.load(RUN/name/'reference/data.npz');data,t,_=flow(name,tuple(COUNTS[name]),1,.002)
 obs=observations(name);out=RUN/name/'partitions';out.mkdir(exist_ok=True)
 common=dict(t=t,volume=z['volume'][::2],x=np.concatenate([d['x'] for d in data]),vessel_ids=np.repeat(np.arange(len(COUNTS[name])),COUNTS[name]),
  face_area=np.concatenate([d['af'] for d in data],1),face_flow=np.concatenate([d['q'] for d in data],1),face_x=np.concatenate([d['xf'] for d in data]),face_vessel_ids=np.repeat(np.arange(len(COUNTS[name])),COUNTS[name]+1),
  lengths=LENGTH[name],counts=COUNTS[name],observation_times=np.array(plan()['observation_times']),sensors=np.array(plan()['networks'][name]['sensors']))
 manifest={}
 for split,ids in [('train',TRAIN),('validation',VALID),('test',TEST)]:
  fields=dict(case_ids=ids,amplitudes=z['amplitudes'][ids],y=z['y'][ids,...,::2],
   reaction=z['reaction'][ids,...,::2],flux=z['flux'][ids,::2],
   exact_observations=obs['truth'][ids],observations=obs['observations'][ids],**common)
  path=out/f'{split}.npz';np.savez_compressed(path,**fields)
  manifest[split]=dict(cases=len(ids),case_ids=ids.tolist(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
   y_shape=list(fields['y'].shape),flux_shape=list(fields['flux'].shape))
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (OUT/f'data-manifest-{name}.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(dict(network=name,partitions={k:dict(cases=v['cases'],bytes=v['bytes'],sha256=v['sha256']) for k,v in manifest.items()})),flush=True)
if __name__=='__main__':main()
