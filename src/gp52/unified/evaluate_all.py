"""Evaluate frozen final checkpoints under the common protocol."""
import subprocess,sys,json,argparse
from .protocol import RUN,SEEDS
def job(module,args,log):
 with (RUN/log).open('w') as f:
  print('START',module,*args,flush=True)
  subprocess.run([sys.executable,'-m',module,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',module,*args,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--networks',nargs='+',choices=['single','three','seven'],default=['single','three','seven']);p.add_argument('--include-legacy',action='store_true');a=p.parse_args()
 for name in a.networks:
  assert (RUN/name/'train-44/model.pt').exists() if name!='single' else (RUN/name/'train-44/reaction.pt').exists()
  methods=[('C0',42)]+[('new',s) for s in SEEDS]
  # Complete all validation selections before any test score is computed.
  for kind,seed in methods:
   args=[name,'--kind',kind,'--seed',str(seed)]
   job('gp52.unified.verify_models',args,f'{name}-verify-{kind}-{seed}.log')
   job('gp52.unified.evaluate',args+['--stage','validation'],f'{name}-validation-{kind}-{seed}.log')
  for kind,seed in methods:
   args=[name,'--kind',kind,'--seed',str(seed)]
   for stage in ['forward','test']:job('gp52.unified.evaluate',args+['--stage',stage],f'{name}-{stage}-{kind}-{seed}.log')
  for seed in (([42] if name=='single' else SEEDS if name=='three' else []) if a.include_legacy else []):
   job('gp52.unified.evaluate',[name,'--kind','old','--seed',str(seed),'--stage','forward'],f'{name}-forward-old-{seed}.log')
  (RUN/name/'evaluation-complete.json').write_text(json.dumps(dict(status='complete')))
 if all((RUN/n/'evaluation-complete.json').exists() for n in ['single','three','seven']):
  (RUN/'evaluation-complete.json').write_text(json.dumps(dict(status='complete',networks=['single','three','seven'])))
if __name__=='__main__':main()
