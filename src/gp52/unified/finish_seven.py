"""Complete the seven-vessel branch after blood generation and the first evaluations."""
import time,subprocess,sys,json
from .protocol import ROOT,RUN,SEEDS
def job(module,args,log):
 with (RUN/log).open('w') as f:
  print('START',module,*args,flush=True)
  subprocess.run([sys.executable,'-m',module,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',module,*args,flush=True)
def main():
 required=[RUN/'seven/blood/metrics.json']
 while not all(p.exists() for p in required):time.sleep(10)
 job('gp52.unified.blood_audit',[],'blood-audit.log')
 for label,args in [('reference',[]),('uniform',['--ids','0','--uniform']),('refined',['--ids','0','31','63','--refine','8','--radial','48'])]:
  target=RUN/'seven'/label
  if not (target/'metrics.json').exists():
   job('gp52.unified.reference',['seven','--label',label,*args],f'seven-{label}.log')
 job('gp52.unified.checks',[],'reference-checks.log')
 job('gp52.unified.export_data',['seven'],'seven-export.log')
 # Reference generation uses relatively small CUDA state arrays and may overlap the earlier inference/CPU output stages.
 # Keep the large neural training job exclusive.
 while not all((RUN/n/'evaluation-complete.json').exists() for n in ['single','three']):time.sleep(10)
 for seed in SEEDS:
  if not (RUN/'seven'/f'train-{seed}/model.pt').exists():
   job('gp52.unified.train_seven',['--seed',str(seed)],f'seven-train-{seed}.log')
 (RUN/'seven/training-complete.json').write_text(json.dumps(dict(seeds=SEEDS)))
 job('gp52.unified.evaluate_all',['--networks','seven'],'evaluation-seven.log')
 job('gp52.unified.report',[],'report.log')
 job('gp52.unified.viewer',[],'viewer.log')
 (RUN/'all-complete.json').write_text(json.dumps(dict(networks=['single','three','seven'],status='complete')))
if __name__=='__main__':main()
