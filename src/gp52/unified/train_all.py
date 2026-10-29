"""Run the predeclared baseline training jobs on the project GPU."""
import sys,subprocess,json,argparse
from .protocol import RUN,SEEDS,plan
def job(args,log):
 with (RUN/log).open('w') as f:
  print('START',' '.join(args),flush=True)
  subprocess.run([sys.executable,'-m',*args],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',' '.join(args),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--networks',nargs='+',choices=['single','three','seven'],default=['single','three','seven']);a=p.parse_args()
 for seed in (SEEDS if 'three' in a.networks else []):
  job(['gp52.unified.train_three','--seed',str(seed)],f'three-train-{seed}.log')
 for seed in (SEEDS if 'single' in a.networks else []):
  job(['gp52.unified.train_single','--seed',str(seed),'--stage','transport'],f'single-transport-{seed}.log')
  rows=[]
  for w in plan()['training']['single_reaction_weights']:
   job(['gp52.unified.train_single','--seed',str(seed),'--stage','reaction','--weight',str(w)],f'single-reaction-{seed}-{w:g}.log')
   p=RUN/'single'/f'train-{seed}'/f'reaction-{w:g}'/'metrics.json';r=json.loads(p.read_text());rows.append(dict(weight=w,score=r['best']))
  best=min(rows,key=lambda r:r['score']);root=RUN/'single'/f'train-{seed}'
  (root/'reaction.pt').write_bytes((root/f"reaction-{best['weight']:g}"/'best.pt').read_bytes())
  (root/'selection.json').write_text(json.dumps(dict(selected=best,records=rows,selection='Validation only'),indent=2))
 for seed in (SEEDS if 'seven' in a.networks else []):
  job(['gp52.unified.train_seven','--seed',str(seed)],f'seven-train-{seed}.log')
 (RUN/'training-complete.json').write_text(json.dumps(dict(seeds=SEEDS,status='complete',networks=a.networks)))
if __name__=='__main__':main()
