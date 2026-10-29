"""Finish verification, evaluation and reporting after the training driver exits successfully."""
import time,sys,subprocess,json
from .protocol import RUN
def main():
 required=[RUN/'single'/f'train-{s}/reaction.pt' for s in [42,43,44]]+[RUN/n/f'train-{s}/model.pt' for n in ['three','seven'] for s in [42,43,44]]
 while not all(p.exists() for p in required):time.sleep(10)
 for module in ['gp52.unified.checks','gp52.unified.evaluate_all','gp52.unified.report','gp52.unified.viewer']:
  print('START',module,flush=True)
  subprocess.run([sys.executable,'-m',module],check=True)
  print('DONE',module,flush=True)
 (RUN/'complete.json').write_text(json.dumps(dict(status='complete')))
if __name__=='__main__':main()
