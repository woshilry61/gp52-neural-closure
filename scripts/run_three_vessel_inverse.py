"""Run the additive three-vessel sparse inverse experiment."""
from pathlib import Path
import os,sys,subprocess,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/inverse-three'
def run(module,*args):
 subprocess.run([sys.executable,'-m',module,*args],cwd=ROOT,env=ENV,check=True)
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True)
 p=OUT/'protected-before.json'
 if not p.exists():
  sources=[ROOT/f'src/gp52/three/{name}.py' for name in ['__init__','sen','reference','closure','evaluate','checks','report']]
  files=sources+list((ROOT/'reports/three-vessel').glob('*'))+[ROOT/f'runs/closure-final-{s}/model.pt' for s in [42,43,44]]
  p.write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()},indent=2))
 ENV=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
 run('gp52.three.inverse','--stage','verify')
 for window in ['0.82','1.23']:
  run('gp52.three.inverse','--stage','validation','--window',window)
  run('gp52.three.inverse','--stage','test','--window',window)
 run('gp52.three.inverse_diagnostics')
 run('gp52.three.inverse_precision')
 ENV.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 for window in ['0.82','1.23']:
  run('gp52.three.inverse_replay','--window',window,'--precision','--workers','4')
 run('gp52.three.inverse_report','--common')
 run('gp52.three.inverse_tuned')
 for window in ['0.82','1.23']:
  run('gp52.three.inverse_replay','--window',window,'--tuned','--workers','4')
 run('gp52.three.inverse_report')
 run('gp52.three.unified_report')
