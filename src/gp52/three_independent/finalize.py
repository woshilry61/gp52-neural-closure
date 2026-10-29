"""Build the report only after the scheduled reference checks finish."""
import json,time,subprocess,sys
from .transfer import RUN,ROOT
def main():
 while True:
  a=json.loads((RUN/'pipeline-status.json').read_text())
  if a.get('state')=='computations_complete' and (RUN/'extra-checks-status.json').exists():break
  for name in ['pipeline.log','finish-checks.log']:
   if 'Traceback' in (RUN/name).read_text():raise RuntimeError(f'Check failed; inspect {name}')
  time.sleep(15)
 for module in ['gp52.three_independent.report','gp52.three_independent.viewer']:
  subprocess.run([sys.executable,'-m',module],cwd=ROOT,check=True)
 (RUN/'report-status.json').write_text(json.dumps(dict(state='ready_for_review')))
if __name__=='__main__':main()
