"""Run the current experiment in a fresh, portable output directory."""
import argparse,os,sys,subprocess
from .protocol import ROOT,RUN
def job(module,*args):
 print('RUN',module,*args,flush=True)
 env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
 subprocess.run([sys.executable,'-m',module,*map(str,args)],cwd=ROOT,env=env,check=True)
def script(name,*args):
 env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
 subprocess.run([sys.executable,str(ROOT/'scripts'/name),*map(str,args)],cwd=ROOT,env=env,check=True)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--stage',choices=['all','blood','data','train','evaluate'],default='all');a=p.parse_args()
 job('gp52.unified.protocol')
 if a.stage in ['all','blood']:
  script('fetch_sen.py','--destination',ROOT/'external/sen2024')
  source=ROOT/'external/sen2024';waveform=RUN/'single/raw/waveform.npz'
  job('gp52.raw.extract','--source',source/'PIGNN_1_artery/Snaps/sim_1_1.his','--output',waveform)
  job('gp52.raw.cfd_flow','--source',waveform,'--output',RUN/'single/blood-initial','--nx',32,'--nr',24,'--dt',.001,'--duration',.82)
  job('gp52.raw.extend_flow','--source',RUN/'single/blood-initial','--waveform',waveform,'--output',RUN/'single/blood','--duration',1.23)
  script('extract_three_blood_input.py','--source',source,'--output',RUN/'three/raw-input.npz')
  job('gp52.three_independent.blood','--source',RUN/'three/raw-input.npz','--output',RUN/'three/blood','--nx',36,12,21,'--nr',15,'--dt',.0005,'--duration',1.23)
  job('gp52.unified.blood_network','--output',RUN/'seven/blood','--duration',1.23)
  job('gp52.unified.blood_audit')
 if a.stage in ['all','data']:
  for name in ['single','three','seven']:
   job('gp52.unified.reference',name,'--label','reference')
   job('gp52.unified.reference',name,'--label','uniform','--ids',0,'--uniform')
   job('gp52.unified.reference',name,'--label','refined','--ids',0,31,63,'--refine',8,'--radial',48)
   job('gp52.unified.export_data',name)
  job('gp52.unified.checks');job('gp52.unified.prepare')
 if a.stage in ['all','train']:job('gp52.unified.train_all')
 if a.stage in ['all','evaluate']:
  for name in ['inference_checks','check_kernel','evaluate_all','report','viewer']:job('gp52.unified.'+name)
if __name__=='__main__':main()
