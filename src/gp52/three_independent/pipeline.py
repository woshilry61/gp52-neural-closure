"""Run additive independent-CFD experiment; never train or overwrite prior results."""
from pathlib import Path
import os,json,time,subprocess,sys
from .transfer import RUN,ROOT
def wait_file(path,log):
    while not path.exists():
        text=log.read_text() if log.exists() else ''
        if 'Traceback' in text:raise RuntimeError(f'Blood process failed: {log}')
        time.sleep(15)
def run(module,args,log,done):
    if done.exists():return
    with log.open('w') as f:
        subprocess.run([sys.executable,'-m',module,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    (RUN/'pipeline-status.json').write_text(json.dumps(dict(state='waiting_for_main_blood')))
    wait_file(RUN/'blood-main/metrics.json',RUN/'blood-main.log')
    (RUN/'pipeline-status.json').write_text(json.dumps(dict(state='chemical_reference')))
    run('gp52.three_independent.chemistry',['--blood',str(RUN/'blood-main'),'--output',str(RUN/'chem-uniform'),'--refine','2','--radial','12','--dt','.001','--indices','32','--uniform'],RUN/'chem-uniform.log',RUN/'chem-uniform/metrics.json')
    run('gp52.three_independent.chemistry',['--blood',str(RUN/'blood-main'),'--output',str(RUN/'chem-main'),'--refine','4','--radial','24','--dt','.001'],RUN/'chem-main.log',RUN/'chem-main/metrics.json')
    for blood in ['cfd','sen']:
        for seed in [0,42,43,44]:
            name='ordinary_1d' if seed==0 else f'neural_{seed}'
            (RUN/'pipeline-status.json').write_text(json.dumps(dict(state='evaluation',blood=blood,method=name)))
            run('gp52.three_independent.evaluate',['--blood',blood,'--seed',str(seed)],RUN/f'evaluation-{blood}-{name}.log',RUN/f'evaluation-{blood}/prediction-{name}.npz')
    (RUN/'pipeline-status.json').write_text(json.dumps(dict(state='waiting_for_refined_blood')))
    wait_file(RUN/'blood-refined/metrics.json',RUN/'blood-refined.log')
    for tag,blood,refine,nr,dt in [('chem-refined', 'blood-main',8,48,.001),('chem-flow-refined','blood-refined',4,24,.001),('chem-time-refined','blood-refined',4,24,.0005)]:
        (RUN/'pipeline-status.json').write_text(json.dumps(dict(state='convergence',stage=tag)))
        run('gp52.three_independent.chemistry',['--blood',str(RUN/blood),'--output',str(RUN/tag),'--refine',str(refine),'--radial',str(nr),'--dt',str(dt),'--indices','0','32','47'],RUN/f'{tag}.log',RUN/tag/'metrics.json')
    (RUN/'pipeline-status.json').write_text(json.dumps(dict(state='computations_complete')))
if __name__=='__main__':main()
