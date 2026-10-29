"""Complete the already scheduled wall/sampling refinement checks."""
import json
from .pipeline import wait_file,run
from .transfer import RUN
def main():
 wait_file(RUN/'blood-main-sampled/metrics.json',RUN/'blood-main-sampled.log')
 run('gp52.three_independent.chemistry',['--blood',str(RUN/'blood-main-sampled'),'--output',str(RUN/'chem-sampling-refined'),'--refine','8','--radial','48','--dt','.001','--indices','0','32','47'],RUN/'chem-sampling-refined.log',RUN/'chem-sampling-refined/metrics.json')
 wait_file(RUN/'blood-wall-refined/metrics.json',RUN/'blood-wall-refined.log')
 run('gp52.three_independent.chemistry',['--blood',str(RUN/'blood-wall-refined'),'--output',str(RUN/'chem-wall-refined'),'--refine','4','--radial','24','--dt','.001','--indices','0','32','47'],RUN/'chem-wall-refined.log',RUN/'chem-wall-refined/metrics.json')
 (RUN/'extra-checks-status.json').write_text(json.dumps(dict(state='complete')))
if __name__=='__main__':main()
