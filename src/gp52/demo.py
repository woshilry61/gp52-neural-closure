import argparse,json
from pathlib import Path
import numpy as np
import torch
from .runtime import ASSETS
from .inference import get_solver
from .data import metric

def main():
    p=argparse.ArgumentParser(description="Predict a single inlet with the frozen final model.")
    p.add_argument("--output",type=Path,default=Path("outputs/demo"))
    p.add_argument("--amplitudes",nargs=7,type=float)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    z=dict(np.load(ASSETS/"example.npz"));amps=z["amps"] if a.amplitudes is None else np.array(a.amplitudes)[None]
    model=get_solver("GG");y,diag=model.solve(amps);y=y.cpu().numpy();x=model.x;t=model.t
    np.savez_compressed(a.output/"prediction.npz",concentration=y,x=x,t=t,amplitudes=amps)
    centre=len(x)//2
    np.savetxt(a.output/"midpoint.csv",np.column_stack([t,y[0,:,centre].T]),delimiter=",",header="time_s,Th,Fg,Fn",comments="")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(t,y[0,2,centre],label="Final model")
    if a.amplitudes is None:ax.plot(t,z["truth"][0,2,centre],"--",label="CFD reference")
    ax.set(xlabel="Time (s)",ylabel="Cross-sectional mean Fn",title="Midpoint fibrin concentration");ax.legend();fig.tight_layout()
    fig.savefig(a.output/"midpoint.png",dpi=160);plt.close(fig)
    result={"output":str(a.output),"concentration_shape":list(y.shape),"species":["Th","Fg","Fn"],"maximum_mass_defect":float(diag[:,0].max()),"minimum_concentration":float(y.min())}
    if a.amplitudes is None:
        error=metric(torch.as_tensor(y),torch.as_tensor(z["truth"]),torch.as_tensor(z["area"]),torch.as_tensor(t))
        result["example_primary_relative_L2"]=error.numpy().tolist()
    (a.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
