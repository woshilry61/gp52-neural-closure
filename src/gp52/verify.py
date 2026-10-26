from pathlib import Path
import hashlib,json
import numpy as np
import torch
from .runtime import ASSETS
from .data import data_dir,load_split
from .inference import get_solver

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path(__file__).resolve().parents[2];manifest=json.loads((root/"benchmarks/artifact-manifest.json").read_text())
    for path,row in manifest["assets"].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==row["sha256"],path
    test=load_split("test")
    with np.load(data_dir()/"reference-forward.npz")as z:archived={k:z[k].copy()for k in z.files}
    records={}
    for name in ["C1","GG","GL","LG","LL"]:
        m=get_solver(name);y,diag=m.solve(test["amps"])
        delta=float(np.max(abs(y.cpu().numpy()-archived[name])))
        assert delta<1e-11,(name,delta)
        assert float(diag[:,0].max())<1e-10 and float(y.min())>=-1e-10
        record={"archived_field_max_difference":delta,"mass_defect":float(diag[:,0].max()),"minimum":float(y.min())}
        if name!="C1":
            params=sum(v.numel()for v in m.model.parameters());assert params==9125
            record["parameters"]=params
            altered=test["amps"][:1].copy();altered[:,4:]=1.2-altered[:,4:]
            changed,_=m.solve(altered)
            prefix=float((changed-y[:1])[...,m.t<=.24].abs().max())
            assert prefix<1e-12;record["causality_max_difference"]=prefix
            # Independent differentiable time-parallel propagation with the same inputs.
            with torch.no_grad():parallel=m.model.float()(m.tensor(test["amps"][:1],torch.float32))[0]
            gap=float((parallel.double()-y[:1]).abs().max())
            assert gap<2e-5;record["parallel_vs_sequential_max_difference"]=gap
        records[name]=record
    target=Path("outputs/verification.json");target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps({"status":"passed","models":records},indent=2)+"\n")
    print(target.read_text())
if __name__=="__main__":main()
