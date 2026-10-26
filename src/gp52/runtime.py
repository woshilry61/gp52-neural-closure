from pathlib import Path
import numpy as np
import torch

ASSETS = Path(__file__).resolve().parent / "assets"
ROOT = Path(__file__).resolve().parents[2]

def tensor(value, dtype=torch.float64):
    return torch.as_tensor(value, dtype=dtype, device="cuda").contiguous()

class FlowPack:
    """Fixed, reconstructed blood flow and finite-volume input arrays."""
    def __init__(self):
        with np.load(ASSETS / "flow.npz") as z:
            self.x=z["x"].copy();self.t=z["t"].copy();self.knots=z["knots"].copy()
            self.dt=float(z["dt"]);self.dx=float(z["dx"]);self.pars=z["pars"].tolist()
            for key in ["F","W0","W1","IC","basis"]:setattr(self,key,tensor(z[key]))
            self.IX=tensor(z["IX"],torch.int32)
            self.f={"xf":z["xf"].copy()}
        self.N=len(self.x);self.T=len(self.t);self.K=64
        self.tau=self.pars[3];self.decay=np.exp(-self.dt/self.tau);self.h=self.tau*(1-self.decay)
    @staticmethod
    def tensor(value,dtype=torch.float64):return tensor(value,dtype)
