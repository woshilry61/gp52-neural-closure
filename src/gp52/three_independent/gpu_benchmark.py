"""Measure the identical assembled CFD linear system on CPU and CUDA."""
import time,json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from . import blood as b
from .transfer import RUN
def benchmark(A,rhs):
    from pathlib import Path
    sp.save_npz(RUN/'cfd-linear-system.npz',A);np.save(RUN/'cfd-linear-rhs.npy',rhs)
    cpu=[];gpu=[];reference=None
    for i in range(5):
        start=time.perf_counter();reference=spsolve(A,rhs);cpu.append(time.perf_counter()-start)
    result=dict(shape=A.shape,nnz=A.nnz,cpu_seconds=cpu,cpu_median_seconds=float(np.median(cpu[1:])))
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix
        from cupyx.scipy.sparse.linalg import spsolve as gpu_solve
        aa=csr_matrix(A);rr=cp.asarray(rhs)
        for i in range(5):
            cp.cuda.Stream.null.synchronize();start=time.perf_counter()
            x=gpu_solve(aa,rr);cp.cuda.Stream.null.synchronize();gpu.append(time.perf_counter()-start)
        xx=cp.asnumpy(x)
        result.update(gpu_seconds=gpu,gpu_median_seconds=float(np.median(gpu[1:])),
          relative_solution_difference=float(np.linalg.norm(xx-reference)/np.linalg.norm(reference)),
          gpu_relative_residual=float(np.linalg.norm(A@xx-rhs)/np.linalg.norm(rhs)))
    except Exception as e:result['gpu_error']=repr(e)
    (RUN/'cfd-cpu-gpu-benchmark.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True);raise SystemExit
def main():
    src=dict(np.load(b.ROOT/'runs/sen-input/input.npz'))
    states=[b.initialize(src,k,[24,8,14][k],10) for k in range(3)]
    b.spsolve=benchmark;b.step(states,.001/b.f.T0,.001)
if __name__=='__main__':main()
