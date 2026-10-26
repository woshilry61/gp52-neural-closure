"""Extract the four raw numerical waveform locations without trained predictions."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 expected="df42a12f6cdee523e7840a1ac5042cfade3f7e61679c83d04511079612614ab2"
 if hashlib.sha256(a.source.read_bytes()).hexdigest()!=expected:raise RuntimeError("Wrong raw waveform.")
 raw=np.loadtxt(a.source)
 rows=[raw[(raw[:,6]==point)&(raw[:,0]>=4.94)&(raw[:,0]<=5.76)]for point in [1,2,3,4]]
 for row in rows[1:]:np.testing.assert_array_equal(row[:,0],rows[0][:,0])
 values=dict(x_m=np.array([0.,.03,.04,.06]),time_s=rows[0][:,0],
             area_m2=np.stack([r[:,5]for r in rows]),flow_m3_s=np.stack([r[:,4]for r in rows]),
             velocity_m_s=np.stack([r[:,3]for r in rows]))
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**values)
 print(json.dumps({"source_sha256":expected,"positions":4,"time_samples":len(rows[0])}))
if __name__=="__main__":main()
