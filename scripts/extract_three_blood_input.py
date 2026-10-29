"""Extract numerical boundary/initial data for the independent three-tube CFD."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
def extract(source,output):
 fields={};hashes={}
 for v in range(1,4):
  p=source/f'PIGNN_3_artery/Snaps/sim_1_{v}.his'
  a=np.loadtxt(p);a=a[(a[:,0]>=4.94)&(a[:,0]<=5.76)]
  for point in np.unique(a[:,6]).astype(int):
   row=a[a[:,6]==point][::2];fields[f'ref{v}_{point}']=row
  fields['t']=row[:,0]-row[0,0];hashes[str(p.relative_to(source))]=hashlib.sha256(p.read_bytes()).hexdigest()
 aa=[];bb=[]
 for i,(length,r0) in enumerate(zip([.06,.02,.034],[.018206,.015799,.008561])):
  x=np.linspace(0,length,10);r=r0+(-.0033667*x if i==0 else np.zeros_like(x))
  a0=3.14*r**2;eh=.1*(3000000*np.exp(-900*r)+489871.1841)*r
  aa.append(a0);bb.append((4/3)*1.772*eh/a0)
 fields['a0']=np.stack(aa);fields['beta']=np.stack(bb)
 output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**fields)
 output.with_suffix('.json').write_text(json.dumps({'revision':'bf0ec89f0aab8a639bdd2d363dc3c0cfae770c1d','sources':hashes,'scope':'Raw numerical snapshots and declared wall parameters. No learned blood or TensorFlow model.'},indent=2)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();extract(a.source,a.output)
if __name__=='__main__':main()
