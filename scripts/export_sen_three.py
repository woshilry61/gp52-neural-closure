"""Export the unmodified Sen three-vessel forward/loss and gradients for port verification."""
from pathlib import Path
import os,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/sen-input';OUT.mkdir(exist_ok=True)
os.chdir(ROOT/'external/sen2024/PIGNN_3_artery')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.random.set_seed(42);np.random.seed(42)
source=Path('GNN.py').read_text()
g={'__name__':'sen_export'}
exec(compile(source.split('\nfit_model(')[0],'GNN.py','exec'),g)
model=g['model']; ds=g['dataset'][0]; lossfn=g['customloss']()
x=tf.convert_to_tensor(ds.x);adj=tf.convert_to_tensor(ds.a.toarray() if hasattr(ds.a,'toarray') else ds.a)
target=tf.convert_to_tensor(ds.y)
with tf.GradientTape() as tape:
 out=model([x,adj],training=False)
 loss=lossfn(target,out)
grad=tape.gradient(loss,model.trainable_weights)
items=dict(features=x.numpy(),graph=adj.numpy(),labels=target.numpy(),output=out.numpy(),loss=loss.numpy(),
 t=g['t'][:,0]-g['t'][0,0],dt=g['dt'],length=np.array([.06,.02,.034]),
 a0=np.stack([g[f'A0_{i}'].numpy()[:,0] for i in [1,2,3]]),
 beta=np.stack([g[f'Beta_{i}'].numpy()[:,0] for i in [1,2,3]]),
 M=np.array([g[f'M_{i}'].numpy() for i in [1,2,3]]),
 M2=np.array([g[f'M2_{i}'].numpy() for i in [1,2,3]]),
 U=np.array([g[f'U_{i}'].numpy() for i in [1,2,3]]),
 p0=np.array([g[f'p0_{i}'].numpy() for i in [1,2,3]]))
for i,w in enumerate(model.get_weights()):items[f'w{i}']=w
for i,v in enumerate(grad):items[f'g{i}']=v.numpy()
for v in [1,2,3]:
 raw=np.loadtxt(f'Snaps/sim_1_{v}.his')
 raw=raw[(raw[:,0]>=4.94)&(raw[:,0]<=5.76)]
 for s in np.unique(raw[:,6]).astype(int):
  sub=raw[raw[:,6]==s][::2]
  items[f'ref{v}_{s}']=sub
np.savez_compressed(OUT/'input.npz',**items)
(OUT/'manifest.json').write_text(json.dumps(dict(loss=float(loss),weights=len(grad),parameters=model.count_params(),N=30,T=len(items['t']),source_commit='bf0ec89f0aab8a639bdd2d363dc3c0cfae770c1d'),indent=2))
print((OUT/'manifest.json').read_text())
