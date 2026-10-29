"""Package independently generated inputs and unchanged checkpoints for release."""
from pathlib import Path
import argparse,hashlib,json,tarfile,io
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);manifest={'version':'v1.1.0','experiment':'unified-cfd-baselines-v1','assets':{}}
 for name in ['single','three','seven']:
  selected=[a.source/n for n in ['protocol.json','inlets.npz','blood-audit.json','reference-validation.json']]
  root=a.source/name
  selected += [root/n for n in ['blood.npz','reference/data.npz','observations.npz']]
  if name=='single':selected+=list((root/'assets').glob('*'))
  for seed in [42,43,44]:
   train=root/f'train-{seed}'
   selected += [train/n for n in (['transport.pt','reaction.pt','selection.json'] if name=='single' else ['model.pt'])]
   selected += list(train.rglob('metrics.json'))+list(train.rglob('gpu.csv'))
  selected=sorted(set(p for p in selected if p.is_file()))
  target=a.output/f'gp52-unified-{name}-v1.1.0.tar'
  entries={}
  with tarfile.open(target,'w') as archive:
   for f in selected:
    rel=f.relative_to(a.source).as_posix()
    archive.add(f,arcname=rel,recursive=False)
    entries[rel]={'bytes':f.stat().st_size,'sha256':sha(f)}
   content=(json.dumps({'network':name,'files':entries},indent=2)+'\n').encode()
   info=tarfile.TarInfo(f'{name}/release-files.json');info.size=len(content);archive.addfile(info,io.BytesIO(content))
  manifest['assets'][name]={'file':target.name,'bytes':target.stat().st_size,'sha256':sha(target),'url':'https://github.com/woshilry61/gp52-neural-closure/releases/download/v1.1.0/'+target.name}
  print(name,target.stat().st_size,flush=True)
 # Split the large seven-vessel archive for reliable transfer. Its full identity stays unchanged.
 row=manifest['assets']['seven'];row['parts']=[]
 with (a.output/row['file']).open('rb') as source:
  i=1
  while True:
   b=source.read(128<<20)
   if not b:break
   name=row['file']+f'.part{i:02d}';part=a.output/name;part.write_bytes(b)
   row['parts'].append(dict(file=name,bytes=len(b),sha256=sha(part),url='https://github.com/woshilry61/gp52-neural-closure/releases/download/v1.1.0/'+name));i+=1
 row.pop('url')
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
