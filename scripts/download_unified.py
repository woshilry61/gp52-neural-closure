"""Fetch the current experiment's inputs and frozen weights with verified checksums."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,os,subprocess,sys,tarfile,urllib.request,shutil
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--mode',choices=['frozen','retrain'],default='frozen')
 p.add_argument('--networks',nargs='+',choices=['single','three','seven'],default=['single','three','seven'])
 p.add_argument('--asset-directory',type=Path,help='Use already downloaded release archives.')
 a=p.parse_args();out=a.output.resolve()
 if out.exists() and any(out.iterdir()):raise SystemExit('Choose an empty output directory.')
 out.mkdir(parents=True,exist_ok=True)
 env={**os.environ,'GP52_UNIFIED_RUN':str(out)}
 subprocess.run([sys.executable,'-m','gp52.unified.protocol'],cwd=ROOT,env=env,check=True)
 manifest=json.loads((ROOT/'data/unified-manifest.json').read_text())
 downloads=a.asset_directory or out/'.downloads';downloads.mkdir(parents=True,exist_ok=True)
 for name in a.networks:
  item=manifest['assets'][name];path=downloads/item['file']
  if not path.exists():
   print('Download',name,item['bytes'],'bytes',flush=True)
   if 'parts' in item:
    with path.with_suffix('.partial').open('wb') as joined:
     for part in item['parts']:
      chunk=downloads/part['file']
      if not chunk.exists():urllib.request.urlretrieve(part['url'],chunk)
      if sha(chunk)!=part['sha256']:raise RuntimeError('Part checksum mismatch: '+part['file'])
      with chunk.open('rb') as source:shutil.copyfileobj(source,joined,8<<20)
    path.with_suffix('.partial').rename(path)
   else:
    urllib.request.urlretrieve(item['url'],path)
  if sha(path)!=item['sha256']:raise RuntimeError('Checksum mismatch: '+str(path))
  with tarfile.open(path,'r:') as archive:
   for entry in archive:
    parts=PurePosixPath(entry.name).parts
    if entry.name.startswith('/') or '..' in parts or not entry.isfile():raise RuntimeError('Invalid archive member.')
    if a.mode=='retrain' and any(s.startswith('train-') for s in parts):continue
    target=out.joinpath(*parts);target.parent.mkdir(parents=True,exist_ok=True)
    with archive.extractfile(entry) as source:
     if target.exists():
      if hashlib.sha256(source.read()).hexdigest()!=sha(target):raise RuntimeError('Conflicting file: '+str(target))
     else:
      with target.open('wb') as destination:shutil.copyfileobj(source,destination,8<<20)
  files=json.loads((out/name/'release-files.json').read_text())['files']
  for rel,row in files.items():
   if a.mode=='retrain' and any(s.startswith('train-') for s in PurePosixPath(rel).parts):continue
   if sha(out/rel)!=row['sha256']:raise RuntimeError('Extracted-file checksum mismatch: '+rel)
  print('Verified',name,flush=True)
 (out/'restored-release.json').write_text(json.dumps({'release':manifest['version'],'mode':a.mode,'networks':a.networks},indent=2)+'\n')
 print('Set GP52_UNIFIED_RUN='+str(out)+' before running experiment commands.')
if __name__=='__main__':main()
