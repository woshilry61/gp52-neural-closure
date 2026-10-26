"""Stage one regenerated partition and verify its recorded content hash."""
from pathlib import Path
import argparse,json,hashlib,shutil
def main():
 p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--destination",type=Path,required=True);p.add_argument("--split",choices=["train","development","test"],required=True);a=p.parse_args()
 name=a.split+".npz";manifest=json.loads((a.source/"manifest.json").read_text())
 source=a.source/name
 if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest[name]["sha256"]:raise RuntimeError("Regenerated dataset hash mismatch.")
 a.destination.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,a.destination/name)
 print("Staged regenerated partition:",name)
if __name__=="__main__":main()
