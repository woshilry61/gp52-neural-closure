from pathlib import Path
import argparse,hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split",choices=["train","development","test","reference-forward","all"],default="test")
    p.add_argument("--destination",type=Path,default=ROOT/"data")
    a=p.parse_args();a.destination.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/"data/manifest.json").read_text())
    names=list(manifest) if a.split=="all" else [a.split+".npz"]
    for name in names:
        target=a.destination/name;record=manifest[name]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==record["sha256"]:
            print("Verified",name);continue
        url="https://github.com/woshilry61/gp52-neural-closure/releases/download/v1.0.0/"+name
        temporary=target.with_suffix(".download")
        urllib.request.urlretrieve(url,temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=record["sha256"]:raise RuntimeError("Checksum mismatch: "+name)
        temporary.replace(target);print("Downloaded and verified",name)
if __name__=="__main__":main()
