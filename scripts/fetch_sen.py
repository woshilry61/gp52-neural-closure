"""Fetch the separately hosted upstream source; do not vendor it in this repository."""
from pathlib import Path
import argparse,hashlib,json,subprocess
URL="https://github.com/ahmetsenemse/PIGNN-1D-Blood-Flow-Simulation.git"
REVISION="bf0ec89f0aab8a639bdd2d363dc3c0cfae770c1d"
WAVEFORM="PIGNN_1_artery/Snaps/sim_1_1.his"
SHA256="df42a12f6cdee523e7840a1ac5042cfade3f7e61679c83d04511079612614ab2"
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--destination",type=Path,default=Path("external/sen2024"));a=p.parse_args()
 a.destination.parent.mkdir(parents=True,exist_ok=True)
 if not a.destination.exists():
  subprocess.run(["git","-c","credential.helper=","clone","--no-checkout",URL,str(a.destination)],check=True)
  subprocess.run(["git","checkout","--detach",REVISION],cwd=a.destination,check=True)
 actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=a.destination,text=True).strip()
 if actual!=REVISION:raise RuntimeError("The upstream checkout is not at the pinned revision.")
 waveform=a.destination/WAVEFORM
 if hashlib.sha256(waveform.read_bytes()).hexdigest()!=SHA256:raise RuntimeError("Upstream waveform checksum mismatch.")
 print(json.dumps({"repository":URL,"revision":actual,"waveform":WAVEFORM,"sha256":SHA256}))
if __name__=="__main__":main()
