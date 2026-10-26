"""Reproduce the final biochemical project from a pinned raw upstream waveform."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=Path("runs/raw-reproduction"));a=p.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/"logs").mkdir()
 env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","PYTHONUNBUFFERED":"1","PYTHONPATH":str(ROOT/"src")}
 start=time.perf_counter()
 record={"status":"running","stages":[],"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
         "input":"Pinned upstream sim_1_1.his, declared geometry/physics, split seeds and frozen numerical configuration"}
 def save():(out/"pipeline.json").write_text(json.dumps(record,indent=2)+"\n")
 def run(name,args):
  row={"name":name,"arguments":[str(x).replace(str(out),"<run>").replace(str(ROOT),"<repository>")for x in args]};record["stages"].append(row);save()
  print("START",name,flush=True);tick=time.perf_counter()
  with (out/"logs"/(name+".log")).open("w")as log:
   proc=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  row.update(seconds=time.perf_counter()-tick,returncode=proc.returncode);save();print("DONE",name,row["seconds"],proc.returncode,flush=True)
  if proc.returncode:raise RuntimeError(name+" failed; inspect its stage log")
 save()
 try:
  upstream=out/"upstream/sen2024";waveform=out/"raw/waveform.npz"
  run("fetch-sen",["scripts/fetch_sen.py","--destination",upstream])
  run("extract",["-m","gp52.raw.extract","--source",upstream/"PIGNN_1_artery/Snaps/sim_1_1.his","--output",waveform])
  run("cfd-flow",["-m","gp52.raw.cfd_flow","--source",waveform,"--output",out/"cfd/initial","--nx","32","--nr","24","--dt","0.001","--duration","0.82"])
  run("extend-flow",["-m","gp52.raw.extend_flow","--source",out/"cfd/initial","--waveform",waveform,"--output",out/"cfd/extended","--duration","1.23"])
  run("blood-observations",["-m","gp52.raw.observations","--flow",out/"cfd/extended/flow.npz","--output",out/"blood-observations"])
  with ThreadPoolExecutor(max_workers=2)as pool:
   blood=pool.submit(run,"train-blood",["-m","gp52.raw.train_blood","--data",out/"blood-observations","--output",out/"blood","--adam","20000","--lbfgs","15000"])
   chemical=pool.submit(run,"cfd-labels",["-m","gp52.raw.generate_labels","--flow",out/"cfd/extended","--output",out/"chemistry"])
   blood.result();chemical.result()
  run("prepare-assets",["-m","gp52.raw.prepare_assets","--blood",out/"blood","--output",out/"assets"])
  run("export-data",["-m","gp52.raw.export_data","--references",out/"chemistry","--assets",out/"assets","--output",out/"data"])
  run("biochemical-retraining",["scripts/retrain_and_evaluate.py","--output",out/"biochemical","--prepared-data",out/"data","--prepared-assets",out/"assets"])
  run("validate",["scripts/validate_raw_reproduction.py","--run-directory",out])
  record.update(status="completed",seconds=time.perf_counter()-start,validation=json.loads((out/"validation.json").read_text()))
  save();print(json.dumps({"status":"completed","seconds":record["seconds"],"passed":record["validation"]["passed"]}),flush=True)
 except Exception as e:
  record.update(status="failed",seconds=time.perf_counter()-start,error=str(e));save();raise
if __name__=="__main__":main()
