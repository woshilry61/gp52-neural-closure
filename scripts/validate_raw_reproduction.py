"""Validate regenerated inputs and end-to-end results against the frozen release."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import numpy as np
from gp52.runtime import ASSETS
ROOT=Path(__file__).resolve().parents[1]

def arrays(new,reference):
 with np.load(new)as a,np.load(reference)as b:
  if set(a.files)!=set(b.files):raise RuntimeError("Array keys differ: "+new.name)
  return {k:{"max_absolute":float(np.max(np.abs(a[k]-b[k]))),
             "relative_l2":float(np.linalg.norm(a[k]-b[k])/max(np.linalg.norm(b[k]),1e-30)),
             "shape":list(a[k].shape),"dtype":str(a[k].dtype)}for k in b.files}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-directory",type=Path,required=True);a=p.parse_args();out=a.run_directory
 reference=out/"comparison-reference"
 for split in ["train","development","test"]:
  subprocess.run([sys.executable,str(ROOT/"scripts/download_data.py"),"--split",split,"--destination",str(reference)],check=True)
 data={split:arrays(out/"data"/(split+".npz"),reference/(split+".npz"))for split in ["train","development","test"]}
 prepared={name:arrays(out/"assets"/name,ASSETS/name)for name in ["flow.npz","inverse-context.npz"]}
 new=json.loads((out/"biochemical/evaluation/metrics.json").read_text())
 frozen=json.loads((ROOT/"benchmarks/retraining/evaluation.json").read_text())
 diffs={}
 for name in ["C1","GG","retrained"]:
  original="GG" if name=="retrained" else name
  for window in ["primary","full"]:
   diffs[f"forward-{name}-{window}"]=float(np.max(np.abs(np.array(new["forward"][name][window]["cases"])-np.array(frozen["forward"][original][window]["cases"]))))
  for layout in new["inverse"]:
   for window in ["primary","full"]:
    left=np.array([r[window+"_error"]for r in new["inverse"][layout][name]["cases"]])
    right=np.array([r[window+"_error"]for r in frozen["inverse"][layout][original]["cases"]])
    diffs[f"inverse-{layout}-{name}-{window}"]=float(np.max(np.abs(left-right)))
 for window in ["primary","full"]:
  diffs["reaction-rate-"+window]=float(np.max(np.abs(np.array(new["direct_reaction"]["retrained"][window])-np.array(frozen["direct_reaction"]["GG"][window]))))
 manifest=json.loads((ROOT/"benchmarks/artifact-manifest.json").read_text())
 unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v["sha256"]for n,v in manifest["assets"].items())
 blood=json.loads((out/"blood/metrics.json").read_text())
 fields=dict(np.load(out/"blood/fields.npz"));area=fields["area"];q=fields["face_flow"]
 continuity=float(np.max(np.abs(np.diff(area,axis=-1)/float(fields["dt"])+np.diff(q[:,:,1:],axis=1)/float(fields["dx"]))))
 chemistry=json.loads((out/"chemistry/generation.json").read_text())
 max_data=max(v["max_absolute"]for group in data.values()for v in group.values())
 max_input=max(v["max_absolute"]for group in prepared.values()for v in group.values())
 worst=max(diffs.values())
 snapshots={"cfd/initial/executed-source.py":"src/gp52/raw/cfd_flow.py",
            "cfd/extended/executed-source.py":"src/gp52/raw/extend_flow.py",
            "blood/executed-source.py":"src/gp52/raw/train_blood.py"}
 for f in (out/"chemistry").glob("batch-*/executed-source.py"):
  snapshots[str(f.relative_to(out))]="src/gp52/raw/cfd_chemistry.py"
 source_matches={name:(out/name).read_bytes()==(ROOT/source).read_bytes()for name,source in snapshots.items()}
 passed=unchanged and all(source_matches.values()) and max_data<=1e-8 and max_input<=1e-5 and worst<=0.0005
 gpu=np.genfromtxt(out/"blood/gpu.csv",delimiter=",",usecols=(1,2,3))
 report={"passed":bool(passed),"data_comparison":data,"prepared_inputs_comparison":prepared,"metric_differences":diffs,
         "maximum_data_difference":max_data,"maximum_prepared_input_difference":max_input,
         "maximum_metric_difference":worst,"thresholds":{"data_absolute":1e-8,"prepared_inputs_absolute":1e-5,"relative_L2_metric_absolute":0.0005},
         "original_assets_unchanged":unchanged,"blood_selected":blood["selected"],"blood_seconds":blood["seconds"],
         "blood_continuity_residual":continuity,"blood_gpu_utilization_mean":float(np.mean(np.atleast_2d(gpu)[:,0])),
         "cfd_chemistry_cases":sum(len(x["seeds"])for x in chemistry["batches"]),
         "biochemical_comparison":json.loads((out/"biochemical/comparison.json").read_text()),
         "input_provenance":"Raw waveform downloaded at the pinned upstream revision; no archived blood checkpoint or CFD label used by generation/training.",
         "fixed_configuration":"Published numerical closure coefficients, physical settings and inverse regularization are declared experiment constants; historical exploratory coefficient searches are not repeated.",
         "executed_source_snapshots_match":source_matches,
         "source_commit_note":"Pipeline source_commit is the checkout base revision; source_sha256 identifies the package and scripts used for this run.",
         "source_sha256":{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for folder in ["src/gp52","scripts"]for f in sorted((ROOT/folder).rglob("*.py"))}}
 (out/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({k:report[k]for k in ["passed","maximum_data_difference","maximum_prepared_input_difference","maximum_metric_difference","original_assets_unchanged","blood_continuity_residual"]},indent=2))
 if not passed:raise SystemExit("Raw-input reproduction did not meet the declared comparison tolerances.")
if __name__=="__main__":main()
