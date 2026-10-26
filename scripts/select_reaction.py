from pathlib import Path
import argparse,json,hashlib,shutil
p=argparse.ArgumentParser(description="Select a reaction candidate using development scores only.")
p.add_argument("--run-directory",type=Path,required=True)
p.add_argument("--kind",choices=["graph","local","polynomial"],default="graph")
a=p.parse_args()
paths=[a.run_directory/"results"/f"training-{a.kind}-{w:g}.json"for w in [.03,.3,3.]]
rows=[json.loads(p.read_text())for p in paths]
assert all(r["status"]=="completed" for r in rows)
assert all(r["selected"]["physical_rate_error"]<r["baseline_physical_rate_error"] for r in rows)
chosen=min(rows,key=lambda r:r["selected"]["score"])
src=a.run_directory/"runs"/chosen["name"]/"best.pt";dst=a.run_directory/"reaction.pt"
shutil.copyfile(src,dst)
record={"selected_trial":chosen["name"],"development_score":chosen["selected"]["score"],"sha256":hashlib.sha256(dst.read_bytes()).hexdigest(),"test_data_used":False}
(a.run_directory/"selected.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
