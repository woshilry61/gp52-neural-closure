"""Train fresh biochemical heads, select on development data, then test them."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/from-scratch"))
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    data = out / "data"
    data.mkdir()
    manifest = json.loads((ROOT / "benchmarks/artifact-manifest.json").read_text())
    expected = {name: row["sha256"] for name, row in manifest["assets"].items()}
    def verify_assets():
        if any(sha(ROOT / name) != value for name, value in expected.items()):
            raise RuntimeError("A distributed asset has changed.")
    verify_assets()
    env = {**os.environ, "GP52_DATA_DIR": str(data), "OPENBLAS_NUM_THREADS": "4",
           "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": str(ROOT / "src")}
    start = time.perf_counter()
    record = {"status": "running", "stages": [],
              "scope": "Biochemical retraining from released fixed flow and CFD labels",
              "original_assets_sha256": expected,
              "source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_files_sha256": {
                  str(f.relative_to(ROOT)): sha(f)
                  for folder in ["src/gp52", "scripts"]
                  for f in sorted((ROOT / folder).glob("*.py"))},
              "acceptance_absolute_relative_L2_tolerance": 0.0005,
              "test_data_downloaded_after_selection": False}
    def save():
        (out / "pipeline.json").write_text(json.dumps(record, indent=2) + "\n")
    def run(name, arguments):
        step = {"name": name, "arguments": [
            str(x).replace(str(out), "<run>").replace(str(ROOT), "<repository>")
            for x in arguments]}
        record["stages"].append(step)
        save()
        print("START", name, flush=True)
        started = time.perf_counter()
        with (out / "logs" / (name + ".log")).open("w") as log:
            proc = subprocess.run([sys.executable, *map(str, arguments)],
                                  cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        step.update(seconds=time.perf_counter() - started, returncode=proc.returncode)
        save()
        if proc.returncode:
            raise RuntimeError(f"{name} failed; see {out / 'logs' / (name + '.log')}")
        print("DONE", name, round(step["seconds"], 2), "s", flush=True)

    save()
    try:
        for split in ["train", "development"]:
            run("download-" + split, ["scripts/download_data.py", "--split", split,
                                      "--destination", data])
        import torch
        record["environment"] = {"python": sys.version.split()[0],
                                 "torch": torch.__version__,
                                 "cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(0)}
        transport = out / "transport"
        reaction = out / "reaction"
        run("train-transport", ["-m", "gp52.train_transport", "--kind", "graph",
                               "--steps", "1500", "--output", transport])
        checkpoint = transport / "runs/graph/best.pt"
        for weight in ["0.03", "0.3", "3"]:
            run("train-reaction-" + weight, ["-m", "gp52.train_reaction",
                "--kind", "graph", "--weight", weight, "--steps", "500",
                "--source-steps", "1000", "--transport-weights", checkpoint,
                "--output", reaction])
        run("select-reaction", ["scripts/select_reaction.py", "--run-directory", reaction,
                                "--kind", "graph"])
        verify_assets()
        if (data / "test.npz").exists():
            raise RuntimeError("Test data appeared before model selection.")
        record["selected_weights"] = {
            "transport": {"path": str(checkpoint.relative_to(out)), "sha256": sha(checkpoint)},
            "reaction": {"path": "reaction/reaction.pt", "sha256": sha(reaction / "reaction.pt")}}
        record["selection"] = json.loads((reaction / "selected.json").read_text())
        record["test_data_downloaded_after_selection"] = True
        save()
        run("download-test", ["scripts/download_data.py", "--split", "test",
                              "--destination", data])
        run("evaluate", ["-m", "gp52.evaluate", "--methods", "C1", "GG", "retrained",
                         "--transport-weights", checkpoint,
                         "--reaction-weights", reaction / "reaction.pt",
                         "--inverse", "--verify-reference",
                         "--output", out / "evaluation"])
        run("compare", ["scripts/compare_retraining.py", "--run-directory", out])
        verify_assets()
        record.update(status="completed", seconds=time.perf_counter() - start,
                      distributed_assets_unchanged=True,
                      comparison=json.loads((out / "comparison.json").read_text()))
        save()
        print(json.dumps({"status": record["status"],
                          "agreement": record["comparison"]["within_tolerance"],
                          "seconds": record["seconds"]}), flush=True)
    except Exception as error:
        record.update(status="failed", error=str(error), seconds=time.perf_counter() - start)
        save()
        raise

if __name__ == "__main__":
    main()
