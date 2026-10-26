"""Compare a fresh training run with the frozen publication, without selection."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from gp52.provenance import state_sha256
from gp52.runtime import ASSETS

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-directory", type=Path, required=True)
    args = p.parse_args()
    root = args.run_directory
    metrics = json.loads((root / "evaluation/metrics.json").read_text())
    pipeline = json.loads((root / "pipeline.json").read_text())
    tolerance = pipeline["acceptance_absolute_relative_L2_tolerance"]
    differences = {}
    for window in ["primary", "full"]:
        before = np.array(metrics["forward"]["GG"][window]["cases"])
        after = np.array(metrics["forward"]["retrained"][window]["cases"])
        differences["forward-" + window] = float(np.max(np.abs(after - before)))
    for layout, results in metrics["inverse"].items():
        for window in ["primary", "full"]:
            before = np.array([r[window + "_error"] for r in results["GG"]["cases"]])
            after = np.array([r[window + "_error"] for r in results["retrained"]["cases"]])
            differences[f"inverse-{layout}-{window}"] = float(np.max(np.abs(after - before)))
    for window in ["primary", "full"]:
        differences["reaction-rate-" + window] = float(np.max(np.abs(
            np.array(metrics["direct_reaction"]["GG"][window]) -
            np.array(metrics["direct_reaction"]["retrained"][window]))))
    weight_comparisons = {}
    for kind, original, trained in [
        ("transport", ASSETS / "transport-graph.pt", root / "transport/runs/graph/best.pt"),
        ("reaction", ASSETS / "reaction-GG.pt", root / "reaction/reaction.pt")]:
        a = torch.load(original, map_location="cpu", weights_only=True)
        b = torch.load(trained, map_location="cpu", weights_only=True)
        if "state" in b:
            b = {k[4:]: v for k, v in b["state"].items() if k.startswith("net.")}
        loaded_key = "loaded_transport_state_sha256" if kind=="transport" else "loaded_reaction_fp32_state_sha256"
        if state_sha256(b) != metrics["model_sources"]["retrained"][loaded_key]:
            raise RuntimeError("Evaluation did not load the specified " + kind + " weights.")
        weight_comparisons[kind] = {
            "released_state_sha256": state_sha256(a), "retrained_state_sha256": state_sha256(b),
            "tensor_max_abs_difference": max(float((a[k] - b[k]).abs().max()) for k in a),
            "identical": state_sha256(a) == state_sha256(b)}
    worst = max(differences.values())
    report = {"tolerance_absolute_relative_L2": tolerance,
              "tolerance_percentage_points": tolerance * 100,
              "maximum_metric_difference": worst,
              "within_tolerance": worst <= tolerance,
              "within_frozen_evaluation_tolerance": worst <= 1e-8,
              "metric_differences": differences, "weights": weight_comparisons,
              "forward": metrics["forward"],
              "inverse": {k: {m: v["summary"] for m, v in group.items()}
                          for k, group in metrics["inverse"].items()},
              "direct_reaction": metrics["direct_reaction"],
              "note": "Comparison is reporting only; test results do not select or modify weights."}
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"maximum_metric_difference": worst,
                      "within_tolerance": report["within_tolerance"],
                      "weights_identical": {k: v["identical"] for k,v in weight_comparisons.items()}},
                     indent=2))
if __name__ == "__main__":
    main()
