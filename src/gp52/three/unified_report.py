"""Recompute the agreed primary metrics from frozen three-vessel inverse runs.

No optimizer, neural inference or training is invoked. Old results stay intact.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SINGLE = ROOT.parent / "gp52-neural-closure"
INPUT = ROOT / "runs/inverse-three/long-window/tuned"
OUTPUT = ROOT / "reports/three-vessel-inverse-unified-v2"
AUDIT = ROOT / "runs/inverse-three/unified-summary-v2"
METHODS = ("ordinary_1d", "neural_42", "neural_43", "neural_44")
METRICS = ("inlet_0_082", "fn_0_123")
KNOTS = np.array([0, .06, .14, .24, .40, .60, .82])
CASE_IDS = tuple(range(32, 48))
PROTOCOL = {
    "version": "three-vessel-unified-report-v2",
    "baseline_id": "three-c0-fv-v1",
    "observation_end_s": 1.23,
    "inlet_score_interval_s": [0, .82],
    "fn_score_interval_s": [0, 1.23],
    "observation_species": ["Th", "Fg"],
    "observations_per_species_per_site": 24,
    "sensors_m": [.03, .01, .017],
    "test_case_ids": list(CASE_IDS),
    "noise_absolute_std": .02,
    "noisy_realizations": 3,
    "neural_seeds": [42, 43, 44],
    "neural_aggregation": "Average independently evaluated errors; not an ensemble",
    "time_metric": "Equal time-sample weights on the existing uniform 0.002 s grid",
    "spatial_metric": "Cell-volume-weighted Fn over the entire three-vessel network",
    "training_performed": False,
    "inference_or_inverse_optimization_performed": False,
    "validation_retuning_performed": False,
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(handle.read()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def csv_write(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def protected_snapshot():
    """Verify the existing method lock, then snapshot all reused/archived inputs."""
    lock_path = ROOT / "benchmarks/three-vessel-method-lock-v1.json"
    lock = json.loads(lock_path.read_text())
    assert lock["current_three_vessel"]["baseline_id"] == PROTOCOL["baseline_id"]
    paths = set()
    for label, root in (("current_three_vessel", ROOT), ("published_single_vessel", SINGLE)):
        for name, expected in lock[label]["sha256"].items():
            path = root / name
            assert digest(path) == expected, f"Changed locked file: {path}"
            paths.add(path)
    old = json.loads((ROOT / "runs/inverse-three/protected-before.json").read_text())
    for name, expected in old.items():
        path = ROOT / name
        assert digest(path) == expected, f"Changed protected forward file: {path}"
        paths.add(path)
    for name in ("three-vessel", "three-vessel-inverse", "three-vessel-inverse-common"):
        paths.update(p for p in (ROOT / "reports" / name).rglob("*") if p.is_file())
    paths.update(INPUT.glob("*.npz"))
    paths.update(INPUT.glob("*.json"))
    paths.add(lock_path)
    return {str(p): digest(p) for p in sorted(paths)}


def archived_rows():
    with (ROOT / "reports/three-vessel-inverse/per-case.csv").open() as handle:
        return {
            (row["method"], int(row["case_id"]), int(row["noise_rep"]), int(row["sensors"])): row
            for row in csv.DictReader(handle)
            if float(row["observation_end"]) == 1.23
        }


def recompute_rows():
    selection = json.loads((INPUT / "selection.json").read_text())
    with np.load(INPUT / "observations.npz") as data:
        observed_t = data["sample_t"]
        assert observed_t.shape == (24,) and np.isclose(observed_t[0], .002)
        assert np.isclose(observed_t[-1], 1.23)
        assert np.array_equal(data["sensors"], np.array(PROTOCOL["sensors_m"]))
        assert data["observations"].shape == (48, 4, 2, 3, 24)
    with np.load(ROOT / "runs/reference-final/training-grid.npz") as raw:
        truth = raw["y"][:, 2]
        true_amplitudes = raw["amplitudes"]
        time = raw["t"]
    assert np.isclose(time[0], 0) and np.isclose(time[-1], 1.23)
    assert np.allclose(np.diff(time), .002, rtol=0, atol=1e-14)
    inlet_time = time[time <= .82 + 1e-12]
    basis = np.stack([np.interp(inlet_time, KNOTS, unit) for unit in np.eye(7)])
    old = archived_rows()
    expected = {(c, rep, n) for c in CASE_IDS for rep in range(4) for n in (1, 2, 3)}
    rows = []
    max_legacy_difference = {metric: 0. for metric in METRICS}
    shared_volume = None
    for method in METHODS:
        fit_path = INPUT / f"test-{method}.npz"
        with np.load(fit_path) as saved:
            fit = {key: saved[key] for key in saved.files}
        with np.load(INPUT / f"prediction-{method}.npz") as saved:
            prediction = saved["y"][:, 2]
            volume = saved["volume"].T
            assert np.array_equal(saved["t"], time)
            for key in ("cases", "reps", "layouts"):
                assert np.array_equal(saved[key], fit[key])
        cases, reps, layouts = fit["cases"], fit["reps"], fit["layouts"]
        assert len(cases) == 192
        assert set(zip(cases.tolist(), reps.tolist(), layouts.tolist())) == expected
        assert prediction.shape == (192, 57, len(time))
        assert float(fit["lam"]) == selection["choices"][method]
        assert np.isfinite(prediction).all() and np.isfinite(fit["amplitudes"]).all()
        assert np.all((fit["amplitudes"] >= 0) & (fit["amplitudes"] <= 1.2))
        if shared_volume is None:
            shared_volume = volume.copy()
        else:
            assert np.array_equal(volume, shared_volume)
        true_inlet = true_amplitudes[cases] @ basis
        inlet_delta = (fit["amplitudes"] - true_amplitudes[cases]) @ basis
        inlet_error = np.sqrt(np.sum(inlet_delta**2, axis=1) / np.sum(true_inlet**2, axis=1))
        reference_fn = truth[cases]
        fn_error = np.sqrt(
            np.sum((prediction - reference_fn)**2 * volume, axis=(1, 2))
            / np.sum(reference_fn**2 * volume, axis=(1, 2))
        )
        for i, case in enumerate(cases):
            chosen = int(fit["chosen"][i])
            row = {
                "method": method,
                "baseline_id": PROTOCOL["baseline_id"] if method == "ordinary_1d" else "",
                "case_id": int(case), "noise_rep": int(reps[i]), "sensors": int(layouts[i]),
                "observation_end_s": 1.23,
                "inlet_score_end_s": .82, "fn_score_end_s": 1.23,
                "regularization": float(fit["lam"]),
                "inlet_0_082": float(inlet_error[i]), "fn_0_123": float(fn_error[i]),
                "chosen_start": chosen,
                "projected_gradient": float(fit["all_pg"][i, chosen]),
                "termination": str(fit["all_reason"][i, chosen]),
            }
            prior = old[(method, int(case), int(reps[i]), int(layouts[i]))]
            for current, previous in (("inlet_0_082", "inlet_082"), ("fn_0_123", "fn_123")):
                difference = abs(row[current] - float(prior[previous]))
                max_legacy_difference[current] = max(max_legacy_difference[current], difference)
                assert difference < 1e-12, (method, case, current, difference)
            rows.append(row)
        print(f"Recomputed {method}: {len(cases)} fits", flush=True)
    assert len(rows) == 768
    return rows, selection, max_legacy_difference


def aggregate(rows):
    summary, paired = {}, []
    bootstrap = np.random.default_rng(2026102805).integers(0, len(CASE_IDS), (10000, len(CASE_IDS)))
    for noise in ("clean", "noisy"):
        summary[noise] = {}
        for n in (1, 2, 3):
            subset = [
                r for r in rows if r["sensors"] == n
                and ((r["noise_rep"] == 0) if noise == "clean" else (r["noise_rep"] > 0))
            ]
            means = {}
            for method in (*METHODS, "neural_mean"):
                selected = [
                    r for r in subset
                    if (r["method"].startswith("neural_") if method == "neural_mean" else r["method"] == method)
                ]
                means[method] = {metric: float(np.mean([r[metric] for r in selected])) for metric in METRICS}
            summary[noise][str(n)] = means
            for metric in METRICS:
                neural = np.array([
                    np.mean([r[metric] for r in subset if r["case_id"] == case and r["method"].startswith("neural_")])
                    for case in CASE_IDS
                ])
                baseline = np.array([
                    np.mean([r[metric] for r in subset if r["case_id"] == case and r["method"] == "ordinary_1d"])
                    for case in CASE_IDS
                ])
                delta = neural - baseline
                lower, upper = np.quantile(delta[bootstrap].mean(axis=1), [.025, .975])
                paired.append({
                    "noise": noise, "sensors": n, "metric": metric,
                    "mean_neural_minus_c0": float(delta.mean()),
                    "ci95_low": float(lower), "ci95_high": float(upper),
                    "neural_better_cases": int(np.sum(delta < 0)), "independent_cases": len(CASE_IDS),
                })
    return summary, paired


def table(summary, noise):
    lines = [
        "| Sites | C0 inlet, 0–0.82 s | Neural inlet, 0–0.82 s | C0 Fn, 0–1.23 s | Neural Fn, 0–1.23 s |",
        "|---|---:|---:|---:|---:|",
    ]
    for n in (1, 2, 3):
        group = summary[noise][str(n)]
        values = [group[m][k] for k in METRICS for m in ("ordinary_1d", "neural_mean")]
        lines.append(f"| {n} | " + " | ".join(f"{100*v:.2f}%" for v in values) + " |")
    return "\n".join(lines)


def plot(summary, paired):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    for noise in ("noisy", "clean"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
        for ax, metric, title in zip(axes, METRICS, ("Inlet recovery: 0–0.82 s", "Unobserved Fn: 0–1.23 s")):
            for method, color, label in (
                ("ordinary_1d", "#2962a0", "Fixed C0"),
                ("neural_mean", "#d46628", "Frozen neural closure"),
            ):
                values = [100*summary[noise][str(n)][method][metric] for n in (1, 2, 3)]
                ax.plot((1, 2, 3), values, "o-", color=color, label=label, lw=2)
            ax.set(title=title, xlabel="Sensor sites", ylabel="Relative L2 error (%)", xticks=(1, 2, 3))
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
        axes[0].legend()
        fig.suptitle(f"Unified v2 | observations through 1.23 s | {noise} observations")
        fig.savefig(OUTPUT / f"comparison-{noise}.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, metric, title in zip(axes, METRICS, ("Inlet: 0–0.82 s", "Fn: 0–1.23 s")):
        data = [r for r in paired if r["noise"] == "noisy" and r["metric"] == metric]
        means = np.array([r["mean_neural_minus_c0"] for r in data]) * 100
        lo = np.array([r["ci95_low"] for r in data]) * 100
        hi = np.array([r["ci95_high"] for r in data]) * 100
        ax.errorbar((1, 2, 3), means, yerr=np.array([means-lo, hi-means]), fmt="o", capsize=5, color="#d46628")
        ax.axhline(0, color="#444444", ls="--", lw=1)
        ax.set(title=title, xlabel="Sensor sites", ylabel="Neural minus C0 (percentage points)", xticks=(1, 2, 3))
        ax.grid(alpha=.2)
    fig.suptitle("Noisy observations | paired 95% case-bootstrap intervals\nNegative differences favor the frozen neural model")
    fig.savefig(OUTPUT / "paired-differences.png", dpi=160)
    plt.close(fig)


def write_report(summary, paired, verification):
    ci_lines = [
        "| Sites | Inlet difference (95% CI), pp | Fn difference (95% CI), pp |",
        "|---|---:|---:|",
    ]
    supported = {}
    for n in (1, 2, 3):
        cells = []
        for metric in METRICS:
            row = next(r for r in paired if r["noise"] == "noisy" and r["sensors"] == n and r["metric"] == metric)
            cells.append(f"{100*row['mean_neural_minus_c0']:.2f} ({100*row['ci95_low']:.2f}, {100*row['ci95_high']:.2f})")
        ci_lines.append(f"| {n} | " + " | ".join(cells) + " |")
    for metric in METRICS:
        supported[metric] = [str(r["sensors"]) for r in paired if r["noise"] == "noisy" and r["metric"] == metric and r["ci95_high"] < 0]
    report = f"""# Unified three-vessel inverse report — v2

This report uses one observation window and one fixed numerical baseline.
Both candidates observe Th/Fg through **1.23 s**. Inlet recovery is scored over
**0–0.82 s**; unobserved Fn is scored over the complete **0–1.23 s** interval.
The seven inlet knots, saved inverse solutions, C0 implementation and all
neural weights are unchanged. Only metrics and presentation are recomputed.

## Fixed comparison

- Baseline: **three-c0-fv-v1**, ordinary conservative 1D finite volume, molecular
  diffusion and mean-product reaction, with no learned closure.
- Neural candidate: the frozen three-vessel transport/reaction closures from
  seeds 42, 43 and 44, using the same conservative numerical framework.
- Same blood, geometry, grid, time step, chemistry, boundaries, observations,
  noise draws, inlet parameterization, bounds and optimization budget.
- One site: parent midpoint (30 mm). Two sites: add daughter 1 midpoint
  (10 mm). Three sites: add daughter 2 midpoint (17 mm).
- 24 observations per species and site, from 0.002 through 1.23 s. Fn is never
  an inverse measurement.
- Inlet varies at seven prescribed knots through 0.82 s, then holds its last value.
- Existing validation-only choices are retained: regularization 0.0001 for C0
  and 0 for all neural seeds, from the same three-value grid. No parameter is
  selected again using these test results.

The reference resolves prescribed-flow axisymmetric reaction–transport in
three connected tubes with a perfectly mixed junction. It is a synthetic
numerical reference, not measured biochemistry or full 3D bifurcation CFD.

## Primary results: noisy observations

Each mean uses 16 test inlet cases and three independent noise realizations,
with Gaussian absolute standard deviation 0.02. Neural values average the
errors of the three frozen models; they are not ensemble predictions.

{table(summary, "noisy")}

![Primary comparison](comparison-noisy.png)

The inlet columns still assess the varying part of the input, while the Fn
columns now assess the entire computed response. The old Fn score ending
at 0.82 s must not be substituted into this table.

## Paired uncertainty

Differences are neural minus C0 in percentage points. Negative favors neural.
Noise and neural seeds are averaged within each case before resampling the
16 independent inlet cases 10,000 times.

{chr(10).join(ci_lines)}

The noisy-data inlet interval favors neural at sensor counts
{', '.join(supported['inlet_0_082']) or 'none'}; the full-window Fn interval
favors neural at sensor counts {', '.join(supported['fn_0_123']) or 'none'}.
An interval spanning zero does not establish a reliable advantage.
A lower mean does not mean every individual inlet is improved.

![Paired differences](paired-differences.png)

## Clean observations

These observations still come from the resolved reference, so candidate
forward-model discrepancies remain even without measurement noise.

{table(summary, "clean")}

![Clean comparison](comparison-clean.png)

## Metric definitions

Let c(t) be the prescribed inlet and c_hat(t) its recovered waveform. Inlet
error is sqrt(sum((c_hat-c)^2) / sum(c^2)), sampled every 0.002 s from 0 to
0.82 s inclusive. The held tail after 0.82 s does not dilute this metric.

Fn error is sqrt(sum(V*(Fn_hat-Fn_ref)^2) / sum(V*Fn_ref^2)), using all 57
physical cells and all samples from 0 through 1.23 s. V is the time-dependent
cell volume. This retains the original equal-time-sample convention; a
constant dt factor cancels in the ratio. Errors are computed per inverse
problem and then averaged, rather than pooling all fields into one ratio.

This is offline inverse reconstruction using all available measurements
through 1.23 s. It is not a causal forecast made at 0.82 s.

## Verification and preservation

- Recomputed **{verification['fits_recomputed']}** selected inverse results:
  16 inlets × 4 noise settings × 3 layouts × 4 candidates.
- Every test case, noisy replicate, model seed and selected fit is retained.
- Maximum absolute difference from the archived per-case metrics:
  inlet {verification['legacy_metric_max_absolute_difference']['inlet_0_082']:.3g},
  Fn {verification['legacy_metric_max_absolute_difference']['fn_0_123']:.3g}.
- Baseline identity and both projects' protected model/source/input hashes
  match the existing lock.
- **{verification['protected_files']}** reused or archived files have identical
  before/after SHA256 hashes. Previous reports and raw inverse runs remain intact.
- No neural training, inference, inverse optimization or validation retuning
  was invoked. Computation consists of reading arrays and recomputing statistics.
- Saved optimizer diagnostics remain visible in per-case.csv. Maximum selected
  projected gradient is {verification['max_selected_projected_gradient']:.3g};
  {verification['selected_over_1e8']} exceed the strict 1e-8 target and
  {verification['selected_over_1e6']} exceed 1e-6. No such case is removed.

## What this update establishes

The reporting convention is now consistent across one, two and three sites.
It does not change the model or resolve the previously identified model
discrepancy and regularization limitations. C0 is the fixed basic comparator,
not a claim to be the strongest calibrated numerical closure. Existing test
cases have already been inspected, so this is a re-summary, not a new blind test.
These observations and fits do not demonstrate a speed advantage.

The [v1 report](../three-vessel-inverse/REPORT.md) retains short-window and
historical scoring results. The [method lock](../../docs/FIXED_METHODS.md)
keeps the published single-vessel C1 experiment separate.

## Files and reproduction

- summary.json: clean/noisy means for C0 and every frozen neural seed.
- per-case.csv: all individual errors and original optimization diagnostics.
- paired-comparisons.csv: paired differences, uncertainty and per-case win counts.
- protocol.json: the exact reporting windows and aggregation rules.
- verification.json and manifest.json: integrity and numeric cross-checks.
- validation-selection.json: unchanged validation decisions.

From the existing the experiment host checkout:

~~~bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src \\
  .venv/bin/python -m gp52.three.unified_report
~~~

See [the v2 protocol](../../docs/THREE_VESSEL_UNIFIED_REPORT_V2.md).
"""
    (OUTPUT / "REPORT.md").write_text(report)


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = protected_snapshot()
    snapshot_path = AUDIT / "protected-before.json"
    if snapshot_path.exists():
        assert json.loads(snapshot_path.read_text()) == before, "Protected inputs changed since v2 was first run"
    else:
        json_write(snapshot_path, before)
    rows, selection, delta = recompute_rows()
    summary, paired = aggregate(rows)
    changed = [path for path, value in before.items() if digest(path) != value]
    assert not changed, changed
    verification = {
        "fits_recomputed": len(rows), "independent_test_inlets": len(CASE_IDS),
        "legacy_metric_max_absolute_difference": delta,
        "protected_files": len(before), "changed_protected_files": changed,
        "training_performed": False, "optimization_performed": False,
        "max_selected_projected_gradient": max(r["projected_gradient"] for r in rows),
        "selected_over_1e8": sum(r["projected_gradient"] > 1e-8 for r in rows),
        "selected_over_1e6": sum(r["projected_gradient"] > 1e-6 for r in rows),
    }
    csv_write(OUTPUT / "per-case.csv", rows)
    csv_write(OUTPUT / "paired-comparisons.csv", paired)
    json_write(OUTPUT / "summary.json", summary)
    json_write(OUTPUT / "protocol.json", PROTOCOL)
    json_write(OUTPUT / "validation-selection.json", selection)
    json_write(OUTPUT / "verification.json", verification)
    plot(summary, paired)
    write_report(summary, paired, verification)
    manifest = {
        "protected_files_sha256": before,
        "reporter": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
        "protocol": {"path": str(ROOT / "docs/THREE_VESSEL_UNIFIED_REPORT_V2.md"),
                     "sha256": digest(ROOT / "docs/THREE_VESSEL_UNIFIED_REPORT_V2.md")},
        "outputs_sha256": {p.name: digest(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name != "manifest.json"},
    }
    json_write(OUTPUT / "manifest.json", manifest)
    print(json.dumps({"report": str(OUTPUT), **verification, "noisy_results": summary["noisy"]}, indent=2))


if __name__ == "__main__":
    main()
