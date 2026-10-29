# Reproduce three-vessel sparse inlet inversion

## Current unified reporting convention (v2)

The current main comparison uses Th/Fg observations through **1.23 s** for both
the fixed **C0** baseline and the frozen neural models. It scores **inlet recovery
over 0–0.82 s** and **Fn over 0–1.23 s**, for one, two and three sensor sites.

~~~bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m gp52.three.unified_report
~~~

This command reuses the saved long-window, validation-tuned fits and predictions.
It does not retrain, run inference, change regularization or optimize the inlet.
See [the current report](../reports/three-vessel-inverse-unified-v2/REPORT.md)
and [the fixed v2 protocol](THREE_VESSEL_UNIFIED_REPORT_V2.md).
The commands below also retain/reproduce the historical v1 experiments.


This extension keeps all final checkpoints and previous forward results unchanged.
It fits seven inlet amplitudes; it does not train a new graph network.

## Prerequisites
Run inside the separate three-vessel checkout and its isolated environment:

~~~bash
cd .
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
~~~

The environment is named **gp52-three**. Package versions are in
**requirements-three-lock.txt** (PyTorch, Triton, NumPy, SciPy, Matplotlib).
The validated GPU is an RTX 5090 with 32 GB VRAM.

First complete the pipeline in [THREE_VESSEL_README.md](THREE_VESSEL_README.md).
Required inputs:

- runs/sen-three-gpu-v2/fields.npz: selected Sen blood prediction.
- runs/reference-final/training-grid.npz: 48 resolved-reference cases; cases
  0–23 train, 24–31 validation, 32–47 test.
- runs/evaluation-{42,43,44}/*.npz: CPU forward verification fields.
- runs/closure-final-{42,43,44}/model.pt: frozen graph-closure checkpoints.

No test case trains the closure or selects regularization.

## Full command
~~~bash
python scripts/run_three_vessel_inverse.py
~~~

Raw results go to **runs/inverse-three**; reports and plots go to
**reports/three-vessel-inverse**. Existing forward files remain unchanged.
Completed fit stages reuse stored fit files. For an independent repetition, use
a separate checkout with forward prerequisites and no runs/inverse-three folder.
The runner records protected forward files before starting.

## Individual stages
~~~bash
python -m gp52.three.inverse --stage verify
python -m gp52.three.inverse --stage validation --window 0.82
python -m gp52.three.inverse --stage test --window 0.82
python -m gp52.three.inverse --stage validation --window 1.23
python -m gp52.three.inverse --stage test --window 1.23
python -m gp52.three.inverse_diagnostics
python -m gp52.three.inverse_precision
python -m gp52.three.inverse_replay --window 0.82 --precision --workers 4
python -m gp52.three.inverse_replay --window 1.23 --precision --workers 4
python -m gp52.three.inverse_report --common
python -m gp52.three.inverse_tuned
python -m gp52.three.inverse_replay --window 0.82 --tuned --workers 4
python -m gp52.three.inverse_replay --window 1.23 --tuned --workers 4
python -m gp52.three.inverse_report
~~~

The single-precision runs check numerical sensitivity and select common
regularization on validation data. Each method then selects its own best value
on the same validation grid; the common-regularization runs are retained as
secondary comparisons in the precision folders. Final tables use double-precision FV
integration and the original float32 neural head. Both candidates receive
three starts and 100 iterations per start. Validation choices are locked
before final test optimization.

Reference replays use the unchanged sparse CPU solver. Repeated candidate
evaluations during inversion use batched CUDA kernels for both models.

## Inputs and outputs
**Known inputs:** fixed blood and geometry, reaction constants and initial
states, Th/Fg readings at nested midpoint sites, observation times and noise.

**Unknown:** seven piecewise-linear Th inlet amplitudes bounded to [0,1.2],
at 0, 0.06, 0.14, 0.24, 0.40, 0.60 and 0.82 s. The inlet stays at its final
concentration after 0.82 s. Fn is never fitted.

**Outputs:** recovered inlet, Th/Fg/Fn fields, objectives and projected
gradients for every start, and recovered-input reference replays.

Historical runs include observation windows ending at 0.82 and 1.23 s,
both with 24 samples per species and site. The current v2 main table uses
only the 1.23 s observation window, scoring inlet recovery over 0–0.82 s
and Fn over 0–1.23 s. The earlier scoring/window tables are preserved in v1.
One site is the parent midpoint (30 mm); two add daughter 1 midpoint (10 mm);
three add daughter 2 midpoint (17 mm).

Each window's **tuned** folder contains the final main results:

- test-ordinary_1d.npz and test-neural_{42,43,44}.npz: all selected fits and
  all three starts, statuses and gradients.
- prediction-*.npz: fields shaped [fit, species, cell, time].
- replay/*/data.npz: resolved fields using the recovered inputs.
- selection.json: per-model validation-only regularization.
- observations.npz: shared clean and noisy readings.

There are 192 fits per model: 16 test inlets × four noise settings (clean and
three independent noisy realizations) × three layouts. Both windows and four
models produce 1,536 selected fits. Noise has Gaussian absolute standard
deviation 0.02, not 2% of each local concentration.

## Read the results
Start with [the unified v2 report](../reports/three-vessel-inverse-unified-v2/REPORT.md).
The [v1 report](../reports/three-vessel-inverse/REPORT.md) preserves the earlier
window comparisons.
**per-case.csv** retains every fit; **paired-comparisons.csv** gives
case-clustered bootstrap intervals. JSON summaries retain individual neural
seeds, clean/noisy separation and both scoring windows.

The neural mean is an average of independently evaluated seed errors, not
a three-model prediction ensemble. Lower Fn error alone does not establish
better inlet recovery; reference replays distinguish these effects.

The reference is prescribed-flow, axisymmetric transport in three connected
tubes with a perfectly mixed junction. It is not full 3D bifurcation CFD;
the biochemical concentrations are synthetic equation-based data.

The long-window result is offline reconstruction using later Th/Fg readings.
It should not be described as a causal forecast made at 0.82 s. Methods are
compared within each common acquisition window.
