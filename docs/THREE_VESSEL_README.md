# Reproducing the isolated three-vessel extension

This checkout adds a three-vessel blood/biochemical experiment. The original release
assets under src/gp52/assets are unchanged. The default original README commands
continue to describe the old single-vessel release.

Read [the protocol](THREE_VESSEL_PROTOCOL.md) and
[the results](../reports/three-vessel/REPORT.md) before interpreting the experiment.

## Environment

All work for this experiment was performed on the experiment host in:
    .

The main virtual environment is .venv, with prompt gp52-three, Python 3.10 and
PyTorch 2.10/CUDA on an RTX 5090. It is independent of prior project environments.

In a fresh checkout:
    python3.10 -m venv --prompt gp52-three .venv
    .venv/bin/pip install -e .

A separate CPU-only oracle environment is needed solely to check equivalence
against the original TensorFlow/Spektral implementation:
    python3.10 -m venv --prompt gp52-three-oracle .venv-oracle
    .venv-oracle/bin/pip install -r requirements-sen-oracle-lock.txt

No model training in the new experiment uses the oracle environment.
During the recorded run, the pre-existing Sen environment was used read-only for
this short forward/loss/gradient check. All new training used .venv and CUDA.

## Complete run

The runner downloads the original Sen source into external/sen2024 and checks out
the pinned commit. Upstream code is not redistributed or modified.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_three_vessel.py \
      --oracle-python .venv-oracle/bin/python

Use a fresh checkout/run directory: completed experiment directories are not overwritten.
The expected result is reports/three-vessel/REPORT.md, summary.json, comparison.csv and PNG plots.
Hardware/library differences may affect optimization trajectories. Compare statistical
results across all saved seeds instead of expecting byte-identical weights.

## Stages and saved evidence

1. scripts/export_sen_three.py evaluates the original TensorFlow model and exports
   features, observations, mechanics, weights, outputs, loss and all gradients.
2. gp52.three.sen checks float64 equivalence and trains 32 independent Sen models in
   a CUDA batch. Selection uses the training objective, never internal reference errors.
3. gp52.three.reference constructs a mass-consistent blood field and solves the
   cross-section-resolved three-tube transport/reaction reference.
4. gp52.three.closure trains graph transport and reaction heads separately on physical
   moments. It uses 24 training cases and 8 validation cases.
5. gp52.three.evaluate rolls out strict one-dimensional concentrations and compares
   matched-grid ordinary 1D, transport-only, reaction-only and combined neural models.
6. gp52.three.checks adds refined ordinary 1D, conventional physical dispersion probes,
   source/flux diagnostics, uniform-concentration and reference-refinement checks.
7. gp52.three.report exports all held-out results, plots and scope statements.

The blood network matches the original three-vessel script, including its original
distal adjacency shortcut. The biochemical network is a Y tree with distinct outlets.
Two hidden flux-memory states per internal edge do not constitute radial concentration bins.

## Data and interpretation

The reference is three axisymmetric transport tubes with prescribed blood velocity
profiles and a perfectly mixed junction. It is not full 3D bifurcation CFD.
There are no measured biochemical concentration labels.

Only the inlet Th control amplitudes are random. The complete Th/Fg/Fn fields are
obtained by solving transport and reaction equations. Initial Fg=1 and Fn=0.
Diffusivity and reaction kinetics are synthetic verification settings.

All three neural seeds are evaluated on the same 16 held-out inlet curves.
This is same-blood/same-geometry inlet generalization. Real-mouse blood reconstruction,
cross-geometry generalization, seven vessels and sparse three-vessel biochemical
inversion remain outside this experiment.

The neural contribution is learned closure, implemented inside a numerical
conservative solver. No general superiority over numerical methods or inference-speed
advantage over ordinary 1D is claimed.
