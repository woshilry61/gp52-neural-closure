# Reproducing the current unified experiment

## Which version is this?

This guide covers the independent-CFD, fixed-C0 experiment for single, three and seven vessels. The 16/8/8 data split and C1 comparison described in the older guides belong to the historical single-vessel release. The current split is **64/16/32 whole inlet cases**, shared across topologies.

The current biochemical networks are trained separately per topology. Sen's original blood-flow model is retained as an upstream reference and historical reproduction, but **Sen-predicted blood does not generate the current biochemical labels**.

## Environment and outputs

Install the package and requirements-raw.txt using the commands in the root README. Use Linux/CUDA and a 32 GB GPU for the recorded training setup. Blood finite-element solves use CPU SciPy/scikit-fem; the resolved chemical generator, training and batched candidate rollouts use CUDA.

Set an absolute run directory before importing the modules:

~~~bash
export GP52_UNIFIED_RUN="$PWD/runs/my-experiment"
~~~

Reports default to `$GP52_UNIFIED_RUN/report`. Override with GP52_UNIFIED_REPORT if desired. The generators and training stages require fresh target folders. Do not rerun generation or training into a completed experiment.

## Route A: frozen-model replay

~~~bash
python scripts/download_unified.py --output runs/unified-replay
export GP52_UNIFIED_RUN="$PWD/runs/unified-replay"
python -m gp52.unified.run --stage evaluate
python -m gp52.unified.paper_figures
~~~

The seven-vessel archive is downloaded in 128 MiB parts and automatically reassembled. Each part and the complete archive are verified. Downloads are checked against SHA-256 identities, and extracted members are checked individually. The three archives contain:

- Canonical independent CFD blood fields and all 112 biochemical reference trajectories.
- Fine-grid-derived observation values and the fixed paired noise realizations.
- Shared inlet amplitudes, partitions and protocol.
- Selected checkpoint tensors for seeds 42, 43 and 44.
- Single-vessel prepared numerical operators and reference/training audit records.

The original waveform snapshots and upstream source are excluded.

The evaluation stage checks numerical implementations, selects inverse regularization on the validation cases, then runs known-inlet prediction and all clean/noisy 1/2/3-site test fits. It writes prediction arrays, recovered amplitudes, diagnostic arrays, CSV metrics and reports.

The default current replay compares C0 with the three new neural seeds. Historical old-weight replay rows remain in the archived report but are not required for the current C0 comparison. Optional --include-legacy requires the historical three-vessel checkpoint folders as well.

For individual commands:

~~~bash
python -m gp52.unified.evaluate seven --kind C0 --stage forward
python -m gp52.unified.evaluate seven --kind new --seed 42 --stage validation
python -m gp52.unified.evaluate seven --kind new --seed 42 --stage test
~~~

Run validation before test for each method and seed, so its regularization choice exists.

## Route B: fresh biochemical training on released CFD data

~~~bash
python scripts/download_unified.py --output runs/unified-retrained --mode retrain
export GP52_UNIFIED_RUN="$PWD/runs/unified-retrained"
python -m gp52.unified.run --stage train
python -m gp52.unified.run --stage evaluate
~~~

The retrain mode omits every selected neural checkpoint and training directory. Prepared blood operators and generated labels are retained. It initializes and trains both modules again; it is not a warm start from the published neural weights.

Single vessel: 1,500 transport loop steps, then 1,000 source-fitting and 500 trajectory steps for each reaction loss weight, with shuffled batches of 16. Three/seven vessels: 3,000 main steps per head with all 64 cases. The original warm-up updates and validation-only selection rules are retained.

## Route C: raw numerical waveforms through full data generation

~~~bash
export GP52_UNIFIED_RUN="$PWD/runs/unified-from-raw"
python -m gp52.unified.run
~~~

The stages are:

1. Fetch the original Sen repository at revision bf0ec89f0aab8a639bdd2d363dc3c0cfae770c1d.
2. Extract numerical boundary/initial waveforms directly. The three-vessel CFD input extraction does not instantiate or train a TensorFlow network.
3. Compute independent compliant-wall axisymmetric ALE blood CFD.
4. Transfer blood conservatively, then generate resolved biochemical references for all inlet cases.
5. Run prescribed uniform-solution and concentration-refinement checks.
6. Train three fresh seeds per topology, select on validation and evaluate C0/neural methods.
7. Build the result report and curve viewer.

The same stages can be invoked with --stage blood, data, train or evaluate in that order.

Blood discretizations preserve the recorded experiment: single-vessel 32 axial/24 radial elements at 1 ms; three-vessel 36/12/21 axial and 15 radial elements at 0.5 ms; seven-vessel 24/8/14/16/56/14/38 axial and 10 radial elements at 1 ms. All extend to 1.23 s. Branched references couple separate axisymmetric tubes through conservative junction constraints; they are not full 3-D bifurcation CFD.

The chemistry uses four axial reference cells per candidate cell, 24 radial cells and a 1 ms reference step. Candidate concentrations use a 2 ms step. Geometry and parameters come from the declared configuration and pinned upstream numerical snapshots. Blood and chemistry are synthetic numerical data, not mouse or patient measurements.

The unified experiment was executed before publication. This repository update validates frozen replay and the portable extraction/entry points; it does not claim an additional complete from-raw CFD and training run was performed during packaging. Exact fresh-training results can depend on hardware and library versions. See [release validation](RELEASE_VALIDATION.md).

## Partitions and scoring

Case IDs: train 0–63, validation 64–79, test 80–111. Seven amplitudes lie in [0.05, 1.15] and are interpolated at [0, .06, .14, .24, .40, .60, .82] s, then held to 1.23 s. Vessels, cells and times from a case stay in one partition.

Only Th/Fg are observed, at 24 times from .002 to 1.23 s. Three paired noise realizations have absolute standard deviation .02. Each method fits seven amplitudes in [0,1.2], tries the same three starts and chooses regularization from {0, 1e-4, .01} using validation inlet error.

Inlet recovery is scored over 0–.82 s. Th/Fg/Fn use the whole network over 0–1.23 s, weighted by reference volume and trapezoidal time. Reported neural means average per-case errors over three seeds. They are not errors of averaged ensemble predictions.

The reference data.npz contains y [case, species, cell, time], reaction moments, advective covariance flux, volumes, times and amplitudes. Use gp52.unified.export_data to export train/validation/test NPZ partitions in the documented common schema.

## What this benchmark does not establish

There is one fixed blood condition per topology and no held-out geometry. The current graph advantage is not isolated by a new equal-parameter no-graph ablation. Seven-vessel blood-mesh convergence is incomplete. Runtime logs are execution records, not a matched speed comparison. Different sensor distances and blood conditions prevent interpreting cross-topology differences as the isolated effect of vessel count.
