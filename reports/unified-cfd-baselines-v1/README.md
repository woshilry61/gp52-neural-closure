# Unified CFD baseline experiment

This additive experiment is implemented in src/gp52/unified. It retains the existing single-vessel and three-vessel neural architectures and locked C0 implementations. The seven-vessel baseline uses the three-vessel heads and training objectives with seven-vessel connectivity. All outputs are separate from the historical models and results.

## Data contract

- One shared set of 112 seven-knot Th inlet curves: 64 training, 16 validation, 32 test.
- Common amplitude range: 0.05–1.15 for every partition.
- Common knots: 0, 0.06, 0.14, 0.24, 0.40, 0.60, 0.82 seconds; hold the last amplitude to 1.23 seconds.
- A complete network trajectory is the partition unit. Vessels, spatial nodes and time samples from a case stay together.
- Independent axisymmetric CFD blood, followed by the same conservative resolved biochemical generator.
- Three concentration labels, physical reaction moments and axial advective covariance labels are saved.
- A shared 1 ms reference step and 2 ms candidate step. Axial candidate cell counts retain the existing architectures.
- A predeclared common observation/noise/optimization/scoring protocol, recorded in protocol.json.

The numerical reference is not clinical data. Each topology currently has one fixed blood condition.

## Existing CFD inputs

The experiment reuses the independently solved and verified blood archives below, without modifying them:

- Single vessel: ../gp52-raw-reproduction/runs/from-raw/cfd/extended/flow.npz
- Three vessels: runs/independent-cfd-v1/blood-refined/flow.npz

Each blood-provenance.json records the exact source file and SHA-256. The source blood-generation implementations and raw-input instructions remain in their original projects. This command sequence assumes those verified single- and three-vessel blood archives already exist. Seven-vessel blood is generated independently from the pinned upstream numerical snapshot files; it does not use a Sen neural prediction.

## Run sequence

With the required single/three blood archives and pinned upstream seven-vessel snapshot files in place, a fresh experiment can execute the entire sequence with:

    .venv/bin/python -m gp52.unified.run

The individual stages below provide the same workflow. Do not run them again over an existing completed output tree.

Run from the gp52-three-vessel repository with its isolated .venv. The recorded configuration uses an RTX 5090 with 32 GiB VRAM; single-vessel trajectory training peaks near 30 GiB. Exact library versions are listed in environment.json. Use a fresh experiment checkout/output tree for a complete rerun; generators refuse to replace an existing result directory.

    .venv/bin/python -m gp52.unified.protocol
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m gp52.unified.blood_network --output runs/unified-cfd-baselines-v1/seven/blood --duration 1.23
    .venv/bin/python -m gp52.unified.blood_audit
    .venv/bin/python -m gp52.unified.reference single --label reference
    .venv/bin/python -m gp52.unified.reference three --label reference
    .venv/bin/python -m gp52.unified.reference single --label uniform --ids 0 --uniform
    .venv/bin/python -m gp52.unified.reference three --label uniform --ids 0 --uniform
    .venv/bin/python -m gp52.unified.reference single --label refined --ids 0 31 63 --refine 8 --radial 48
    .venv/bin/python -m gp52.unified.reference three --label refined --ids 0 31 63 --refine 8 --radial 48
    .venv/bin/python -m gp52.unified.reference seven --label reference
    .venv/bin/python -m gp52.unified.reference seven --label uniform --ids 0 --uniform
    .venv/bin/python -m gp52.unified.reference seven --label refined --ids 0 31 63 --refine 8 --radial 48
    .venv/bin/python -m gp52.unified.checks
    .venv/bin/python -m gp52.unified.export_data single
    .venv/bin/python -m gp52.unified.export_data three
    .venv/bin/python -m gp52.unified.export_data seven
    .venv/bin/python -m gp52.unified.prepare
    .venv/bin/python -m gp52.unified.train_all
    .venv/bin/python -m gp52.unified.inference_checks
    .venv/bin/python -m gp52.unified.check_kernel
    .venv/bin/python -m gp52.unified.evaluate_all
    .venv/bin/python -m gp52.unified.report
    .venv/bin/python -m gp52.unified.viewer

The training and evaluation drivers cover all three topologies by default. Use --networks single three or --networks seven for a subset. Every individual training/evaluation command is also executable. The training driver retains three independent seeds and selects reaction source weights on validation only. The evaluation driver retains all regularization candidates, starts and optimizer diagnostics.

A throughput pilot used full 64-case single-vessel updates. It was preserved separately, stopped for execution-speed reasons, and never used to initialize a final model or select based on test scores. Final single-vessel updates use shuffled batches of 16, cycling through every training case in four updates.

## Frozen architecture and C0 identities

The source hashes in protocol.json identify the original neural definitions. The single-vessel architecture and three-vessel architecture remain different; they are not advertised as one universal network.

C0 means conservative one-dimensional finite volume with molecular diffusion and the uncorrected mean-product reaction. Existing temporal schemes are preserved: coupled implicit reaction for the single-vessel C0; an exact local reaction step after transport for three- and seven-vessel C0. No learned or fitted closure is added to either C0.

Final inference uses FP64 arithmetic. Sparse adjacency application evaluates the same fixed reaction head while bounding activation memory; this changes neither its weights nor graph. Numerical parity is retained in reaction-inference-parity.json and the per-model verification files.

Only inlet regularization is selected from the same validation grid for each method. Neural weights stay frozen during inversion.

## Seven-vessel extension

The topology configuration includes the pinned Sen seven-vessel tree:

- Inlet node IDs: [0, 1, 1, 2, 2, 3, 3]
- Outlet node IDs: [1, 2, 3, 4, 5, 6, 7]
- Upstream snapshot file IDs in model order: [1, 2, 3, 6, 7, 4, 5]
- Three coupled junctions and five external endpoints.

The reference generator constructs conservation constraints and mixing nodes from topology, rather than assuming a single bifurcation. Seven-vessel blood uses the same canonical archive schema: t, and per-vessel v{v}_xf, v{v}_volume, v{v}_face_area, v{v}_face_flow, v{v}_radial_axial_velocity. Arrays are time-major except the axial face coordinate.

The seven-vessel blood solver couples seven P2/P1 axisymmetric ALE Navier–Stokes tubes through three common-traction junction constraints. Per-vessel finite-element assembly and sampling run in seven CPU workers; the coupled sparse solve runs on the CPU. The resolved biochemical reference, neural training and batched inverse rollouts use CUDA. Short-run serial and parallel blood outputs were numerically identical. The full run uses 1 ms steps, 10 radial finite elements, and axial element counts [24,8,14,16,56,14,38]. This is a coupled axisymmetric network reference, not a full three-dimensional bifurcation geometry. The report is generated only after all three topologies finish evaluation.

## Outputs

Numerical data and weights: runs/unified-cfd-baselines-v1/

Human-readable results: reports/unified-cfd-baselines-v1/

- REPORT.md and report.html: complete English report.
- SUMMARY_ZH.md and summary-zh.html: Chinese reading guide.
- curves.html: all test cases, forward and sparse-inverse concentration curves.
- summary.csv, per-case-errors.csv, paired.csv: numerical comparisons.
- optimizer-summary.csv, inverse-selection.csv: convergence and selection.
- training-runs.csv: measured duration and GPU utilization.
- protocol.json, manifest.json, integrity.json: protocol, identities and preservation.

The test set is held out during this run. After its results become visible, future model development should avoid repeated test-driven tuning; a later final blind assessment should be separately predeclared.

## Partition array schema

Each topology exports train.npz, validation.npz and test.npz under its partitions directory. All use numeric arrays without pickle:

| Array | Shape / meaning |
|---|---|
| case_ids, amplitudes | Shared case ID and seven inlet parameters |
| y | Case × 3 species × axial cell × time |
| reaction | Case × cell × time, physical mean reaction rate |
| flux | Case × time × interior axial face, advective covariance label |
| t, volume, x, vessel_ids | Scoring time grid, reference cell volumes and geometry |
| face_area, face_flow | Common candidate blood arrays, time × vessel face |
| face_x, face_vessel_ids | Coordinates and vessel identity of each face |
| exact_observations | Case × 2 species × 3 sites × 24 times |
| observations | Case × 4 realizations × 2 species × 3 sites × 24 times; realization 0 is clean |
| lengths, counts, sensors | Per-vessel geometry and predeclared observation sites |

Topology connectivity is recorded in protocol.json. All three topologies use this schema.

Observations are sampled from the 1 ms resolved reference before export. The exported y and volume arrays use the common 2 ms scoring grid; use the stored observations to reproduce the inverse benchmark exactly.

Synthetic observation sites use linear interpolation of reference cell means. The named site coordinates do not imply clinical point measurements.
