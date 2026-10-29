# Independent three-tube CFD blood experiment

All commands run on the project server from the repository root. This is an additive experiment for the existing three-vessel research checkout; it requires its raw-input archive and frozen checkpoints. It is not a new standalone public release. Existing models, baseline source and previous results remain unchanged.

## Environment

Use the project's isolated environment. In addition to the pinned core requirements:

    .venv/bin/python -m pip install scikit-fem==12.0.2

The optional CPU/CUDA sparse-solver timing comparison uses:

    .venv/bin/python -m pip install numpy==1.26.4 cupy-cuda12x==13.6.0

The chemistry and inference implementations use the already installed PyTorch/CUDA stack, not CuPy. A CUDA GPU is required for the batched chemistry, frozen inference and inverse benchmark.

## Raw inputs and independence

The independent blood solver reads runs/sen-input/input.npz, exported previously from the pinned upstream numerical three-vessel .his archive and its geometry/wall parameters. It reads no Sen model checkpoint or learned blood field. Original upstream files remain in external/sen2024/PIGNN_3_artery/Snaps/.

It uses initial area/mean velocity at the numerical sampling positions and the three external endpoint mean-velocity waveforms. It does not prescribe internal junction flow: a shared junction traction is solved along with the three finite-element velocity/pressure fields to enforce flow balance.

The main and refined blood solvers write into new output directories and refuse to overwrite existing directories.

## Generate independent blood

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -m gp52.three_independent.blood \
      --output runs/independent-cfd-v1/blood-main \
      --nx 24 8 14 --nr 10 --dt .001 --duration 1.23 \
      --sample-refine 4 --sample-radial 24

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -m gp52.three_independent.blood \
      --output runs/independent-cfd-v1/blood-refined \
      --nx 36 12 21 --nr 15 --dt .0005 --duration 1.23 \
      --sample-refine 4 --sample-radial 24

Both commands can run simultaneously on separate CPU cores. Current measured sparse CPU solves are faster than the tested CUDA sparse QR backend for these small CFD matrices. The benchmark is reproducible with:

    OPENBLAS_NUM_THREADS=1 .venv/bin/python -m gp52.three_independent.gpu_benchmark

## Reference, frozen forward, inverse and convergence

For a fresh output tree, first copy the published report protocol to runs/independent-cfd-v1/protocol.json. The locked protocol in runs/independent-cfd-v1/protocol.json must exist. It records the fixed old inverse regularization and the observation/error windows. No model training command is called.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -m gp52.three_independent.pipeline

The pipeline waits for each blood output, generates the resolved chemical reference on CUDA, performs frozen forward checks and all inverse fits, and completes numerical refinement checks. It skips already completed stages; partially failed output directories must be retained under a new name before rerunning that stage. Each active stage is recorded in runs/independent-cfd-v1/pipeline-status.json.

Individual evaluations:

    .venv/bin/python -m gp52.three_independent.evaluate --blood cfd --seed 0
    .venv/bin/python -m gp52.three_independent.evaluate --blood cfd --seed 42
    .venv/bin/python -m gp52.three_independent.evaluate --blood sen --seed 0
    .venv/bin/python -m gp52.three_independent.evaluate --blood sen --seed 42

Seed 0 denotes the fixed C0 baseline. Repeat neural evaluations for seeds 43 and 44. The two blood arms share one observations.npz file. The Sen arm uses the original projected blood and evaluates against the independent CFD biochemical reference.

## Report

    .venv/bin/python -m gp52.three_independent.report
    .venv/bin/python -m gp52.three_independent.viewer

The report checks protected-file hashes before finishing. It writes only reports/independent-cfd-v1 and the new experiment folder. Previous reports/all-species-v3 files are preserved.

## Meaning of the files

- blood-main/flow.npz: independent area, flow, axial/radial velocity profiles and pressure.
- blood-main/finite-elements-*.npz: complete finite-element states, meshes and wall motion.
- chem-main/data.npz: synthetic Th/Fg/Fn reference for the fixed 48 inlet waveforms.
- observations.npz: one shared clean/noisy sparse observation set.
- evaluation-cfd: candidates using common independent blood.
- evaluation-sen: candidates using common Sen blood against the same independent targets.
- cache-cfd and cache-sen: separate flow-dependent operators.
- protocol.json: frozen evaluation rules.
- protected-files.json and report integrity.json: preservation audit.

Only cases 32–47 contribute to the reported held-out performance. Cases 24–31 retain their validation labels, but no new model or inverse hyperparameter selection is performed in this experiment. Cases 0, 32 and 47 are used for numerical convergence checks.

The word "CFD" here refers to three coupled axisymmetric Navier–Stokes tubes, not a geometrically resolved three-dimensional bifurcation. Radial chemical states exist only in the reference; all evaluated C0/neural biochemical states remain strictly one-dimensional.


## Additional reference checks already included in this run

The following checks do not change any learned weights or baseline equations.

    .venv/bin/python -m gp52.three_independent.blood \
      --output runs/independent-cfd-v1/blood-wall-refined \
      --nx 36 12 21 --nr 12 --grading 2 --dt .001 --duration 1.23 \
      --sample-refine 4 --sample-radial 24

    .venv/bin/python -m gp52.three_independent.resample \
      --blood runs/independent-cfd-v1/blood-main \
      --output runs/independent-cfd-v1/blood-main-sampled --radial 96

    .venv/bin/python -m gp52.three_independent.finish_checks

The profile-assumption diagnostic, which is explicitly excluded from the primary reference:

    .venv/bin/python -m gp52.three_independent.chemistry \
      --blood runs/independent-cfd-v1/blood-main \
      --output runs/independent-cfd-v1/chem-parabolic-diagnostic \
      --refine 4 --radial 24 --dt .001 --parabolic

Run the frozen inference batch-size benchmark before evaluating the inverse fits:

    .venv/bin/python -m gp52.three_independent.batch_benchmark

This writes gpu-batch-benchmark-large.json. The measured selected batch size is 1280 on the RTX 5090. It affects only computational batching, not model outputs or the inverse objective.
