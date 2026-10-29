> Historical single-vessel release. For the current single/three/seven-vessel independent-CFD experiment, see [current model](CURRENT_MODEL.md), [reproduction](UNIFIED_REPRODUCTION.md) and [results](../reports/unified-cfd-baselines-v1/REPORT.md).

# Reproduce the final model from raw upstream input

This workflow regenerates the blood-flow reconstruction, CFD chemical labels,
solver buffers, inverse initialization, and biochemical neural models. It begins
with the raw numerical waveform in the pinned Sen repository and the declared
experiment configuration in this project.

## One command after installation

Complete the normal README installation, then install the finite-element dependency:

~~~bash
python -m pip install -r requirements-raw.txt
python scripts/reproduce_raw.py --output runs/from-raw
~~~

The output directory must be new. No existing published checkpoint is overwritten.
The command downloads upstream source and the raw waveform itself. Released CFD
labels are downloaded only at the final comparison stage, after regeneration and
training have finished.

## Actual dependency chain

1. Clone the original Sen repository at revision
   bf0ec89f0aab8a639bdd2d363dc3c0cfae770c1d and verify the waveform checksum.
2. Extract the four raw locations at 0, 30, 40 and 60 mm during 4.94–5.76 s.
3. Solve independent axisymmetric compliant-wall ALE Navier–Stokes flow with
   P2/P1 finite elements, 32 axial by 24 radial mesh divisions and a 0.001 s step.
4. Continue the stored fluid/wall state from 0.82 to 1.23 s using the periodically
   repeated endpoint waveform.
5. Sample synthetic velocity/area observations at 6, 30 and 54 mm, with 48 times.
   Create clean observations and three fixed noise realizations.
6. Train the project's conservative Sen-inspired blood PIGNN from random
   initialization: three initializations for each observation condition,
   20,000 Adam loop steps and up to 15,000 L-BFGS inner iterations.
   Choose the seed using the physics/data objective.
7. Generate 16 training, eight development and eight test chemical CFD trajectories
   in four batches. Transport and reaction use the independent CFD velocity field.
8. Build the 1-D solver buffers from the newly trained clean blood reconstruction.
   Recompute the numerical unit responses used to initialize sparse inversion.
9. Export concentration means, direct physical/discrete reaction moments, training
   features and sparse observation values. Reinitialize and train the transport
   and reaction heads, select on development data, then evaluate all test cases.
10. Compare the generated arrays and the new forward/inverse results with the
    unchanged public release and verify the original asset hashes.

The finite-element calculations use CPU sparse assembly/factorization. The blood
PIGNN and biochemical training use CUDA; the biochemical trainers retain full
trajectory batching and CUDA Graphs.

## Inputs and intermediate formats

| Stage | Inputs | Outputs |
|---|---|---|
| Raw extraction | Pinned upstream .his file | Four-location time, area, velocity and flow arrays in SI units |
| Fluid reference | Initial fields, endpoint velocities, wall law and mesh/time settings | Finite-element velocity/pressure/wall states and section averages |
| Blood learning | Three interior sites, 48 times, geometry and wall law | Twelve independently initialized reconstructions; clean seed selected by training loss |
| Chemical reference | Independent CFD velocity, prescribed diffusivity/reaction and 32 inlet seeds | Fine-time section means, physical reaction moments and observation values |
| Solver preparation | Newly reconstructed clean blood flow and fixed numerical coefficients | Finite-volume buffers and linear-response inverse initialization |
| Biochemical learning | 16 regenerated training and eight development cases | Fresh transport head and development-selected reaction head |
| Final evaluation | Eight regenerated test cases, fixed inverse settings and new weights | Forward errors, 1/2/3-site inverse errors and direct reaction-rate errors |

The two kinds of sparse observations are different. Blood training uses velocity
and area at 6/30/54 mm over the full 1.23 s. Biochemical inversion uses Th and Fg
at one, two or three of those sites, at 24 times through 0.82 s. Fn is not an
observed variable in biochemical inversion.

The 1-D concentration grid has 41 cells and 616 times. The CFD generator retains
radial resolution only in the independent reference. No radial concentration
bins are added to the reduced model.

## Running stages separately

The main driver records its exact commands in pipeline.json and each stage's
standard output in logs/. For example, after the initial CFD solve:

~~~bash
python -m gp52.raw.extend_flow --source runs/from-raw/cfd/initial \
  --waveform runs/from-raw/raw/waveform.npz \
  --output runs/from-raw/cfd/extended --duration 1.23

python -m gp52.raw.observations --flow runs/from-raw/cfd/extended/flow.npz \
  --output runs/from-raw/blood-observations
python -m gp52.raw.train_blood --data runs/from-raw/blood-observations \
  --output runs/from-raw/blood --adam 20000 --lbfgs 15000

python -m gp52.raw.generate_labels --flow runs/from-raw/cfd/extended \
  --output runs/from-raw/chemistry
python -m gp52.raw.prepare_assets --blood runs/from-raw/blood \
  --output runs/from-raw/assets
python -m gp52.raw.export_data --references runs/from-raw/chemistry \
  --assets runs/from-raw/assets --output runs/from-raw/data

python scripts/retrain_and_evaluate.py --output runs/from-raw/biochemical \
  --prepared-data runs/from-raw/data --prepared-assets runs/from-raw/assets
python scripts/validate_raw_reproduction.py --run-directory runs/from-raw
~~~

These commands illustrate the dependencies; use fresh stage directories rather
than overwriting a completed run. The complete driver overlaps blood training on
the GPU with four CPU chemical batches. Its remaining stages follow dependency
order. The chemical test trajectories are generated alongside the other labels,
but are staged into the biochemical training run only after model selection.

## Relationship to Sen's original software

The upstream source is fetched directly from
[the authors' repository](https://github.com/ahmetsenemse/PIGNN-1D-Blood-Flow-Simulation).
Its pinned single-vessel waveform is the raw input. The checkout remains separate
from this repository and is not redistributed in our source archive.

The final project's blood model is **a project-authored PyTorch adaptation inspired
by Sen's ARMA graph architecture**, not the unchanged original TensorFlow training
script. It retains graph aggregation and the wall law, uses spline time
coefficients, predicts positive areas, and constructs flux through exact discrete
continuity. Both endpoint velocities are hidden from this blood-network training;
its observations come from the independent CFD calculation.

The original TensorFlow experiment is not an intermediate checkpoint dependency
of the final model. Running it would produce a different historical experiment.
The raw-input workflow trains the actual blood model used by the final project.

The CFD solver never uses the PIGNN prediction as its reference velocity. It uses
the raw numerical initial area/velocity and endpoint velocity traces, with the
documented assumed radial initial profile and wall law.

## Fixed experiment configuration

Geometry, constitutive constants, diffusivity, chemical rate, inlet knot locations,
random seeds, numerical closure coefficients and inverse regularization settings
are explicit experiment inputs. The legacy numerical closure coefficients are in
src/gp52/raw/config.py; the C1 corrections remain in src/gp52/transport.py.
The shared inverse settings are in src/gp52/assets/inverse-settings.json.

The workflow regenerates the derived arrays and trains the neural models. It does
not repeat the historical exploratory searches that selected those fixed numerical
coefficients or hyperparameters. This distinction is also recorded in the machine-
readable validation report.

## Outputs

~~~text
runs/from-raw/
  upstream/sen2024/       original pinned source and raw waveform
  raw/waveform.npz        extracted physical fields
  cfd/initial/           0–0.82 s fluid/wall trajectory and finite-element state
  cfd/extended/          0–1.23 s continuation
  blood-observations/    sparse observations used by blood training
  blood/                 newly trained blood weights, fields, losses and GPU log
  chemistry/             32 newly computed finite-element chemical trajectories
  assets/                regenerated flow.npz and inverse-context.npz
  data/                  regenerated train/development/test archives
  biochemical/           new closure checkpoints and unified evaluation
  comparison-reference/ released labels, downloaded only for final checking
  logs/                  stage output
  pipeline.json
  validation.json
~~~

The published model remains the default for ordinary demo/evaluation commands.
To evaluate the new raw-input result explicitly:

~~~bash
GP52_PREPARED_ASSETS=runs/from-raw/assets \
GP52_DATA_DIR=runs/from-raw/data \
python -m gp52.evaluate --methods C1 GG retrained --inverse \
  --transport-weights runs/from-raw/biochemical/transport/runs/graph/best.pt \
  --reaction-weights runs/from-raw/biochemical/reaction/reaction.pt \
  --output outputs/from-raw
~~~

In this command C1 and GG use the regenerated flow inputs too. The raw validation
script separately compares their scores with the historical scores using the
original packaged flow. This separates flow regeneration differences from neural
retraining differences.

## Validation

The comparison tolerances are fixed in scripts/validate_raw_reproduction.py:

| Quantity | Acceptance threshold |
|---|---:|
| Maximum absolute change in regenerated dataset arrays | 1e-8 |
| Maximum absolute change in prepared solver/initialization arrays | 1e-5 |
| Largest change in any compared forward/inverse relative-L2 error or reported direct-rate error | 0.0005 (0.05 percentage points) |
| Distributed original asset hashes | All unchanged |

These tests assess reproduction of the declared experiment. They do not establish
new-flow generalization or replace a mesh-convergence study.

The independent run on 26 October 2026 **passed all checks**. It used a fresh
checkout, an isolated Python environment and a newly fetched upstream repository.
The complete pipeline took **1,914.48 s (31.91 minutes)** on the documented Linux /
RTX 5090 system, excluding environment installation.

| Check | Observed result |
|---|---:|
| All arrays in regenerated train/development/test archives | Maximum absolute difference **0** |
| All 24 flow buffers and eight inverse-initialization arrays | Maximum absolute difference **0** |
| Newly trained transport parameter tensors vs published model | **Identical** |
| Newly trained selected reaction parameter tensors vs published model | **Identical** |
| Largest compared forward/inverse/direct-rate metric difference | **0** |
| Discrete continuity residual across the 12 blood reconstructions | 7.23e-17 |
| Original distributed model and input assets | All hashes unchanged |
| Executed CFD/blood source snapshots vs released implementation | All matched |

The selected clean blood reconstruction exactly reproduces the packaged area,
face area and flow arrays. The reaction selection again chooses source-loss weight
0.03, using development data. Test labels do not choose the candidate.

### Reproduced accuracy

These are eight-inlet means in the primary 0-0.82 s window. The inverse rows use
observation noise standard deviation 0.02, averaged over three realizations per
inlet. The complete per-inlet, noise-free and full-window records are also saved.

| Quantity | Calibrated numerical C1 | Newly trained final model |
|---|---:|---:|
| Forward Th relative L2 | 4.9360% | 2.1426% |
| Forward Fg relative L2 | 1.9159% | 1.4286% |
| Forward Fn relative L2 | 1.8204% | 0.4612% |
| One-site inverse Fn relative L2 | 4.5848% | 4.0213% |
| Two-site inverse Fn relative L2 | 1.7825% | 0.8090% |
| Three-site inverse Fn relative L2 | 1.8032% | 0.7461% |

This is a reproduction, not an additional independent test population. The
existing limitations remain: one-site noisy extrapolation is weak, new blood
flows/geometries are untested, and inference has no demonstrated speed advantage
over the calibrated numerical baseline.

### Measured execution

| Stage | Wall time |
|---|---:|
| Initial CFD flow, 0-0.82 s | 642.16 s |
| CFD continuation to 1.23 s | 340.64 s |
| Blood training, including process setup | 75.46 s |
| All 32 chemical CFD cases | 229.20 s |
| Biochemical training, selection and unified evaluation | 688.06 s |
| Complete pipeline including preparation and validation | 1,914.48 s |

Blood training overlaps chemical CFD generation, so the stage times do not add
directly to the total. CFD uses CPU sparse solvers. The small blood model's
one-second sampled GPU utilization averaged 67.86%; its L-BFGS stage includes host
line-search synchronization. Full-trajectory biochemical training averaged
99.64% for transport and 98.57-99.14% for the three reaction candidates.

### Records and downloadable outputs

- [Pipeline and validation records](../benchmarks/raw-reproduction/)
- [Per-case unified evaluation](../benchmarks/raw-reproduction/evaluation.json)
- [Training initialization, update counts and GPU measurements](../benchmarks/raw-reproduction/training-summary.json)
- [Separate run archive](https://github.com/woshilry61/gp52-neural-closure/releases/tag/raw-reproduction-20261026)
- [Archive size and SHA-256](../benchmarks/raw-reproduction/archive.json)

The 82.1 MB run archive includes the new blood and biochemical weights, regenerated
training/evaluation datasets, prepared solver inputs, stage logs and validation
records. Its MANIFEST.json lists every included file's checksum. Full CFD
finite-element states and full prediction arrays are generated locally by the
workflow; they are not duplicated in this compact archive. The original upstream
checkout and waveform are also excluded and are fetched from the authors.

The pipeline records the checkout base revision; validation.json records the
individual source-file SHA-256 values and checks the executed solver snapshots.
Exact equality was observed in this environment. It is not a guarantee of
bitwise-identical results on every GPU or future library version.
