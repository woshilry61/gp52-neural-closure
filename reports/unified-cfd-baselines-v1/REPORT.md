# Unified independent-CFD neural baselines

## Scope and protocol

This experiment fixes the data-generation, partition and evaluation protocol before designing a new model. PROTOCOL_NOTES.md records the scope extension and numerical execution changes; the shared inlet arrays and partition IDs remain fixed. The existing single-vessel and three-vessel architectures are retained and freshly trained. The seven-vessel baseline reuses the three-vessel heads and losses, extending the adjacency and numerical operators to three junctions. C0 is the fixed conventional numerical comparison. The independent CFD calculations provide numerical reference data, not laboratory measurements.

All three topologies use exactly the same 112 seven-knot biochemical inlet curves: 64 training, 16 validation and 32 test cases. Every amplitude is sampled uniformly from 0.05 to 1.15. Splits contain complete network trajectories, not individual points, vessels or time samples. Training and validation alone determine weights and inverse regularization. Test labels are generated in advance but are excluded from all fitting and selection losses.

Each topology has separately trained weights, rather than one shared model across vessel counts. There is one independent blood-flow condition per topology. This tests new chemical inlet curves under fixed flow, not new-flow or new-patient generalization. The test cases are first scored after checkpoint selection in this run; once these results are inspected they must not be called a fresh blind test for subsequent model design.

## Shared reference generation

Blood comes from the existing independently solved compliant-wall axisymmetric ALE Navier–Stokes calculations. The single-vessel reference uses the verified raw-input reproduction; the three-vessel reference uses the refined independent coupled calculation. Seven-vessel blood is freshly solved from the original numerical inlet/outlet archives, with seven axisymmetric tubes coupled by three common-traction constraints. None of the biochemical references uses Sen-predicted blood. The blood solvers retain their declared initialization and prescribed external velocity conditions; these are not full three-dimensional bifurcations or periodic-steady clinical flow fields.

All three use the same conservative cross-section-resolved finite-volume biochemical generator, prescribed base diffusivity 2e-4 and reaction rate 2 Th Fg. This is a computational baseline with fixed chemical parameters; the errors measure agreement with that simulator. Concentrations and inlet amplitudes use the existing normalized model units, not measured biochemical concentrations. Th is catalytic; Fg is consumed and immobile Fn accumulates. There is no Fn-to-blood feedback. The upstream concentration reservoir stays connected; distal inflow has Th=0 and Fg=1. Junctions are perfectly mixed and have zero volume.

Reference grids use four axial cells per candidate cell, 24 radial cells and a 1 ms time step. Physical reaction means and advective flux covariance labels are obtained from the resolved radial states. A least-squares blood-flow transfer enforces the same discrete volume conservation in reference and candidate grids. Every compared method within a topology receives identical independent area/flow arrays.

## What remains unchanged

The single-vessel baseline retains its two-channel advected flux memory, local graph transport head and graph reaction predictor-corrector. The three-vessel baseline retains its original graph adjacency, two local memory channels, mean-speed-scaled transport amplitude and graph reaction head. No architecture limitation is repaired in this experiment.

The original training objectives remain topology-specific: single-vessel trajectory training and source-plus-trajectory reaction training; three-vessel physical flux/source supervision. Source hashes are recorded in protocol.json. Three fresh random seeds are trained per topology. These are three independent fits, not an ensemble used to improve the reported predictions.

Single-vessel training uses shuffled batches of 16 from the 64 training cases, with the original 1,500 transport steps, 1,000 reaction source steps and 500 reaction trajectory steps. Three source-loss weights are selected on validation only. Three- and seven-vessel training use 3,000 main steps per head on all 64 cases. CUDA warm-up updates are retained where used by the original source-training procedure.

C0 has no memory or fitted closure. Its existing per-topology temporal discretizations are retained: coupled implicit reaction in the single vessel and transport followed by exact local reaction in three and seven vessels. Both use the same physical equations and 2 ms candidate grid. This protocol does not silently replace the locked C0 implementations or claim bitwise-identical temporal schemes.

## Known-inlet prediction

Errors are per-case, reference-volume and trapezoidal-time weighted relative L2 norms over the whole network and 0–1.23 s, then averaged equally across test cases. Retrained neural errors are averaged across the three seeds. Old weights are replayed on the same new blood and reference for the forward comparison only; their historical results and files remain unchanged.

### Single

| Method | Th | Fg | Fn |
| --- | --- | --- | --- |
| Fixed C0 | 14.09% | 5.31% | 8.27% |
| Retrained baseline (3 seeds) | 1.41% | 0.60% | 0.52% |
| Old weights | 1.57% | 0.60% | 0.52% |

### Three

| Method | Th | Fg | Fn |
| --- | --- | --- | --- |
| Fixed C0 | 13.56% | 5.22% | 8.90% |
| Retrained baseline (3 seeds) | 4.92% | 1.85% | 2.74% |
| Old weights | 10.28% | 3.97% | 6.29% |

### Seven

| Method | Th | Fg | Fn |
| --- | --- | --- | --- |
| Fixed C0 | 14.19% | 4.60% | 9.65% |
| Retrained baseline (3 seeds) | 4.91% | 1.60% | 2.78% |

This is a harmonized baseline experiment, not a one-factor ablation of data provenance: the training case count and sampling range also change.

## Sparse inverse reconstruction

Only Th and Fg are observed, at 24 common times from 0.002 to 1.23 s. Fn is never observed. Layouts add the first one, two or three predeclared sites. Their distances along the network path from the upstream inlet are:

| Topology | First site | Second site | Third site |
| --- | --- | --- | --- |
| Single | 30 mm | 6 mm | 54 mm |
| Three | 30 mm | 70 mm | 77 mm |
| Seven | 30 mm | 99.5 mm | 141 mm |

The second single-vessel site is close to the inlet; the added branched-network sites are downstream. Comparisons between NN and C0 use identical sites within each topology. Absolute inverse errors across topologies also reflect different observation distances, geometry and blood flow, so they are not a controlled test of vessel count alone.

 Noise has absolute standard deviation 0.02 with three identical paired realizations across candidate methods. Clean results are also retained.

All methods fit seven inlet amplitudes bounded in [0,1.2], using the same objective normalization, three starts, bounded optimization and stopping criteria. Each method chooses from lambda={0,0.0001,0.01} using validation inlet error only. Inlet error is scored over 0–0.82 s; all three concentration errors use 0–1.23 s. Network weights are frozen during every inverse fit. Final inference uses FP64 arithmetic, including the reaction head and nonlinear reaction features. For the branched networks, sparse adjacency application evaluates the same fixed graph operator without dense zero multiplications; numerical parity is checked. Preliminary FP32 three-vessel validation exposed gradient stagnation and is retained separately as a numerical diagnostic, not included in the final results.

The following are noisy means. The complete clean/noisy and per-case results are in summary.csv and per-case-errors.csv.

### Single

| Sites | Method | Inlet | Th | Fg | Fn |
| --- | --- | --- | --- | --- | --- |
| 1 | Fixed C0 | 35.12% | 13.42% | 3.85% | 8.23% |
| 1 | Retrained baseline (3 seeds) | 21.78% | 6.10% | 0.76% | 2.38% |
| 2 | Fixed C0 | 13.71% | 9.93% | 3.77% | 6.25% |
| 2 | Retrained baseline (3 seeds) | 4.15% | 1.96% | 0.62% | 0.86% |
| 3 | Fixed C0 | 15.08% | 8.41% | 3.40% | 4.24% |
| 3 | Retrained baseline (3 seeds) | 4.14% | 1.87% | 0.60% | 0.78% |

### Three

| Sites | Method | Inlet | Th | Fg | Fn |
| --- | --- | --- | --- | --- | --- |
| 1 | Fixed C0 | 30.47% | 13.74% | 3.89% | 9.64% |
| 1 | Retrained baseline (3 seeds) | 26.44% | 7.83% | 1.85% | 4.73% |
| 2 | Fixed C0 | 41.09% | 15.42% | 3.79% | 10.44% |
| 2 | Retrained baseline (3 seeds) | 25.56% | 7.58% | 1.62% | 4.50% |
| 3 | Fixed C0 | 39.37% | 14.89% | 3.74% | 10.00% |
| 3 | Retrained baseline (3 seeds) | 24.25% | 7.24% | 1.60% | 4.05% |

### Seven

| Sites | Method | Inlet | Th | Fg | Fn |
| --- | --- | --- | --- | --- | --- |
| 1 | Fixed C0 | 30.17% | 14.64% | 3.84% | 10.54% |
| 1 | Retrained baseline (3 seeds) | 25.36% | 7.60% | 1.70% | 4.72% |
| 2 | Fixed C0 | 32.50% | 13.91% | 3.31% | 9.80% |
| 2 | Retrained baseline (3 seeds) | 24.60% | 7.22% | 1.52% | 4.27% |
| 3 | Fixed C0 | 40.43% | 15.25% | 3.21% | 10.71% |
| 3 | Retrained baseline (3 seeds) | 23.91% | 6.94% | 1.40% | 4.23% |

## Interpretation and uncertainty

Concentration prediction, inlet recovery and unobserved Fn reconstruction are distinct outcomes. A lower Fn error does not establish better inlet recovery. Paired intervals in paired.csv resample the 32 inlet cases after averaging noise repetitions and training-seed errors; neither noise nor seeds are counted as additional independent cases. These tests do not establish superiority over all possible calibrated traditional closures.

More observation sites do not guarantee lower inlet error when the candidate forward model differs from the reference physics. Additional sites constrain an approximate model differently; inlet regularization is selected across all layouts for each method. Convergence diagnostics distinguish this model/data limitation from an optimizer that stopped too early, but this experiment does not isolate the contribution of each cause.

Optimizer diagnostics include every selected fit, projected-gradient residual, stagnation and iteration-limit status. A selected fit that does not meet the gradient target is not silently labelled converged. See optimizer-summary.csv and the complete fit arrays. The same summary also reports whether additional starts improved the objective relative to the Th-based initializer and how far their recovered amplitudes differed. Multiple starts are a stability check, not a guarantee of global optimality. Inverse elapsed times are execution diagnostics; some overlapped reference-data generation and are not used to claim a speed advantage.

## Seven-vessel extension

The shared topology configuration includes Sen's seven-vessel tree with three junctions and five external endpoints. The generator handles multiple junctions using a topology-derived incidence matrix. The vessel ordering and source-file mapping are explicit, including the nonsequential upstream file order.

The same inlet IDs, partitions, reaction laws, noise schedule, inverse bounds and scoring rules apply. Independent seven-vessel blood and radial biochemical references have been generated, three new neural seeds have been trained, and fixed-C0 forward and sparse-inverse comparisons have been evaluated. A short-run serial/parallel blood-solver comparison was bitwise equal. There are no historical seven-vessel biochemical neural weights to replay.

## Reference and preservation checks

The seven-vessel CFD resolution is recorded in blood-audit.json. Its conservation and linear/nonlinear solver checks are retained. The biochemical mesh checks below refine the concentration solver; they do not by themselves establish convergence of the seven-vessel blood mesh.

- single: maximum concentration change on the three predeclared training-only mesh checks is 0.219%.

- three: maximum concentration change on the three predeclared training-only mesh checks is 0.316%.

- seven: maximum concentration change on the three predeclared training-only mesh checks is 0.420%.

- All 967 protected pre-existing files match their original SHA-256 values.

- Numerical replay and finite-difference checks are retained in each evaluation directory. GPU utilization and training timings are in training-runs.csv.
