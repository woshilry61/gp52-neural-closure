# Independent three-vessel CFD blood validation

## Main result

With common independent CFD blood, the frozen neural models improve known-inlet prediction: Th error falls from 14.60% to 11.17%, Fg from 4.87% to 3.77%, and Fn from 9.47% to 6.84%.

However, the current frozen inverse setup recovers the inlet less accurately than C0 at all three sensor counts. Its Fn predictions remain better. This supports a concentration-prediction advantage on this test, not an overall claim that the neural method outperforms C0 for every task.

Substituting Sen blood increases the neural known-inlet Fn error from 6.84% to 7.02%. The much larger change from the old results is associated with using solved radial velocity profiles instead of a prescribed parabola; the controlled profile diagnostic below tests this directly.

## What changed

This additive experiment replaces the former Sen-driven chemical reference with an independently solved blood field. Three axisymmetric, compliant tubes solve the ALE Navier–Stokes equations. Their open junction sections share one normal traction, whose Lagrange multiplier imposes exact net flow balance. The junction has zero volume and biochemical concentrations mix perfectly.

The boundary and initial data come from the original numerical three-vessel archive, not Sen predictions. External endpoint mean-velocity waveforms are prescribed, with periodic continuation after 0.82 s. Tube length, reference area and wall stiffness are unchanged. This is a coupled axisymmetric tube reference, not a three-dimensional Y-shaped bifurcation simulation.

No neural network was trained. The three existing biochemical checkpoints and the fixed C0 equations were retained.

## What was trained previously, and what is frozen now

The existing Sen three-vessel blood model was trained separately before this experiment. The biochemical graph model then learned two modules on the three-vessel prescribed-profile data: a transport memory-flux closure and a reaction-source closure.

The biochemical split is 24 complete inlet cases for training, 8 for validation and 16 for held-out testing. Each module used 3,000 main Adam iterations (3,004 optimizer updates including setup), with checkpoints selected by validation flux/source error. Training was staged: transport first, reaction second. The network was not trained directly to minimize inlet-recovery error. Training features used reference concentration fields; deployment obtains them from the one-dimensional rollout.

In this experiment all learned parameter tensors are frozen. The inverse optimizer adjusts only seven inlet-amplitude parameters per case. The new independent CFD data are never used to update network weights.

## Two controlled comparisons

1. **Common independent blood:** C0 and all frozen networks receive exactly the same CFD-derived area and mean flow. Both are compared against the same radially resolved biochemical reference.
2. **Blood substitution:** both candidates receive the original projected Sen area and flow. The independent biochemical reference, observations, noise, inverse regularization and fit settings remain identical.

The second-minus-first change measures the impact of substituting blood fields on this experiment. It is not an additive decomposition of error. Sen-versus-CFD discrepancy includes both neural reconstruction error and differences between the original one-dimensional blood model and axisymmetric CFD. A separate table compares raw Sen predictions with the original numerical midpoint samples.

## Blood discrepancy and the existing conservation projection

The Sen mean-flow relative L2 errors against the independent CFD are:

| Vessel | Raw Sen Q | Projected Sen Q used by both candidates |
| --- | --- | --- |
| Parent | 2.24% | 5.08% |
| Daughter 1 | 1.61% | 5.60% |
| Daughter 2 | 4.40% | 23.24% |

The projection reconciles the predicted area changes and branch flows with volume/junction conservation. It changes the smaller daughter branch substantially. This separates an additional projection effect from raw blood-network approximation error. The existing projection is retained in this experiment; C0 and the neural candidates always share its same output in the Sen-input arm. The original-source midpoint comparison is separately reported in blood-errors.csv.

## Known-inlet prediction

Volume-weighted space-time relative L2 error over 0–1.23 s, 16 held-out chemical inlet cases. Neural values average the three frozen seeds; no seed is selected on these results.

| blood | method | Th | Fg | Fn |
| --- | --- | --- | --- | --- |
| cfd | C0 | 14.60% | 4.87% | 9.47% |
| cfd | neural | 11.17% | 3.77% | 6.84% |
| sen | C0 | 15.14% | 5.25% | 9.77% |
| sen | neural | 12.02% | 4.36% | 7.02% |

## Sparse inversion

All fits observe **Th and Fg only**; Fn is an unobserved prediction. Sites are parent 30 mm, daughter 1 10 mm, daughter 2 17 mm, added in that order. Every site contributes 24 time samples through 1.23 s. Each case has a clean fit and three fixed noise realizations (absolute standard deviation 0.02). The following table averages the noisy fits.

Inlet error uses the agreed 0–0.82 s interval. Concentration errors use 0–1.23 s. All methods and blood inputs use the **independent reference volume weights**, preventing the scoring weights from changing with a candidate's blood field.

| blood | layout | method | inlet | Th | Fg | Fn |
| --- | --- | --- | --- | --- | --- | --- |
| cfd | 1 | C0 | 33.92% | 13.82% | 3.72% | 9.29% |
| cfd | 1 | neural | 45.34% | 13.93% | 3.38% | 7.74% |
| cfd | 2 | C0 | 41.86% | 14.36% | 3.31% | 8.32% |
| cfd | 2 | neural | 46.98% | 13.68% | 2.95% | 6.46% |
| cfd | 3 | C0 | 39.52% | 13.79% | 3.29% | 7.58% |
| cfd | 3 | neural | 49.73% | 13.97% | 2.90% | 6.20% |
| sen | 1 | C0 | 33.44% | 14.41% | 4.28% | 9.40% |
| sen | 1 | neural | 45.21% | 14.50% | 3.93% | 7.68% |
| sen | 2 | C0 | 41.60% | 15.07% | 3.85% | 8.32% |
| sen | 2 | neural | 47.30% | 14.53% | 3.54% | 6.53% |
| sen | 3 | C0 | 39.30% | 14.49% | 3.81% | 7.63% |
| sen | 3 | neural | 49.44% | 14.62% | 3.50% | 6.26% |

Clean-fit results are in summary.csv. Per-case errors, all seed results and optimizer termination diagnostics are provided separately. Paired uncertainty in paired-comparison.csv bootstraps the 16 inlet cases after averaging noise and seeds within each case; noise replicates are not treated as independent cases.

Inverse regularization is locked to the previous validation selection: C0 0.0001, neural seeds 0. The same three starts, bounds [0,1.2], maximum 100 iterations and projected-gradient tolerance 1e-8 apply to both blood inputs. There is no new tuning on the independent CFD test cases.

## Why the independent reference is harder

A diagnostic keeps the CFD area/mean flow identical but replaces the solved radial velocity profile with the former prescribed parabola. The candidate predictions are unchanged; only this diagnostic reference profile changes. This prescribed-profile calculation is not used as ground truth for the primary evaluation.

| reference_profile | method | Th | Fg | Fn |
| --- | --- | --- | --- | --- |
| solved_CFD_profile | ordinary_1d | 14.60% | 4.87% | 9.47% |
| parabolic_at_identical_CFD_AQ | ordinary_1d | 9.21% | 3.79% | 5.76% |
| solved_CFD_profile | neural_42 | 11.21% | 3.75% | 6.86% |
| parabolic_at_identical_CFD_AQ | neural_42 | 2.05% | 0.83% | 1.60% |
| solved_CFD_profile | neural_43 | 11.11% | 3.77% | 6.80% |
| parabolic_at_identical_CFD_AQ | neural_43 | 2.14% | 0.86% | 1.69% |
| solved_CFD_profile | neural_44 | 11.19% | 3.79% | 6.86% |
| parabolic_at_identical_CFD_AQ | neural_44 | 2.09% | 0.84% | 1.64% |

The difference tests sensitivity to the radial-profile assumption. Th has no reaction sink in these equations, so its degradation already identifies a transport-model mismatch. Fn additionally depends on the transported Th/Fg fields and their reaction covariance; its error cannot all be assigned to the reaction closure. The frozen network was fitted to prescribed-profile reference data and was not trained to accommodate the independently solved velocity profiles.

There is also a structural restriction in the frozen transport closure: its memory-flux amplitude is proportional to the absolute cross-section-mean velocity. At zero mean flow this correction vanishes, even though forward core flow and backward near-wall flow can coexist in the independent CFD. The same mean flow therefore need not imply the same unresolved solute flux. No change to that architecture or its parameters was made here.

These results apply to the current frozen model and its prescribed-profile training data. Improving the model is outside this frozen evaluation.

## Reference checks

| comparison | case | Th | Fg | Fn |
| --- | --- | --- | --- | --- |
| chem-refined | 0 | 0.26% | 0.10% | 0.18% |
| chem-refined | 32 | 0.30% | 0.05% | 0.15% |
| chem-refined | 47 | 0.35% | 0.18% | 0.14% |
| chem-flow-refined | 0 | 0.28% | 0.13% | 0.21% |
| chem-flow-refined | 32 | 0.49% | 0.08% | 0.23% |
| chem-flow-refined | 47 | 0.40% | 0.20% | 0.25% |
| chem-time-refined | 0 | 0.36% | 0.17% | 0.22% |
| chem-time-refined | 32 | 0.59% | 0.09% | 0.23% |
| chem-time-refined | 47 | 0.44% | 0.24% | 0.25% |
| chem-sampling-refined | 0 | 0.26% | 0.10% | 0.18% |
| chem-sampling-refined | 32 | 0.29% | 0.05% | 0.14% |
| chem-sampling-refined | 47 | 0.34% | 0.17% | 0.13% |
| chem-wall-refined | 0 | 0.28% | 0.09% | 0.22% |
| chem-wall-refined | 32 | 0.43% | 0.08% | 0.20% |
| chem-wall-refined | 47 | 0.28% | 0.13% | 0.24% |

chem-refined doubles axial/radial transport resolution with the same CFD blood and time step. chem-flow-refined changes the blood finite-element mesh and blood time step while retaining the reference chemical discretization. chem-time-refined additionally halves the chemical time step. chem-sampling-refined repeats the fine chemical mesh after independently resampling the saved finite-element velocity into 96 radial bins (rather than subdividing the original 24-bin velocity). chem-wall-refined uses a radially graded blood mesh concentrated near the wall. These comparisons use prespecified cases 0, 32 and 47; case 0 is a numerical check, not a new training sample.

Uniform concentration preservation error: 0.000e+00. Maximum chemical mass-balance residual, normalized by network volume: 2.328e-12. The CUDA linear solver is checked against a CPU sparse direct solve and its residual is checked at every step.

Interpolated finite-element flow is transferred conservatively to the chemical mesh. Corrections and exact continuity residuals are saved in flow-transfer.json. This projection is applied to the independent CFD field, not learned from biochemical targets.

## Compute and preservation

On the identical 4,120-order CFD system, warm CPU sparse-LU median solve time was 10.07 ms; the tested CUDA sparse-QR implementation took 56.53 ms. Their relative solution difference was 2.117e-14. The current CPU implementation therefore remains the blood linear solver. This benchmark does not claim that every GPU sparse solver is slower.

Resolved chemistry runs batched on CUDA, as do frozen neural inference and inverse forward/Jacobian evaluations. GPU utilization logs are retained with each run. Matrix assembly and small optimization control steps use the CPU.

The main prediction/inverse suite finished before the slower reference-refinement checks. Those checks dominated elapsed time; the sparse-solver microbenchmark covers only the linear solve, not finite-element assembly or interpolation.

All 269 protected pre-existing files are unchanged. Frozen parameter tensors are checked when loading; flow-dependent feature buffers are recomputed from the supplied blood. With the original blood, the new adapter reproduces all four previous solvers to approximately 1e-15.

## Interpretation limits

- This validates one three-vessel geometry and one external blood driving condition, with 16 held-out biochemical inlet waveforms. It does not establish population-wide blood-flow generalization.
- The reference omits a geometric bifurcation angle, local junction vortices, junction volume and detailed wall mechanics.
- Biochemical concentrations are synthetic solutions of the specified transport/reaction equations, not measured Th/Fg/Fn data.
- The initial axial velocity profile is assumed parabolic, with a divergence-compatible radial initial velocity reconstructed from the raw one-dimensional fields. There is no additional periodic CFD spin-up; this is the declared initial-value experiment.
- The blood solver uses raw numerical initial states and measured-in-the-archive endpoint waveforms; it is independent of the trained Sen model, not independent of the original numerical problem definition.
- Existing Sen-conditional results remain available and have not been replaced.

## Files

- summary.csv: complete clean/noisy aggregate tables
- per-case-errors.csv: all forward and inverse errors
- per-vessel-errors.csv: separate Th/Fg/Fn errors for each vessel
- paired-comparison.csv: paired case-level uncertainty
- blood-substitution-impact.csv: Sen-input minus CFD-input error, for each method
- blood-errors.csv: blood reconstruction and refinement errors
- reference-convergence.csv: biochemical reference checks
- optimizer-*.csv: selected-fit convergence diagnostics
- protocol.json, integrity.json: frozen protocol and preservation audit

## Reproduce

See README.md in this report directory. Open curves.html for all test cases, noise replicates and sensor layouts. The original experiment remains under reports/all-species-v3.
