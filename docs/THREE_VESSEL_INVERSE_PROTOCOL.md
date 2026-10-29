# Frozen three-vessel sparse inverse experiment

The forward extension at e827705245f5c9b56658a61bb8da70c6bf6c005a, its source,
results, and all model checkpoints remain unchanged. New code/results are additive.

## Methods and observations
- Ordinary one-dimensional finite-volume solver with mean-product reaction.
- Frozen graph transport/reaction model, all three training seeds 42/43/44.
- Identical projected Sen blood, 30/10/17 axial cells, dt=0.002 s, chemistry,
  initial states and reservoir boundaries.
- Nested sensor layouts: one = parent midpoint (30 mm); two adds daughter 1
  midpoint (10 mm); three adds daughter 2 midpoint (17 mm).
- Observe Th and Fg, never Fn, at 24 equally spaced times from 0.002 to 0.82 s.
  Use spatial/time interpolation rather than rounding sensors to nodes.
- Clean observations plus three independent Gaussian noise realizations with
  absolute standard deviation 0.02, generated once and shared by all methods.
- Main held-out inlet cases 32-47; validation cases 24-31 only choose a common
  regularization strength. No model weight training or test-based selection.

## Inverse specification
- Seven original piecewise-linear inlet amplitudes, constrained to [0,1.2].
- Equal least-squares observation weighting and the same smoothness prior.
- Three initializations: each model's bounded linear Th fit, constant 0.3,
  constant 0.9. Same candidate regularization strengths and optimizer limits.
- Select one common strength from 0, 0.0001, 0.01 by the pooled validation inlet
  waveform error across methods/layouts. The final test does not select it.
- Report projected gradients, convergence status, initialization and compute time.
  Never remove unconverged or poor cases.
- Verify accelerated GPU rollouts against the frozen CPU implementation before
  interpreting inverse results.

## Evaluation
- Inlet waveform relative L2 on [0,0.82] (and held extension to 1.23 s).
- Volume/time-weighted relative L2 of unobserved Fn on [0,0.82] and [0,1.23].
- Separate clean and noisy results; aggregate cases and noise, retain seed spread.
- Replay recovered inlets through the same resolved three-tube reference to
  distinguish boundary error from compensating forward-model error.
- This remains a synthetic fixed-geometry/fixed-blood experiment. The reference
  is axisymmetric prescribed-flow transport with a mixed junction, not 3D
  bifurcation CFD or measured mouse chemistry.

## Observation-window diagnostic (declared before test evaluation)
Validation-only fits exposed large inlet errors despite accurate forward fields.
The terminal inlet amplitude at 0.82 s has little time to reach midpoint sensors.
Keep the original 0.82 s experiment and additionally use 24 equally spaced
observations through 1.23 s, after the inlet has been held constant for 0.41 s.
Both methods receive exactly the same longer observation window, number of
measurements, noise distribution, starts, bounds and optimization budget.
Choose its regularization independently on the same validation cases and grid.
Report both windows; do not replace or hide the original shorter-window result.
The seven input knots and primary inlet/Fn scoring window remain [0,0.82].
This extension tests acquisition design, not an architecture change or retraining.

## Precision audit
Single-precision runs exposed a small residual projected-gradient floor. Keep
those results, but use double-precision FV integration for the final test tables,
with the frozen graph head at its original float32 precision. Both methods use
the same three starts, at most 100 iterations and a projected-gradient target
of 1e-8. The validation-selected regularization is locked; no model parameters
or acquisition choices are retuned during this numerical precision check.
Repeat reference replays with these final recovered inputs.

## Equal-budget per-model validation tuning
Also let each candidate independently select its own regularization using
the already-completed, identical validation grid. No new test-based tuning
is allowed. Use these stronger, individually tuned baselines for the final
comparison and retain the shared-regularization result as a controlled
secondary experiment. Both observation windows select 0.0001 for ordinary
1D and 0 for each neural seed. Reuse exactly identical recovered-input
reference replays; rerun the reference whenever recovered inputs differ.
