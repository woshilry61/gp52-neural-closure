# Unified three-vessel reporting, v2

This version changes the primary reporting windows only. It reuses the existing
double-precision, model-specifically tuned inverse results from
runs/inverse-three/long-window/tuned. It does not retrain neural networks,
change the C0 forward model, refit inlet amplitudes, or retune regularization.

## Fixed meaning of each time interval

- The seven inlet amplitudes describe 0–0.82 s. The final value is held after
  0.82 s; the input knots and inlet parameterization remain unchanged.
- Both candidates observe the same Th/Fg measurements at 24 equally spaced
  times from 0.002 through 1.23 s. Fn is never observed by the inverse solver.
- Primary inlet recovery: relative L2 over 0–0.82 s.
- Primary Fn recovery/prediction: volume-weighted relative L2 over 0–1.23 s,
  including every cell in the three-vessel network.
- Preserve the established discrete metric: equal time-sample weights on the
  0.002 s comparison grid, with cell-volume weights for concentration fields.
  Do not change quadrature while changing the reporting window.
- The old 0.82 s observation window and the old Fn 0–0.82 s score remain in
  the archived v1 reports. They do not appear in the new primary table.

This is offline inverse reconstruction using the full acquisition interval,
not a causal prediction issued before all measurements are available.

## Fixed candidates and comparison rules

The baseline is three-c0-fv-v1, as identified by the source/input hashes in
benchmarks/three-vessel-method-lock-v1.json. It is the conservative ordinary
1D finite-volume model with molecular diffusion and mean-product reaction.
The three neural checkpoints (42, 43, 44) are all retained and frozen.

Use precisely the same blood, geometry, grid, time step, chemistry, boundaries,
observations, noise draws, inlet bounds, and inverse search budget. Preserve
the existing validation-only regularization decisions: C0 uses 0.0001 and
the three neural models use 0. The candidate grid was 0, 0.0001, 0.01.
These are per-model choices over the shared validation protocol, not
test-selected choices. This reporting update does not expand that search.

Report one, two and three nested sites: parent midpoint; add daughter 1
midpoint; then add daughter 2 midpoint. Retain clean observations and all
three noisy realizations (absolute Gaussian standard deviation 0.02).
The 16 test inlets are the independent cases; do not count noise realizations
or training seeds as extra independent inlets.

Average per-inlet errors, retain every model seed and every fit, and compute
paired case-clustered bootstrap intervals. The neural mean is an average of
individual model errors, not an ensemble prediction.

## Reproduction and preservation

Run on the experiment host from the existing three-vessel checkout:

~~~bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src \
  .venv/bin/python -m gp52.three.unified_report
~~~

The module reads saved inverse fits and full prediction arrays, recomputes
both primary metrics, checks them against the archived per-case metrics,
and writes reports/three-vessel-inverse-unified-v2.

A before/after hash snapshot protects the previous reports, saved fits,
predictions, reference inputs, baseline implementation and checkpoints.
The published single-vessel C1 and neural checkpoints are also checked
against their existing lock. No model is loaded for training or inference.

This update makes the presentation consistent. It does not claim that C0
is the strongest possible calibrated numerical model, remove the previously
identified regularization/model-discrepancy limitations, or constitute a new
blind test of these already-inspected test cases.
