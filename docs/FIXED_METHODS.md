# Current comparison: unified independent CFD and fixed C0

The current single-, three- and seven-vessel experiment uses **C0** on identical independent CFD blood within each topology. See [the current protocol](UNIFIED_REPRODUCTION.md). The sections below preserve the historical method locks; their Sen-driven and C1 results are not the current unified scores.

# Fixed method identities and retraining policy

This document resolves baseline naming. It freezes the current forward methods;
it does not declare that the inverse configuration or baseline has been fully
optimized. No neural retraining is required merely to clarify or strengthen a
comparison.

## Current three-vessel experiment

Use **three-c0-fv-v1**, displayed as **Ordinary 1D finite volume, C0**, as the
fixed basic numerical baseline. It has conservative advection, molecular
diffusion and the mean-product catalytic reaction, with no learned correction
and no additional memory closure.

Keep the same Sen-derived projected blood, boundary/initial conditions,
30/10/17 axial cells, 0.002 s step and perfectly mixed junction. Molecular
diffusion is 2e-4 m2/s and reaction is 2 Th Fg. These definitions are not
changed to obtain a preferred ranking.

The neural comparators are the existing three-vessel checkpoints for seeds
42, 43 and 44. Keep all three; do not pick the best test seed.

The current forward and inverse results remain valid records of this specific
comparison. Calling C0 the fixed basic baseline does not establish that it is
the strongest possible calibrated numerical model.

## Published single-vessel experiment

Keep **single-c1-published**, displayed as **Calibrated numerical closure, C1**,
with its published five parameters. Keep the published graph transport and
reaction checkpoints unchanged.

C1 is a different closure from the three-vessel C0. Its published scores
belong to the single-vessel experiment. Do not combine these two numerical
methods under an unlabeled "numerical baseline", or copy single-vessel
calibration coefficients into another geometry without validation.

## What is frozen, and what still needs reevaluation

Frozen now:
- Forward equations, discretization, physical/boundary settings and named
  baseline identity within each experiment.
- Existing neural weights and published C1 coefficients.
- Previous reports and all completed inverse runs.

A revised inverse evaluation may change observation weighting or how its
regularization strength is selected, provided the new protocol is explicit,
validation-only and applied consistently to both candidates. Such a change
requires new inverse fits and result tables, not neural-network retraining.
Estimating seven inlet amplitudes is not training the biochemical network.

The completed inverse protocol is v1. Any changes motivated by the fairness
audit must be recorded as a separate evaluation version. Do not overwrite
v1 or choose inverse settings based on test error. Existing test cases have
already been inspected and should not be described as a new blind evaluation.

If a stronger calibrated traditional closure is subsequently introduced for
the three-vessel experiment, give it a separate method/version and preserve
C0 as the fixed basic baseline. It must not silently replace C0 in earlier
claims or tables.

## When retraining is actually warranted

Retraining is a separate decision if the neural architecture or closure
equations change, if new blood/geometry conditions are intended to enter the
learned distribution, or if a confirmed implementation defect invalidates
the existing weights. None of those changes is being made in this lock.

A baseline change, a new inverse regularization choice, or an additional
measurement site alone does not invalidate the existing neural weights.
Evaluate the frozen model first; do not retrain it merely to restore a win.

## Integrity record

benchmarks/three-vessel-method-lock-v1.json records the exact commits,
definitions, published C1 coefficients, and SHA256 identities of the relevant
source, flow, reference and checkpoint files. It is an identity record, not
a new accuracy result.

## Current unified result table

Reporting version v2 uses observations through 1.23 s for both fixed C0 and
the frozen neural candidates. It scores inlet recovery on 0–0.82 s and Fn
on 0–1.23 s. This is a re-summary of the existing long-window fits, without
new inverse optimization, regularization selection or neural training.

See [the v2 protocol](THREE_VESSEL_UNIFIED_REPORT_V2.md) and
[the updated report](../reports/three-vessel-inverse-unified-v2/REPORT.md).
