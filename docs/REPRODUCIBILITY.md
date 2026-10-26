# Reproduction and verification

## Levels of reproduction

1. **Bundled example:** installation plus the frozen model and one CFD example.
2. **Forward and inverse evaluation:** download test.npz and rerun inference and
   inlet optimization.
3. **Frozen-field equivalence:** additionally download reference-forward.npz.
4. **Biochemical retraining:** additionally download train.npz and development.npz.

Rebuilding the upstream blood-flow PIGNN and the complete ALE CFD reference is
outside this public package. The generated input and label arrays make levels
1-4 independently executable without access to the original research machine.

## Environment

Follow the README installation sequence. The tested top-level versions are pinned
in pyproject.toml; requirements-tested.txt records the complete verification
environment. Use the CUDA 12.8 PyTorch index before installing the package.

All numerical work runs on a CUDA GPU. NumPy-based scoring, compressed file I/O and
bounded optimizer control also use the CPU. The archive CPU checks require no GPU:

~~~bash
python -m unittest discover -s tests -v
~~~

## Full evaluation

~~~bash
python scripts/download_data.py --split test
python -m gp52.evaluate --methods C1 GG GL LG LL --inverse --verify-reference --output outputs/reproduction
~~~

Each model is evaluated on all eight inlets. Each inverse layout has eight clean
cases and 24 noisy cases. Noise uses a deterministic per-inlet/per-replicate seed,
standard deviation 0.02, and the same realizations for every method.

The primary window is 0-0.82 s; the full window is 0-1.23 s. The relative L2 norm
uses CFD area and trapezoidal time weights. Errors are computed per inlet before
averaging. The original statistical intervals resample eight inlet-level pairs;
noise repeats are averaged within each inlet first.

The comparison tolerance for reproduced scalar metrics is 1e-8 absolute.
This allows floating-point/optimizer differences while remaining far below the
reported method differences. Passing this tolerance is not a universal guarantee
across all future CUDA and library releases.

## Field and physical checks

~~~bash
python scripts/download_data.py --split reference-forward
python -m gp52.verify
~~~

This checks unchanged artifact hashes, archived forward fields, parameter counts,
nonnegative concentrations, mass balance, causal dependence on the inlet and
time-parallel versus sequential propagation. Verification reports are written to
outputs/ and do not overwrite the published weights.

## Training

The README commands retrain the graph transport and all three graph-reaction
source-loss weights. Selection uses development scores and source eligibility.
The selection script requires all three candidate records. Test labels are never
used to select checkpoints.

For pointwise controls use --kind local in the transport trainer, or --kind local
in the reaction trainer. To pair a reaction head with pointwise transport, also
pass --transport-kind local and its transport checkpoint. For polynomial reaction
use --kind polynomial.

The released model uses one initialization per architecture. Repeated seeds are
needed to quantify training variability; the reported inlet bootstrap intervals
do not include that uncertainty.

Short training smoke checks verify that the packaged loss, backward pass, optimizer
and checkpoint saving execute. They are not new trained model releases. The
distributed biochemical checkpoint hashes remain identical to the original final
version.

## Output and asset records

- benchmarks/reference-results.json: all graph/pointwise test records and paired statistics.
- benchmarks/published-results.json: original final-model and baseline evaluation.
- benchmarks/reproduction.json: independent public-package evaluation.
- benchmarks/verification.json: field and physical checks.
- benchmarks/artifact-manifest.json: distributed asset checksums.
- data/manifest.json: downloadable data checksums.

Run directories and predictions are ignored by Git. Keep retraining outputs
separate from the distributed frozen assets.
