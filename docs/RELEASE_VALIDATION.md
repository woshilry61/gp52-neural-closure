# Validation of the v1.1.0 publication

This update publishes the completed unified experiment with portable output paths, data downloads and current documentation. It does not retrain or replace the selected models.

## Frozen-model replay

An isolated checkout restored the release archives and ran the published code on an NVIDIA RTX 5090.

- All 32 test inlets were evaluated for C0 and seeds 42, 43 and 44 in each of the three topologies: 384 forward trajectories in total.
- All Th/Fg/Fn error values matched the archived per-case metrics exactly.
- Test case 80 was independently refitted for one, two and three sites, with clean observations and all three noise realizations, for every topology/method: 144 selected inverse fits, each using three initial guesses.
- Inlet and Th/Fg/Fn inverse error values also matched exactly.
- The inverse replay used the already archived validation-selected regularization values. It did not select settings on test errors or repeat the entire validation sweep during this publication check.
- The combined GPU replay took 104.50 seconds, excluding installation, archive creation/restoration and other checks.

The acceptance tolerances were declared before replay: 1e-8 for forward error differences and 5e-4 for inverse error differences, in relative-L2 units. Observed maximum differences were zero.

See [machine-readable validation](../benchmarks/unified-release-validation.json). The standard evaluation command independently reruns validation selection and all 32 test inlet inverse cases; the shorter publication replay is not described as that complete rerun.

## Portable raw-input entry points

The pinned upstream checkout was fetched into the isolated publication checkout. The new three-vessel extractor was compared with the original numerical input arrays. Waveforms match exactly. The largest wall-coefficient difference is 4.66e-10 in absolute units, within 1e-13 relative tolerance, from floating-point evaluation of the same formula.

A two-step three-vessel CFD check at the recorded resolution compared the new input route with the archived CFD trajectory. The largest absolute difference over saved fields is 3.64e-12. This is an entry-point and input-parity check, not a new full-duration blood convergence study.

The original unified training and reference-generation records are retained. No additional full raw-CFD generation and neural training run was performed while packaging this update.

## Preservation and release contents

The old single-vessel model/data assets retain their original SHA-256 identities. The selected current checkpoints and generated numerical arrays are copied unchanged into three independently checksummed release archives. The root data manifest and each archive's member manifest record their identities.

The numerical tables, figures and selected models are preserved. Documentation distinguishes current independent-CFD/C0 results from historical Sen-driven and C1 comparisons. The clean presentation preserves its 20 slide canvases and removes notes and author metadata.

The historical report's source hashes describe the original experiment implementation. Packaging-only changes to run paths, report handling and the public driver are recorded by the new Git commit; they do not replace those historical execution hashes.

The twelve seven-vessel download parts were reassembled through the public downloader into a fresh run directory. The full archive SHA-256 and every extracted file identity matched the original release data.
