# Full biochemical retraining validation

A fresh public-repository clone and a separate Python environment completed the
entire training, development selection and new-weight evaluation on an RTX 5090.
The original distributed model and all bundled assets remained unchanged.

## Outcome

- Complete pipeline wall time: **691.02 seconds** (environment installation excluded).
- Selected reaction trial: **graph-0.03**, chosen using development data only.
- Retrained transport and reaction tensors identical to the published tensors: **true**.
- Largest change in the compared forward, inverse and reaction-rate error metrics:
  **0** in relative-L2 units.
- Prespecified acceptance tolerance: **0.0005** (0.05 percentage points).
- Agreement with the stricter frozen-evaluation tolerance of 1e-8:
  **true**.
- Published-model reference check: **1.11022302463e-16** maximum difference.

These are new predictions and newly solved inverse optimizations from the retrained
checkpoints, not a display of archived scores. The evaluator records both file hashes
and hashes of the parameter tensors it actually loaded. Different checkpoint file
hashes can occur even for identical tensors because the transport training checkpoint
also contains operator buffers and training metadata.

## Reproduce the complete run

Follow the README installation instructions, then run:

~~~bash
python scripts/retrain_and_evaluate.py --output runs/from-scratch
~~~

The output directory must not already exist. Training uses 16 trajectories and
eight development trajectories. Test labels are downloaded into the run directory
only after checkpoint selection. The eight test inlet trajectories, three inverse
layouts and three noise realizations per noisy case are unchanged.

The transport hidden layers use seed 20261026. Each reaction candidate uses seed
2026102620. Output layers begin at zero, as in the original model. Neither neural
head loads its published trained weights. The reaction stage freezes the newly
trained transport head.

## Forward relative-L2 errors

| Species | Window | Calibrated C1 | Published GG | Retrained GG |
|---|---|---:|---:|---:|
| Th | 0–0.82 s | 4.936041% | 2.142569% | 2.142569% |
| Th | 0–1.23 s | 4.589176% | 2.092453% | 2.092453% |
| Fg | 0–0.82 s | 1.915907% | 1.428596% | 1.428596% |
| Fg | 0–1.23 s | 2.481263% | 1.551800% | 1.551800% |
| Fn | 0–0.82 s | 1.820382% | 0.461219% | 0.461219% |
| Fn | 0–1.23 s | 1.432290% | 0.412420% | 0.412420% |

## Sparse inverse Fn errors, 0–0.82 s

| Layout | Observations | Calibrated C1 | Published GG | Retrained GG |
|---|---|---:|---:|---:|
| one | clean | 4.562787% | 1.816151% | 1.816151% |
| one | noisy | 4.584776% | 4.021311% | 4.021311% |
| two | clean | 1.787512% | 0.678507% | 0.678507% |
| two | noisy | 1.782459% | 0.809038% | 0.809038% |
| three | clean | 1.794524% | 0.652739% | 0.652739% |
| three | noisy | 1.803239% | 0.746118% | 0.746118% |

Each layout has eight clean cases and 24 noisy cases (three noise draws for each
of eight inlets). All 96 inverse cases were solved for each of the three methods.
The complete per-case records, full-window errors and inlet reconstruction errors
are in [evaluation.json](../benchmarks/retraining/evaluation.json).

## GPU training measurements

| Stage | Optimizer updates | Training seconds | Mean GPU utilization | Peak allocated GiB |
|---|---:|---:|---:|---:|
| transport | 1505 | 300.36 | 99.50% | 19.70 |
| graph-0.03 | 1003 + 503 | 102.49 | 98.51% | 19.25 |
| graph-0.3 | 1003 + 503 | 102.52 | 98.31% | 19.25 |
| graph-3 | 1003 + 503 | 102.52 | 99.04% | 19.25 |

Reaction update counts are source fitting plus trajectory fitting. Counts include
warm-up optimizer updates. CUDA Graphs replay full-trajectory batches. Utilization
is sampled during each training stage, including checkpoint validation; it is not
an assertion that data downloads and CPU optimizer control saturate the GPU.
Peak allocated memory is PyTorch's allocation statistic, which differs from
nvidia-smi's process/device memory reservation.

## Records and retrained checkpoints

- [Pipeline record](../benchmarks/retraining/pipeline.json): executed stages, source
  fingerprints, selection, timing, data access order and original-asset checks.
- [Training summary](../benchmarks/retraining/training-summary.json): initialization,
  updates, development selection and GPU measurements.
- [Comparison](../benchmarks/retraining/comparison.json): per-condition metric
  differences and trained-versus-published parameter comparisons.
- [Environment](../benchmarks/retraining/environment.txt): installed dependencies.
- [Download the run archive](https://github.com/woshilry61/gp52-neural-closure/releases/download/retraining-20261026/retraining-20261026.tar.gz): new checkpoints, all training histories,
  stage logs and evaluation records. Its checksum is in
  [archive.json](../benchmarks/retraining/archive.json).

Published logs replace the absolute checkout path with a repository placeholder. Framework
stream-association and scalar-conversion warnings are retained in the logs; the
CUDA Graph runs completed and the trained parameter/metric equivalence checks
passed. No warning filtering was used during training.

The archive deliberately excludes the large prediction arrays and datasets.
Prediction arrays are generated by the pipeline; datasets remain available from
the original v1.0.0 data release.

To evaluate the archived retrained checkpoints after extraction:

~~~bash
python scripts/download_data.py --split test
python -m gp52.evaluate --methods C1 GG retrained --inverse --verify-reference \
  --transport-weights retraining-20261026/transport-retrained.pt \
  --reaction-weights retraining-20261026/reaction-retrained.pt \
  --output outputs/retraining-archive
~~~

## Scope

This validates one same-seed repetition in the recorded hardware/software
environment. It does not measure variability across training seeds or guarantee
bitwise equality on other hardware.

The retraining starts from the released fixed reconstructed blood-flow arrays,
numerical operators, physical settings and CFD-derived labels. It does not
rebuild Sen's blood-flow training, recalibrate the numerical baseline, or regenerate
the full ALE CFD reference. It therefore validates the complete biochemical
train-to-evaluation workflow within this release's documented scope.
