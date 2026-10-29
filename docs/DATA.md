> Historical single-vessel release. For the current single/three/seven-vessel independent-CFD experiment, see [current model](CURRENT_MODEL.md), [reproduction](UNIFIED_REPRODUCTION.md) and [results](../reports/unified-cfd-baselines-v1/REPORT.md).

# Data and provenance

## Origin

The reference is an independent axisymmetric compliant-wall ALE Navier-Stokes
calculation followed by cross-section-resolved reaction-diffusion-convection.
The archived flow calculation used P2/P1 finite elements. The biochemical reference
used scikit-fem and outputs every 0.001 s. The public package contains the generated
cross-sectional labels needed to reproduce biochemical learning and evaluation;
the optional [raw-input workflow](RAW_REPRODUCTION.md) now provides the project CFD
generator and creates the full finite-element states locally.

The waveform source and reconstructed flow provenance are credited in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). No patient data are included.
The same idealized geometry and true flow condition are shared across all inlet
splits. No new-patient, new-geometry or new-flow generalization is claimed.

The physical reaction-rate label averages the product at finite-element
integration points. It is distinct from the average of the discrete nodal source;
both are released. Training uses the physical label.

## Split

| Partition | Inlets | Seeds |
|---|---:|---|
| Train | 16 | 29009-29012, 826201-826204, 826301-826308 |
| Development | 8 | 863601-863608 |
| Test | 8 | 2026102621-2026102628 |

Training inlet amplitudes were sampled in [0.08,0.9]; development/test amplitudes
in [0,1.2]. The original test references were generated after model selection and
freezing. The later graph ablation reused this already observed test set.
The public release makes all labels visible; it should not be described as an
unseen benchmark for future tuning.

## Downloadable assets

Run from the repository root:

~~~bash
python scripts/download_data.py --split test
python scripts/download_data.py --split all
~~~

Files are release assets rather than Git history. Download sizes and SHA-256
checksums are in [data/manifest.json](../data/manifest.json).

| File | Purpose |
|---|---|
| train.npz | 16 complete training cases |
| development.npz | 8 development cases for selection |
| test.npz | 8 test cases and exact sparse observation values |
| reference-forward.npz | Archived predictions for packaging-equivalence checks |

Each partition uses numeric arrays and can be read with NumPy without pickle:

| Key | Shape | Meaning |
|---|---|---|
| seeds | [B] | Case identifiers |
| amps | [B,7] | True inlet amplitudes |
| y | [B,3,41,616] | CFD mean Th/Fg/Fn on the scoring grid |
| area | [B,41,616] | CFD area used in the error weights |
| reaction_physical | [B,41,616] | Cross-sectional physical reaction rate |
| reaction_discrete | [B,41,616] | Discrete-source diagnostic |
| rate_features | [B,615,41,10] | CFD-derived features for source supervision |
| exact_observations | [B,2,3,24] | Noiseless Th/Fg observations |

The sparse observations were sampled from the original finer-time CFD fields.
They must not be replaced by resampling the already downsampled scoring fields
if exact reproduction of the inverse experiment is intended.

CFD-derived rate features are training/diagnostic inputs only. Normal prediction
constructs features from its own concentration trajectory. Test arrays do not
enter model inference.

## Bundled small assets

The package contains the unchanged biochemical checkpoints, a single example,
the fixed reconstructed flow prepared for the solver, inverse initialization
arrays and the shared observation/regularization settings.

The inverse initialization is a precomputed numerical linear Th response basis
from the original procedure, not a fitted map from test observations to inlets.
The released flow/operator buffers retain the original numerical precision.
Packaging-equivalence checks verify the resulting full fields.

Concentrations use the synthetic model's units and are not a calibrated clinical
assay. Position is in metres and time in seconds. The input diffusivity is a
numerical verification setting.
