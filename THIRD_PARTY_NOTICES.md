# Attribution and provenance

The MIT license covers project-authored implementation, documentation, biochemical
checkpoints and generated numerical artifacts distributed in this release. It does
not relicense external research software, papers or datasets.

## Blood-flow framework

The research uses reconstructed haemodynamics from:
Ahmet Sen et al., *Physics-Informed Graph Neural Networks to Solve 1-D Equations
of Blood Flow*, Computer Methods and Programs in Biomedicine 257 (2024), 108427.
DOI: https://doi.org/10.1016/j.cmpb.2024.108427

Upstream: https://github.com/ahmetsenemse/PIGNN-1D-Blood-Flow-Simulation
Pinned research revision: bf0ec89f0aab8a639bdd2d363dc3c0cfae770c1d.

That revision does not contain an explicit license. This release therefore does
not redistribute its source code, original waveform archive or trained blood-flow
network. The compact flow asset contains numerical arrays produced by the project's
blood-flow reconstruction and preparation pipeline. The biochemical implementation
can run from these arrays without importing the upstream software.

The original single-vessel boundary-waveform source was
PIGNN_1_artery/Snaps/sim_1_1.his. It informed both the project-generated independent
CFD reference and the blood-flow reconstruction. The reconstructed input field is
not used as the CFD ground truth. Obtain upstream materials directly from their
authors under their applicable terms if rebuilding that stage.

The default public example supports biochemical inference and retraining from
prepared inputs. The optional raw-input workflow fetches this upstream revision
directly, regenerates the independent ALE CFD reference, and trains the project's
conservative PyTorch blood model from sparse CFD observations. This blood model
adapts the Sen graph architecture with spline time coefficients and exact discrete
continuity; it is not the unchanged upstream TensorFlow training experiment.
No upstream source, waveform or original blood checkpoint is added to our Git
history or release archives. See docs/RAW_REPRODUCTION.md.

## Modelling references

- Masri, Puelz and Riviere (2021), *A reduced model for solute transport in compliant
  blood vessels with arbitrary axial velocity profile*, International Journal of
  Heat and Mass Transfer 176, 121379.
- Masri, Puelz and Riviere (2022), *A Discontinuous Galerkin Method for Blood Flow and
  Solute Transport in One-Dimensional Vessel Networks*, Communications on Applied
  Mathematics and Computation 4, 500-529.
- Melidoro et al. (2026), *Sensitivity analysis of factors in a microfluidics CFD
  model of coagulation and cardiac applications*.
  https://doi.org/10.1007/s10237-025-02039-1

These references motivate the mathematical model. Their article PDFs and course
materials are not included.

## Software dependencies

PyTorch, Triton, NumPy, SciPy and Matplotlib are installed separately and retain
their respective licenses. No dependency source trees or virtual environments are
bundled. The optional CFD generator uses scikit-fem 12.0.2; it is not required for
the released biochemical inference and evaluation commands.

## Unified independent-CFD release

The v1.1.0 archives distribute project-generated CFD and biochemical arrays and project-trained biochemical checkpoints for one, three and seven vessels. Their blood fields are solved independently from the pinned numerical boundary/initial snapshots. They do not contain the upstream source tree, original snapshot files or original blood-network weights. The raw workflow fetches upstream materials from their authors.
