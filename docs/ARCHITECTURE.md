> Historical single-vessel release. For the current single/three/seven-vessel independent-CFD experiment, see [current model](CURRENT_MODEL.md), [reproduction](UNIFIED_REPRODUCTION.md) and [results](../reports/unified-cfd-baselines-v1/REPORT.md).

# Model architecture

## Scope and state

The vessel has length 0.06 m and 41 axial cells. Each cell stores cross-sectional
mean Th, Fg and Fn. Th and Fg are transported; Fn is immobile. Two causal memory
variables per mobile species live on faces. They describe unresolved flux history,
not radial subcompartments.

The reconstructed blood flow supplies area A and flow rate Q. The coupling is
one-way: chemistry does not change the flow. The biochemical model is a
neural-closure/conservative-solver hybrid, rather than a network that directly
outputs the entire concentration field.

## Equations

For a transported concentration C,

$$\partial_t(A\bar C)+\partial_x(Q\bar C+J)=A R_C.$$

The flux combines molecular diffusion and learned memory:

$$J=-A D_m\partial_x\bar C+A\sum_k a_k m_k,\quad D_m=2\times10^{-4}.$$

Memory is advected and exponentially relaxed. In the implemented implicit update,
each channel uses a learned amplitude and relaxation time. Eliminating memory
creates the effective diffusion and history terms used by the finite-volume
operator. A single face flux enters adjacent cells with opposite signs.

The reaction is

$$R=2\alpha_\theta\overline{Th}\,\overline{Fg},\quad
R_{Th}=0,\quad R_{Fg}=-R,\quad \partial_t(A\,Fn)=A R.$$

Th is catalytic and is not consumed. Fg consumption and Fn production use the same
discrete reaction amount.

## Transport network

Eight face features encode normalized axial position, area, velocity, three causal
velocity histories, a velocity difference and radius/length. The graph head has
width 32, two residual neighbor-message layers and four outputs: two memory
amplitude/time-scale pairs. It has 4,580 parameters.

The current graph message averages the two axial neighbors. It has no explicit
edge-direction gate or learned edge geometry. The two layers exchange information
over two adjacent grid steps. This is a deliberately small fixed-chain graph head.

Transport coefficients depend on the prescribed flow and its history, not on the
current chemical inlet or concentration. In the fixed-flow benchmark, the same
coefficient field is reused for every inlet. This is a conditional shared model
over inlet curves, not a demonstrated arbitrary-flow operator.

## Reaction predictor-corrector

1. Predict Th/Fg/Fn using the learned transport and the mean-product reaction.
2. Build reaction features from that predicted trajectory.
3. Predict a positive multiplier, alpha = exp(0.25 tanh(head)).
4. Correct Fg and Fn conservatively using that multiplier.

The ten reaction features are Th, Fg, their product, radius-scaled gradients of Th
and Fg, area, velocity, a normalized gradient-product descriptor and two causal
gradient-history descriptors. The history time scales are 0.02 and 0.1 s.

The reaction graph has 4,545 parameters. Prediction uses the model's own states.
CFD concentrations are used only as training labels and for the separate direct
physical-source diagnostic.

Cross-sectional averaging generally gives
mean(Th Fg) = mean(Th) mean(Fg) + covariance(Th,Fg). The reaction closure approximates
the unresolved correction. Axial neighbors cannot directly reveal an arbitrary
radial concentration distribution.

## Training

The transport loss is the mean normalized, area/time-weighted squared trajectory
error for all three species. Each update contains all 16 training inlets.
Development full-window error selects checkpoints.

The reaction stage first fits the physical reaction-rate labels, then optimizes
complete self-predicted Fg/Fn trajectories plus an auxiliary source loss.
Three weights (0.03, 0.3, 3) are compared using development data only. Selection also
requires direct physical source error to improve over the mean-product closure.

The full time-parallel scan is a different evaluation order of the same discrete
linear evolution. It is used to expose large GPU batches during training.
Deployment uses independent sequential Triton kernels. The package checks the
two implementations against one another.

## Pointwise controls

The pointwise MLP uses widths 32-64-32, three tanh activations, no first-layer bias
and a zero-initialized output head. Its parameter count exactly matches the graph
head for each module. Every parameter is active. The internal wiring differs, so
this is an equal-parameter architecture comparison, not a single-edge intervention.

All controls retain the same gradient/history features and physical spatial
coupling. GG and GL share the graph transport checkpoint; LG and LL share the
pointwise transport checkpoint. Reaction heads are adapted to their respective
transport model using the same development protocol.

## Sparse inverse

The seven inlet amplitudes lie in [0,1.2]. Only Th/Fg observations enter the
objective. All methods use the same observations, scaling, regularization,
bounds and three initializations. A bounded damped least-squares iteration is
followed by the same SciPy trust-region refinement when needed.

The neural weights are frozen. Fn is predicted after inferring the inlet.
Convergence of the optimizer does not imply that the inlet is identifiable from
a single noisy observation site.
