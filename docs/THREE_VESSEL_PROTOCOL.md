# Three-vessel extension protocol

The original public release, Sen baseline, and CCA experiment remain unchanged.
This experiment uses an isolated environment named gp52-three.

1. Reproduce the pinned upstream Sen three-vessel architecture/loss on GPU; verify forward, loss and parameter gradients against unmodified TensorFlow code. Train 32 independent initializations for 100,000 Adam updates; select by training objective only.
2. Drive all biochemical models with the same selected Sen field, followed by an explicit discrete conservation projection. Report the projection size and raw blood errors.
3. Use a Y network with lengths 60, 20, and 34 mm. The biochemical graph has distinct distal outlets; the upstream original graph's distal shortcut is preserved only for faithful blood reproduction.
4. Reference: cross-section-resolved axisymmetric advection-diffusion-reaction within each tube, prescribed parabolic axial profile and continuity-compatible radial flux, connected through a perfectly mixed zero-volume junction. This reference is not a full 3D bifurcation Navier-Stokes CFD simulation.
5. Reduced models: strict one-dimensional concentrations; ordinary finite volume with mean-product reaction; graph-learned two-channel flux memory; graph reaction correction; full model. Radial reference cells are not neural model states.
6. Same molecular diffusivity 2e-4 m2/s, catalytic reaction rate 2, blood, chemistry, initial states, reservoirs, timestep and coarse grid across reduced models. These remain synthetic verification parameters.
7. Fixed inlet seed 2026102801, 48 independent piecewise-linear waveforms, seven amplitudes in [0.05,1.15]. Train 0-23, validation 24-31, held-out test 32-47. This evaluates new chemical inlets on one blood/network condition, not cross-geometry generalization.
8. Fit transport to physical advective covariance flux and reaction to the cross-sectional physical source. No joint trajectory objective that can hide transport error in reaction. Select checkpoints on validation physical moments only.
9. Main grid: 30/10/17 axial cells in the reduced model, dt=0.002 s. Reference initially twice this axial resolution and 12 radial cells. Check finer axial/radial/time resolution on prespecified cases 0 and 32, and report reference uncertainty. Also refine the ordinary 1D baseline.
10. Primary window 0-0.82 s; secondary window 0-1.23 s. Report volume/time-weighted relative L2 Th/Fg/Fn, individual-vessel errors, all 16 paired test cases, conservation, positivity and runtimes. No claim of superiority if reference uncertainty or a stronger numerical baseline changes the ranking.
11. Three-vessel sparse biochemical inversion is not part of this forward-extension benchmark.

## Reference refinement amendment (before final closure training)
The initial 2x axial / 12 radial / 0.002 s reference changed Fn by about 0.75-0.94%
on the two prespecified refinement cases. The final teacher/reference is therefore
4x axial / 24 radial / 0.001 s, sampled at the common 0.002 s reduced-model times.
A further 8x axial / 32 radial / 0.0005 s calculation checks remaining uncertainty.
The unchanged physical-moment training protocol is repeated with three seeds
42, 43, 44. All three held-out results will be retained; no test-based seed selection.
