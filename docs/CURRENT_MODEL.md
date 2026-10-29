# Current model: physical solver and learned closures

## What is being predicted?

At every axial cell and time, the model predicts cross-sectional mean thrombin H (Th), fibrinogen G (Fg) and fibrin F (Fn). Blood cross-sectional area A and flow Q are supplied inputs. Blood is held fixed during biochemical training and inlet reconstruction.

The reference solves blood motion independently, then resolves radial biochemical states. The candidate models use one concentration per species per axial cell. Memory variables describe unresolved transport history; they are not radial concentration bins.

## Conservation equations

For the mobile species, the one-dimensional conservation law has the form

$$
\frac{\partial(A\bar c)}{\partial t}
+\frac{\partial J_c}{\partial x}
=A\bar R_c,
\qquad
J_c=Q\bar c-AD\frac{\partial\bar c}{\partial x}+J_c^{\mathrm{closure}}.
$$

A is cross-sectional area, Q is volume flow, D is the prescribed molecular diffusivity and J is total axial transport. Finite volumes exchange fluxes across cell faces. Junctions enforce conservation with perfect mixing.

The simplified reaction is

$$
\bar R_H=0,\quad
\bar R_G=-2\kappa\bar H\bar G,\quad
\bar R_F=+2\kappa\bar H\bar G.
$$

Th is catalytic. Fg is consumed, and Fn accumulates without axial transport. Fn does not feed back into blood motion. These are normalized verification settings, not patient-calibrated coagulation kinetics.

## C0: the fixed ordinary numerical baseline

C0 sets the additional transport closure to zero and the reaction multiplier to one. It solves conservative one-dimensional transport with molecular diffusivity D = 2e-4 m²/s and the mean-product reaction 2HG.

The single-vessel implementation couples implicit transport and reaction. The branched implementations apply an exact local reaction step after transport. These pre-existing time discretizations are retained and documented, rather than being claimed identical.

C1 is the separately calibrated historical single-vessel numerical closure. C1 scores in the older release must not be mixed with the current C0 table.

## What the two neural modules learn

**Transport module.** A graph head uses local blood/geometry features and neighbouring cells to produce coefficients for two temporal flux-memory channels. These channels approximate transport information lost by cross-sectional averaging, including correlation between axial velocity and concentration.

**Reaction module.** A second graph head uses concentration, gradient and history features to predict a bounded multiplier. Its form is

$$
\kappa=\exp\left(0.25\tanh z_\theta\right),
$$

so it stays between approximately 0.779 and 1.284. It corrects the fact that the mean of HG generally differs from the product of mean H and mean G. The physical solver uses the same rate with opposite signs for Fg and Fn.

The single-vessel architecture and branched architecture have different implementations. Three and seven vessels share the same type of heads and losses, with topology-specific adjacency, buffers and separately trained parameters. This is not a single model trained to transfer across all vessel counts.

## Training

All topologies use case IDs 0–63 for training, 64–79 for validation and 80–111 for testing. Seeds 42, 43 and 44 produce three separate fits.

For the single vessel, the transport head is trained through concentration rollouts. The reaction head first fits physical reaction labels and then concentration trajectories, using the frozen trained transport head. Three source-loss weights are selected on validation data.

For three and seven vessels, the transport head fits reference advective covariance labels, while the reaction head fits physical mean reaction rates. Both heads use all 64 training cases. Validation selects checkpoints.

The conservative solver remains part of every prediction. A neural head does not directly replace the entire concentration field with an unconstrained output.

## Sparse inverse reconstruction

A known-inlet forward prediction starts with seven amplitudes defining the Th inlet curve. In the inverse problem those amplitudes are unknown.

The procedure observes Th and Fg at one, two or three declared sites and 24 times. It varies the seven amplitudes to reduce observation mismatch plus an inlet regularization term. All neural weights remain frozen. Fn is withheld from the observations and is predicted using the recovered inlet.

C0 and the neural model use identical observation values, noise, bounds, three initial guesses and stopping rules. Each method selects regularization from the same validation grid. A low Fn error does not automatically imply an accurately recovered inlet.
