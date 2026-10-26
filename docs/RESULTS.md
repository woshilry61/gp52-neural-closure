# Main evaluation results

All errors are per-inlet weighted relative L2 values averaged over the eight test inlets.

## Forward prediction

| Model | Th primary | Fg primary | Fn primary | Th full | Fg full | Fn full |
|---|---|---|---|---|---|---|
| C1 | 4.936% | 1.916% | 1.820% | 4.589% | 2.481% | 1.432% |
| GG | 2.143% | 1.429% | 0.461% | 2.092% | 1.552% | 0.412% |
| GL | 2.143% | 1.428% | 0.473% | 2.092% | 1.552% | 0.422% |
| LG | 2.426% | 1.565% | 0.502% | 2.536% | 1.818% | 0.460% |
| LL | 2.426% | 1.564% | 0.503% | 2.536% | 1.819% | 0.459% |

## Sparse Fn inference

| Sites | Noise | Window | C1 | GG | GL | LG | LL |
|---|---|---|---|---|---|---|---|
| 1 | clean | primary | 4.563% | 1.816% | 1.827% | 3.294% | 3.298% |
| 1 | noisy | primary | 4.585% | 4.021% | 4.025% | 4.245% | 4.242% |
| 2 | clean | primary | 1.788% | 0.679% | 0.690% | 0.728% | 0.732% |
| 2 | noisy | primary | 1.782% | 0.809% | 0.821% | 0.806% | 0.809% |
| 3 | clean | primary | 1.795% | 0.653% | 0.663% | 0.718% | 0.720% |
| 3 | noisy | primary | 1.803% | 0.746% | 0.756% | 0.771% | 0.773% |
| 1 | clean | full | 11.082% | 4.110% | 4.128% | 11.117% | 11.124% |
| 1 | noisy | full | 12.630% | 12.717% | 12.737% | 13.004% | 13.008% |
| 2 | clean | full | 1.492% | 0.649% | 0.658% | 0.654% | 0.661% |
| 2 | noisy | full | 1.904% | 1.256% | 1.258% | 1.277% | 1.275% |
| 3 | clean | full | 1.488% | 0.641% | 0.648% | 0.653% | 0.659% |
| 3 | noisy | full | 1.886% | 1.231% | 1.231% | 1.257% | 1.254% |

The one-site noisy full window is a remaining failure mode. It must not be hidden by reporting only the primary window.

## Direct physical source

| Model | Primary | Full |
|---|---|---|
| GG | 0.298% | 0.383% |
| GL | 0.312% | 0.405% |
| LG | 0.282% | 0.376% |
| LL | 0.288% | 0.393% |

This source diagnostic uses CFD mean inputs; normal trajectory prediction uses model-generated concentrations.

## Graph contribution

GG and LL have 9,125 biochemical parameters each. The primary Fn means are 0.461% and 0.503%. The GG advantage is positive in 5/8 inlets; the paired 95% interval for LL minus GG spans approximately -0.018 to +0.102 percentage points. This does not establish a robust overall Fn advantage across arbitrary inlets or training seeds.

With graph transport fixed, graph reaction improves primary Fn relative to the local reaction head in all eight test inlets. With local transport fixed, the reaction-architecture difference is smaller and uncertain.

## Runtime and checks

The original warm eight-trajectory benchmark was C1 44.92 ms and GG 48.72 ms on RTX 5090, excluding training and using cached fixed-flow coefficients. It does not demonstrate a speed advantage.

The public-package evaluation took 145.82 s including the requested inversions and output writing. Its maximum reproduced scalar-metric discrepancy was 1.110e-16. This duration is an evaluation-run record, not an isolated inference benchmark.

[Raw reproduced results](../benchmarks/reproduction.json) and [original paired statistics](../benchmarks/reference-results.json) preserve per-case results.
