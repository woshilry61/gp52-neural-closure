# Protocol and execution record

The shared chemical inlet arrays, 64/16/32 case IDs, observation times and noise, parameter bounds, regularization grid, three inverse starts, stopping rules and scoring windows were fixed before final training and evaluation. The inlet archive and partition files retain their recorded hashes.

Seven vessels were added to the experimental scope at the user's request. Their inlet curves, partitions and observation noise are the same as for the other topologies. The seven-vessel model uses the original three-vessel heads and objectives with an extended adjacency. No additional chemical training cases were introduced.

The following execution changes did not alter the trained architecture or the data/scoring contract:

- A full-batch single-vessel throughput trial was retained separately. Final single-vessel training starts from fresh seeds and uses shuffled batches of 16 from the 64 training cases, with the original number of optimizer updates.
- Seven-vessel CFD assembly and sampling were parallelized across vessels. The short serial and parallel runs produced identical sampled fields.
- Biochemical finite-volume assembly was vectorized. Its matrices, boundary vectors and cell volumes matched the original assembly bitwise on the recorded checks.
- Three-vessel validation using FP32 reaction inference showed projected-gradient stagnation. These validation files are retained in the run's three/validation-fp32-diagnostic directory. Final three-vessel validation was repeated in FP64 before its test evaluation. Seven-vessel evaluation also uses FP64. The neural checkpoint values, equations, optimizer objective and stopping criteria did not change.
- The fixed reaction adjacency is applied sparsely in FP64 inference, with bounded activation batches. Its output agrees with the original dense head; this is an execution change, not another learned model.
- Some seven-vessel reference generation overlapped three-vessel evaluation. Inverse elapsed times therefore serve as execution diagnostics, not a controlled speed benchmark. Neural training jobs ran individually.

All protected models and results from before this experiment are checked by SHA-256 when the final report is generated. This experiment does not replace the earlier final model.
