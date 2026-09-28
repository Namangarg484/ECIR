# Frozen-center radius experiment

Dataset: `lastfm`

Validation-selected fixed kappa: `500`

All conditions use the identical learned center within each training
seed. Only the adaptive radius head is trained; fixed and adaptive
conditions share event-, split-, and inference-seed-specific directions.

## Paired hierarchical comparisons

- fixed_kappa_500 - no_expansion: -0.000088, 95% [-0.000436, +0.000238]
- adaptive_hard_clip - fixed_kappa_500: -0.000043, 95% [-0.000186, +0.000067]
- adaptive_smooth - fixed_kappa_500: -0.000033, 95% [-0.000163, +0.000072]
- adaptive_smooth - adaptive_hard_clip: +0.000010, 95% [-0.000049, +0.000085]

Intervals are descriptive with five center-training
seeds; they are not multiplicity corrected.
