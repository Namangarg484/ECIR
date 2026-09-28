# Frozen-center radius experiment

Dataset: `ml1m`

Validation-selected fixed kappa: `500`

All conditions use the identical learned center within each training
seed. Only the adaptive radius head is trained; fixed and adaptive
conditions share event-, split-, and inference-seed-specific directions.

## Paired hierarchical comparisons

- fixed_kappa_500 - no_expansion: -0.000054, 95% [-0.000225, +0.000110]
- adaptive_hard_clip - fixed_kappa_500: +0.000004, 95% [-0.000046, +0.000071]
- adaptive_smooth - fixed_kappa_500: +0.000011, 95% [-0.000070, +0.000095]
- adaptive_smooth - adaptive_hard_clip: +0.000007, 95% [-0.000052, +0.000073]

Intervals are descriptive with five center-training
seeds; they are not multiplicity corrected.
