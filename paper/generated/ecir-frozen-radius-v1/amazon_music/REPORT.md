# Frozen-center radius experiment

Dataset: `amazon_music`

Validation-selected fixed kappa: `500`

All conditions use the identical learned center within each training
seed. Only the adaptive radius head is trained; fixed and adaptive
conditions share event-, split-, and inference-seed-specific directions.

## Paired hierarchical comparisons

- fixed_kappa_500 - no_expansion: +0.000051, 95% [-0.000439, +0.000513]
- adaptive_hard_clip - fixed_kappa_500: +0.000143, 95% [-0.000140, +0.000487]
- adaptive_smooth - fixed_kappa_500: +0.000141, 95% [-0.000141, +0.000486]
- adaptive_smooth - adaptive_hard_clip: -0.000002, 95% [-0.000107, +0.000076]

Intervals are descriptive with five center-training
seeds; they are not multiplicity corrected.
