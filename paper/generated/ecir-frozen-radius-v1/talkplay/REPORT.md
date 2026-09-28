# Frozen-center radius experiment

Dataset: `talkplay`

Validation-selected fixed kappa: `500`

All conditions use the identical learned center within each training
seed. Only the adaptive radius head is trained; fixed and adaptive
conditions share event-, split-, and inference-seed-specific directions.

## Paired hierarchical comparisons

- fixed_kappa_500 - no_expansion: +0.000163, 95% [-0.000131, +0.000530]
- adaptive_hard_clip - fixed_kappa_500: +0.000006, 95% [-0.000005, +0.000050]
- adaptive_smooth - fixed_kappa_500: +0.000005, 95% [-0.000005, +0.000046]
- adaptive_smooth - adaptive_hard_clip: -0.000001, 95% [-0.000007, +0.000000]

Intervals are descriptive with five center-training
seeds; they are not multiplicity corrected.
