# Matched validation-inference-seed comparison

Dataset: ml1m; selected kappa: 500.
Fixed radius and adaptive epochs use validation seed 3101. Test averages
the original ten paired inference seeds; no model was retrained.
Search spaces and global-fixed/per-center-adaptive selection still differ.
This post-hoc robustness check reuses an already inspected test set.

- fixed_kappa_500 - no_expansion: -0.000054 [-0.000225, +0.000110]
- adaptive_hard_clip - fixed_kappa_500: +0.000004 [-0.000046, +0.000071]
- adaptive_smooth - fixed_kappa_500: +0.000011 [-0.000070, +0.000095]
- adaptive_smooth - adaptive_hard_clip: +0.000007 [-0.000052, +0.000073]

Descriptive crossed-bootstrap 95% intervals; no multiplicity correction.
