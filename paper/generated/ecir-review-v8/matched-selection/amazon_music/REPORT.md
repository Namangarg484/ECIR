# Matched validation-inference-seed comparison

Dataset: amazon_music; selected kappa: 100.
Fixed radius and adaptive epochs use validation seed 3101. Test averages
the original ten paired inference seeds; no model was retrained.
Search spaces and global-fixed/per-center-adaptive selection still differ.
This post-hoc robustness check reuses an already inspected test set.

- fixed_kappa_100 - no_expansion: +0.000256 [-0.000479, +0.001015]
- adaptive_hard_clip - fixed_kappa_100: -0.000063 [-0.000328, +0.000151]
- adaptive_smooth - fixed_kappa_100: -0.000065 [-0.000325, +0.000162]
- adaptive_smooth - adaptive_hard_clip: -0.000002 [-0.000107, +0.000076]

Descriptive crossed-bootstrap 95% intervals; no multiplicity correction.
