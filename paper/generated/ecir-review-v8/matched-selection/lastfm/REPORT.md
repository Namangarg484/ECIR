# Matched validation-inference-seed comparison

Dataset: lastfm; selected kappa: 500.
Fixed radius and adaptive epochs use validation seed 3101. Test averages
the original ten paired inference seeds; no model was retrained.
Search spaces and global-fixed/per-center-adaptive selection still differ.
This post-hoc robustness check reuses an already inspected test set.

- fixed_kappa_500 - no_expansion: -0.000088 [-0.000436, +0.000238]
- adaptive_hard_clip - fixed_kappa_500: -0.000043 [-0.000186, +0.000067]
- adaptive_smooth - fixed_kappa_500: -0.000033 [-0.000163, +0.000072]
- adaptive_smooth - adaptive_hard_clip: +0.000010 [-0.000049, +0.000085]

Descriptive crossed-bootstrap 95% intervals; no multiplicity correction.
