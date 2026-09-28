# Matched validation-inference-seed comparison

Dataset: talkplay; selected kappa: 50.
Fixed radius and adaptive epochs use validation seed 3101. Test averages
the original ten paired inference seeds; no model was retrained.
Search spaces and global-fixed/per-center-adaptive selection still differ.
This post-hoc robustness check reuses an already inspected test set.

- fixed_kappa_50 - no_expansion: +0.000172 [-0.000362, +0.000755]
- adaptive_hard_clip - fixed_kappa_50: -0.000003 [-0.000417, +0.000385]
- adaptive_smooth - fixed_kappa_50: -0.000004 [-0.000412, +0.000377]
- adaptive_smooth - adaptive_hard_clip: -0.000001 [-0.000007, +0.000000]

Descriptive crossed-bootstrap 95% intervals; no multiplicity correction.
