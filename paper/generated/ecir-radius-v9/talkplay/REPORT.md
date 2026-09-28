# Ten-seed radius selection: talkplay

Global fixed kappa: 500.0
Per-center kappas: {'42': 500.0, '123': 500.0, '456': 450.0, '789': 450.0, '2026': 500.0}

Heads and fixed radii use the same ten validation inference seeds.
Head early stopping also uses the ten-seed mean. Centers remain frozen.
Per-center fixed selection matches the adaptive selection unit, not its search space or compute.
Test uses ten paired inference seeds. Only validation-selected fixed radii are tested.

## NDCG@10

- no_expansion: 0.024609
- fixed_global: 0.024771
- fixed_per_center: 0.024775
- adaptive_hard_clip: 0.024778
- adaptive_smooth: 0.024777

## Paired crossed-bootstrap contrasts

- fixed_global - no_expansion: +0.000163 [-0.000125, +0.000520]
- fixed_per_center - no_expansion: +0.000167 [-0.000133, +0.000534]
- adaptive_hard_clip - fixed_global: +0.000006 [-0.000004, +0.000050]
- adaptive_hard_clip - fixed_per_center: +0.000002 [-0.000059, +0.000066]
- adaptive_smooth - fixed_global: +0.000006 [-0.000004, +0.000046]
- adaptive_smooth - fixed_per_center: +0.000002 [-0.000065, +0.000067]
- adaptive_smooth - adaptive_hard_clip: -0.000001 [-0.000007, +0.000000]
- fixed_per_center - fixed_global: +0.000004 [-0.000052, +0.000068]

Post-hoc robustness analysis on an already examined test set. Intervals are descriptive,
conditional on the selected configurations, and uncorrected for multiple comparisons.
An interval containing zero does not demonstrate equivalence.
