# Ten-seed radius selection: lastfm

Global fixed kappa: 500.0
Per-center kappas: {'42': 500.0, '123': 400.0, '456': 500.0, '789': 450.0, '2026': 350.0}

Heads and fixed radii use the same ten validation inference seeds.
Head early stopping also uses the ten-seed mean. Centers remain frozen.
Per-center fixed selection matches the adaptive selection unit, not its search space or compute.
Test uses ten paired inference seeds. Only validation-selected fixed radii are tested.

## NDCG@10

- no_expansion: 0.017421
- fixed_global: 0.017334
- fixed_per_center: 0.017344
- adaptive_hard_clip: 0.017316
- adaptive_smooth: 0.017327

## Paired crossed-bootstrap contrasts

- fixed_global - no_expansion: -0.000088 [-0.000441, +0.000239]
- fixed_per_center - no_expansion: -0.000077 [-0.000444, +0.000264]
- adaptive_hard_clip - fixed_global: -0.000017 [-0.000128, +0.000068]
- adaptive_hard_clip - fixed_per_center: -0.000028 [-0.000129, +0.000046]
- adaptive_smooth - fixed_global: -0.000007 [-0.000112, +0.000083]
- adaptive_smooth - fixed_per_center: -0.000017 [-0.000118, +0.000066]
- adaptive_smooth - adaptive_hard_clip: +0.000011 [-0.000031, +0.000075]
- fixed_per_center - fixed_global: +0.000010 [-0.000045, +0.000079]

Post-hoc robustness analysis on an already examined test set. Intervals are descriptive,
conditional on the selected configurations, and uncorrected for multiple comparisons.
An interval containing zero does not demonstrate equivalence.
