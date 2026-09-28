# Ten-seed radius selection: ml1m

Global fixed kappa: 500.0
Per-center kappas: {'42': 150.0, '123': 400.0, '456': 500.0, '789': 500.0, '2026': 20.0}

Heads and fixed radii use the same ten validation inference seeds.
Head early stopping also uses the ten-seed mean. Centers remain frozen.
Per-center fixed selection matches the adaptive selection unit, not its search space or compute.
Test uses ten paired inference seeds. Only validation-selected fixed radii are tested.

## NDCG@10

- no_expansion: 0.018642
- fixed_global: 0.018588
- fixed_per_center: 0.018600
- adaptive_hard_clip: 0.018609
- adaptive_smooth: 0.018598

## Paired crossed-bootstrap contrasts

- fixed_global - no_expansion: -0.000054 [-0.000224, +0.000116]
- fixed_per_center - no_expansion: -0.000043 [-0.000305, +0.000175]
- adaptive_hard_clip - fixed_global: +0.000021 [-0.000086, +0.000135]
- adaptive_hard_clip - fixed_per_center: +0.000009 [-0.000124, +0.000180]
- adaptive_smooth - fixed_global: +0.000010 [-0.000093, +0.000116]
- adaptive_smooth - fixed_per_center: -0.000002 [-0.000141, +0.000182]
- adaptive_smooth - adaptive_hard_clip: -0.000011 [-0.000076, +0.000038]
- fixed_per_center - fixed_global: +0.000012 [-0.000152, +0.000150]

Post-hoc robustness analysis on an already examined test set. Intervals are descriptive,
conditional on the selected configurations, and uncorrected for multiple comparisons.
An interval containing zero does not demonstrate equivalence.
