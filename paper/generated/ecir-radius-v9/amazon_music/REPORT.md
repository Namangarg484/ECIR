# Ten-seed radius selection: amazon_music

Global fixed kappa: 500.0
Per-center kappas: {'42': 20.0, '123': 50.0, '456': 100.0, '789': 250.0, '2026': 400.0}

Heads and fixed radii use the same ten validation inference seeds.
Head early stopping also uses the ten-seed mean. Centers remain frozen.
Per-center fixed selection matches the adaptive selection unit, not its search space or compute.
Test uses ten paired inference seeds. Only validation-selected fixed radii are tested.

## NDCG@10

- no_expansion: 0.051729
- fixed_global: 0.051780
- fixed_per_center: 0.051892
- adaptive_hard_clip: 0.051953
- adaptive_smooth: 0.051871

## Paired crossed-bootstrap contrasts

- fixed_global - no_expansion: +0.000051 [-0.000444, +0.000519]
- fixed_per_center - no_expansion: +0.000163 [-0.000600, +0.000932]
- adaptive_hard_clip - fixed_global: +0.000173 [-0.000200, +0.000581]
- adaptive_hard_clip - fixed_per_center: +0.000061 [-0.000131, +0.000270]
- adaptive_smooth - fixed_global: +0.000091 [-0.000452, +0.000596]
- adaptive_smooth - fixed_per_center: -0.000021 [-0.000276, +0.000212]
- adaptive_smooth - adaptive_hard_clip: -0.000081 [-0.000409, +0.000151]
- fixed_per_center - fixed_global: +0.000112 [-0.000288, +0.000556]

Post-hoc robustness analysis on an already examined test set. Intervals are descriptive,
conditional on the selected configurations, and uncorrected for multiple comparisons.
An interval containing zero does not demonstrate equivalence.
