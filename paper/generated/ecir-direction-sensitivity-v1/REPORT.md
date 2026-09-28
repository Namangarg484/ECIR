# TalkPlay query-direction sensitivity

This is a post-hoc, inference-only sensitivity analysis using frozen
learned-fixed and ACE checkpoints. The test results are not used to
select or retrain a model.

## Seed-aware S=5 comparison

ACE - learned fixed: +0.000526, hierarchical 95% interval 
[-0.000494, +0.001618]. The difference is positive for 
4/5 training seeds.
The interval jointly resamples training seeds, paired inference seeds,
and session/user clusters; it is exploratory and not multiplicity-corrected.

## Generated files

- `direction_sensitivity.csv`: all S-by-method metric summaries
- `seed_differences_s5.csv`: per-training-seed paired differences
- `seed_aware_comparison_s5.csv`: hierarchical interval
- `direction_sensitivity.tex`: compact paper table
- `direction_sensitivity.pdf` and `.png`: sensitivity plot
