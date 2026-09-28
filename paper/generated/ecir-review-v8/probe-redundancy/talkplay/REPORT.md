# Probe redundancy diagnostics

CSV means average all test events, five training seeds, and ten inference seeds.
Top-k ties use ascending catalog index after full-history exclusions.
Pairwise Jaccard averages all probe pairs, not only adjacent probes.
Unique candidates and target union coverage refer to the union of per-probe top-k sets;
this union can contain S*k items and is not fixed-budget Recall@k.
merged_recall and merged_ndcg10 instead use the original max-score ranker.
target_gain_last_probe compares S with S-1; target_gain_vs_first compares S with 1.
Angles/cosines measure query-vector geometry, not retrieved-item geometry.
S=1 pairwise statistics are undefined. These are descriptive post-hoc diagnostics,
not significance tests, causal explanations, or hyperparameter-selection evidence.
Every seed/count NDCG was checked against the retained sensitivity sweep.

## Top-10 diagnostics

| Method | S | Jaccard | Unique candidates | Union hit | Merged Recall@10 | Angle (deg) |
|---|---:|---:|---:|---:|---:|---:|
| learned_fixed | 1 | undefined | 10.000 | 0.046160 | 0.046160 | undefined |
| learned_fixed | 2 | 0.8256934598734668 | 11.008 | 0.048020 | 0.046200 | 8.064959388919105 |
| learned_fixed | 5 | 0.825639434398933 | 12.191 | 0.050580 | 0.046040 | 8.0669927466402 |
| learned_fixed | 10 | 0.8255258075998071 | 12.997 | 0.052300 | 0.046280 | 8.066829578662968 |
| learned_fixed | 20 | 0.8254607563468523 | 13.743 | 0.053780 | 0.046220 | 8.067215038342338 |
| full | 1 | undefined | 10.000 | 0.047300 | 0.047300 | undefined |
| full | 2 | 0.9121418981019059 | 10.490 | 0.048140 | 0.047400 | 3.6187439084061856 |
| full | 5 | 0.9119963010322976 | 11.036 | 0.049120 | 0.047160 | 3.6196548803342097 |
| full | 10 | 0.9119422544862549 | 11.389 | 0.049800 | 0.047280 | 3.6195816565162353 |
| full | 20 | 0.9118456755174654 | 11.706 | 0.050300 | 0.047360 | 3.619754368182909 |
