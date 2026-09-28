# ECIR additional experiment report

All inputs and source hashes were verified before aggregation. V-SKNN
hyperparameters were chosen on validation NDCG@10. Full effectiveness
is averaged over all declared training and inference seeds before paired
cluster bootstrap comparisons. Efficiency is inference-only, and the
κ/radius analysis uses validation queries only.

Generated artifacts:

- `session_knn.csv` and `session_knn.tex`
- `paired_session_knn.csv`
- `efficiency.csv` and `efficiency.tex`
- `kappa_diagnostics.csv` and `kappa_diagnostics.tex`

The paired intervals are exploratory and not multiplicity-corrected.
