# Probe redundancy on validation and test

The validation side uses existing selected checkpoints with the same five training and ten inference seeds as the earlier test diagnostic. Directions are generated for validation events using the validation split key. Results are descriptive and post-hoc; they are not used to select any checkpoint, radius, or probe count.

Each Jaccard is the mean pairwise overlap of per-probe top-10 sets. Unique candidates are their union size; merged NDCG uses the original max-score ranker. The validation and test columns have different events, so similarity is a replication check, not a paired effect.
