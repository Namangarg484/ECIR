# Training-item support and conditional test effectiveness

Warm means the target occurs in the prepared training context or target of at least one event; cold means it never occurs in training interactions. Validation/test items do not define support. The catalog and eligible candidate set are unchanged in all metrics.

`target_coverage.csv` reports event-weighted coverage on both held-out splits. `warm_cold_metrics.csv` reports the original saved full-catalog rankings, partitioned by test-target support. Each method's repeated files are averaged equally; every file must have identical events. Empty partitions have blank metric cells. This is a descriptive post-hoc analysis; no model is selected or evaluated again.
