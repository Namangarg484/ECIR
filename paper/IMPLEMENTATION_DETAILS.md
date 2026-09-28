# Implementation details accompanying the ECIR manuscript

This artifact document retains implementation detail omitted from the compact
manuscript. The evaluated sources, checkpoints, splits, and numerical result
files are unchanged by the manuscript revision.

## Metadata-weighted center

Each catalog item is a document `D_j`. The query is the concatenation of all
context-item metadata, retaining repeated terms. The tokenizer is lowercased
`\w+`, without stemming or stop-word removal. BM25 uses `k1 = 1.5`, `b = 0.75`,
full-catalog document frequencies, and full-catalog mean document length.

For a context item `i`, its score is

\[
s_i=\sum_{t\in\operatorname{supp}(D_i)}qf(t)
\log\!\left(1+\frac{N-df(t)+0.5}{df(t)+0.5}\right)
\frac{tf_i(t)(k_1+1)}{tf_i(t)+k_1(1-b+b|D_i|/\overline{|D|})}.
\]

The sum visits distinct document terms; query and document frequencies retain
multiplicity. Normalize by the sum of context-item scores. If all scores are
zero, use uniform context weights. Form and normalize the weighted sum of
fixed item embeddings. No target metadata is added to the context query.

Metadata fields are TalkPlay title/artist, MovieLens title/genres, Last.fm
artist name, and Amazon title/store text before the `Format:` suffix.
The historical artifact identifier is `weighted_prf`.

## V-SKNN comparator

For each training group, reconstruct the longest available training prefix.
Historical sessions are binary item-incidence vectors. For active context `A`,
the most recent occurrence of item `i` receives weight
`lambda ** (len(A) - 1 - last_position_A(i))`. Cosine similarity between this
weighted context and historical binary vectors selects the top `K` overlapping
neighbors.

For candidate `j` in historical session `s`, add
`similarity(A, s) * lambda ** (len(s) - 1 - last_position_s(j))`.
Sum neighbor votes without normalization. A `1e-12` multiple of max-normalized
training-item frequency provides popularity backoff. The common evaluator
masks the complete observed history and breaks remaining ties by canonical
catalog index.

Validation searches `K = {50, 100, 250, 500}` and
`lambda = {0.7, 0.85, 1}`. Selected pairs are `(100, .85)` for TalkPlay,
`(250, .85)` for MovieLens, `(50, 1)` for Last.fm, and `(50, 1)` for Amazon.

All prefixes with the active group identifier are excluded. This avoids
retrieving the current group's training-derived prefix as a neighbor. On
per-user datasets, it also discards legitimate same-user history. The
manuscript therefore describes this as a conservative comparator choice,
not an unavoidable leakage restriction or an optimal personalized baseline.
Warm-target status is global occurrence in prepared training contexts/targets;
it does not imply occurrence in a retrieved or eligible V-SKNN neighbor.

## Center constraint and diagnostics

The evaluated learned center is `normalize(q + g*u)`, with unit `q,u` and
`0 < g < .2`. A ray tangent to the radius-`g` ball centered on `q` gives
`angle(q, mu) <= arcsin(g) < 11.54 degrees`. The source retains the
nonstandard `r + FFN(h)` residual. Neither its optimality nor robustness to
replacing it with `h + FFN(h)` is established.

The complete radius table, boundary fractions, coherence/length correlations,
and definitions remain in `generated/ecir-additional-v1/kappa_diagnostics.csv`
and the corresponding report. Training and inference dispersion remain in
`generated/ecir-mps-v2/metrics.csv`. They were not removed from the evidence
package when their manuscript presentation was shortened.

The full six-row synchronized Apple-MPS efficiency table, including scoring
latency, median/p95 total latency, and sampled accelerator allocation for the
learned-center, learned+fixed, and adaptive variants on TalkPlay and Amazon,
remains in `generated/ecir-additional-v1/efficiency.csv`. Sampled live
allocation is not a high-water memory measurement. Batch size was 128 with
three warm-ups and ten measured, synchronized float32 batches. The manuscript
retains the main scoring/total-time contrast and machine description.

## Completed reviewer diagnostics

`generated/ecir-review-v10/target-support/` records validation/test target
support and unchanged full-catalog test rankings partitioned into warm/cold
events. It includes all eleven configurations. These are descriptive subsets,
without new tuning or causal attribution.

`generated/ecir-review-v10/validation-probe-redundancy/talkplay/` compares the
validation diagnostic with the earlier test diagnostic. Validation uses all
1,520 events, five training seeds, ten inference seeds, and 1/5/20 probes.
Validation/test agreement is post-hoc robustness evidence, not independent
confirmation; the checkpoints were already selected on validation.

`generated/ecir-radius-v9/*/metrics.csv` contains the frozen-center secondary
metrics. The manuscript's small Recall@10/MRR@50 effect bounds summarize point
estimates only. Crossed-bootstrap intervals in this package concern NDCG@10;
the secondary statement does not claim equivalence or nonsignificance on
untested endpoints.
