# Verified bibliography corrections

Checked 30 September 2026 against primary publication records, author-hosted
papers, and publisher-deposited metadata. This checks the proposed corrections;
it is not a new independent audit of every field in all 27 references.
Citation keys and the 27-entry bibliography are preserved.

| Reference | Applied correction | Supporting record |
| --- | --- | --- |
| 1, HetRec | Added pages 387–388. | [Publisher-deposited Crossref record](https://api.crossref.org/works/10.1145/2043932.2044016) |
| 5, kNN-Embed | Added PAKDD pages 374–386. | [Springer](https://doi.org/10.1007/978-3-031-33380-4_29) |
| 6, MovieLens | Corrected article pagination to 19:1–19:19; retained 2015. | [GroupLens citation instructions](https://files.grouplens.org/datasets/movielens/ml-32m-README.html) |
| 13, NARM | Added the omitted author Tao Lian between Zhaochun Ren and Jun Ma. | [Author-hosted paper](https://renzhaochun.github.io/assets/pdf/1711.04725.pdf) |
| 14, MIND | Put Huan Zhao before Pipei Huang, matching the published CIKM version. | [Author-hosted CIKM paper, first-page author list and ACM reference format](https://hzhaoaf.github.io/data/cikm19-mind.pdf) |
| 17, constant-space retrieval | Replaced the arXiv-only entry with ECIR 2025, pages 237–245, and its DOI. | [Springer](https://doi.org/10.1007/978-3-031-88714-7_22) |
| 19, challenge | Used the official page title, ACM RecSys attribution, and access date. | [Official RecSys page](https://recsys.acm.org/recsys26/challenge/) |
| 21, Rocchio | Expanded the host book title to include *Experiments in Automatic Document Processing*. | [Book record/preview](https://books.google.com/books/about/The_SMART_Retrieval_System.html?id=7-M8AAAAIAAJ) |
| 25, MiniLM | Added NeurIPS pages 5776–5788. | [Official proceedings metadata](https://papers.nips.cc/paper_files/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) |
| 27, ANCE-PRF | Corrected CIKM pages from 2805–2809 to 3592–3596. | [Publisher-deposited Crossref record](https://api.crossref.org/works/10.1145/3459637.3482124) |

The MIND correction is version-specific: its [arXiv record](https://arxiv.org/abs/1904.08030)
lists Huang before Zhao, whereas the published CIKM paper lists Zhao before
Huang. The manuscript cites CIKM, so the published ordering is appropriate.

[TalkPlayData 2](https://arxiv.org/abs/2509.09685) is a real paper, but this pass
did not establish that the evaluated challenge files are that exact release.
It was not added as a dataset-provenance citation. The official challenge
reference remains the source for the evaluated challenge data.

The corrections apply to both current manuscript versions. Frozen historical
review ZIPs are unchanged. Recompile the final manuscript before submission:
bibliography line wrapping and reference-page count may change.
