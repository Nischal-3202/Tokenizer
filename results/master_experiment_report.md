# Master Experiment Report

Derived from saved JSON by `python src/consolidate_results.py`; no training was run and raw outputs were not modified.

## Design

Two complementary byte-BPE regimes use the same requested vocabulary and fixed held-out tests. The fixed-per-language regime trains each dedicated tokenizer on about 300 KB and the joint tokenizer on about 900 KB. The total-budget-matched regime trains dedicated and joint conditions from the same approximately 100 KB per-language samples; the joint tokenizer totals about 300 KB. Both are intrinsic token-count experiments, not downstream-quality evaluations.

## Evidence tier 1: single-run vocabulary sweep

One deterministic balanced run was saved at each size; these are observations, not replicate means.

| Vocab | English | Hindi | Nepali |
| --- | --- | --- | --- |
| 512 | 15.74% | 14.95% | 17.53% |
| 1024 | 13.65% | 15.41% | 19.58% |
| 2048 | 14.33% | 14.66% | 18.11% |

`balanced_vocabulary_sharing_v1.json` repeats the same 1024 selection and is not an independent run.

## Evidence tier 2: three-resample summary at vocabulary 1024

Each dedicated tokenizer trains on one language sample of about 300 KB. Joint training bytes are 896,306, 895,182, and 899,404. The three dedicated models sum to 896,304, 895,180, and 899,402 bytes. Each joint total is 2 bytes larger than that sum because concatenation inserts a newline between the language samples.

| Language | Dedicated tokens (1/2/3) | Shared tokens (1/2/3) | Mean ± sample SD |
| --- | --- | --- | --- |
| English | 35298/35218/35574 | 40286/40472/40413 | 14.22% ± 0.66 pp |
| Hindi | 44435/44545/44170 | 51365/51219/51368 | 15.62% ± 0.66 pp |
| Nepali | 53593/53864/53466 | 63858/63636/63382 | 18.61% ± 0.51 pp |

## Evidence tier 3: total-budget-matched three-resample control at vocabulary 1024

Within each replicate, the dedicated and joint conditions use the same per-language source sample of approximately 100 KB. Joint training bytes are 297,882, 297,304, and 295,867. The three dedicated models sum to 297,880, 297,302, and 295,865 bytes. Each joint total is 2 bytes larger than that sum because concatenation inserts a newline between the language samples.

| Language | Replicate | Dedicated tokens | Joint tokens | Penalty (%) | Exact reconstruction |
| --- | --- | ---: | ---: | ---: | --- |
| English | 1 | 37210 | 40901 | 9.92 | exact |
| English | 2 | 36696 | 41746 | 13.76 | exact |
| English | 3 | 36649 | 41740 | 13.89 | exact |
| Hindi | 1 | 45874 | 52042 | 13.45 | exact |
| Hindi | 2 | 45657 | 51848 | 13.56 | exact |
| Hindi | 3 | 46513 | 52206 | 12.24 | exact |
| Nepali | 1 | 56362 | 65924 | 16.97 | exact |
| Nepali | 2 | 55890 | 64622 | 15.62 | exact |
| Nepali | 3 | 55364 | 64213 | 15.98 | exact |

Exact reconstruction means both the dedicated tokenizer and the joint tokenizer encode and decode that language's held-out test without change.

| Language | Mean ± sample SD |
| --- | --- |
| English | 12.52% ± 2.26 pp |
| Hindi | 13.08% ± 0.73 pp |
| Nepali | 16.19% ± 0.69 pp |

## Claims and limitations

- The shared 1024-token vocabulary produced more tokens than the dedicated vocabulary for every language in every saved resample in both budget regimes.
- Mean penalties are lower under the approximately 300 KB joint budget than under the approximately 900 KB joint budget. Absolute test-token counts are higher for both dedicated and joint models in the matched regime, so the smaller percentage is not a reduction in joint-model tokens.
- That cross-regime difference changes dedicated sample size, joint sample size, and document composition together. It does not isolate total training bytes as a cause.
- For the approximately 300 KB-per-language resamples, document overlap has Jaccard ranges of English 0.14–0.23, Hindi 0.07–0.09, and Nepali 0.05–0.10. Sample SDs describe that resample spread, not confidence intervals or statistical significance. Overlap among the 100 KB subsets was not re-audited in this reporting pass.
- Corpus checks cover hashes, budgets, duplicates, and normalization collisions, not upstream provenance or pre-split contamination.
- Results are limited to this corpus, implementation, byte-BPE setup, budget allocation, and token-count metric. They do not establish novelty, downstream-quality effects, or broad generalization.

All tabulated controlled runs reconstruct inputs exactly after encoding and decoding.
