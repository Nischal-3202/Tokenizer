# Final research handoff

**Status:** experimental phase frozen on 2026-10-29. This note checks the saved artifacts. It does not add a training run, change raw result JSON, or change the reporting pipeline.

Use this file, together with `results/master_experiment_report.md`, as the numerical source of truth. `docs/literature_positioning_and_design_decision.md` is an earlier positioning memo. Its quoted 1024-resample means match the fixed-per-language result below, but its status line still says the budget-matched control had not been run. That control has been run. Do not take the memo's "current evidence" section as the complete result.

## Numerical check

Formula, applied to saved test-token counts:

`penalty_percent = 100 * (joint_tokens - dedicated_tokens) / dedicated_tokens`

Mean is the arithmetic mean of the three replicate penalties. Sample standard deviation is `statistics.stdev` (divisor n−1). Published tables and figures round those values with Python `round(x, 2)`.

No material discrepancy was found. Recomputed penalties match `results/total_budget_matched_v1_summary.json` and `results/repeated_vocabulary_sharing_v1_summary.json` to within `1e-12`. One repeated Hindi mean differs from the stored summary by about `2e-15` (`15.624848189283723` versus `15.624848189283725`). Both round to 15.62%. That is floating-point noise, not a conflicting result.

The CSV rows for the sweep, the three resamples, and the budget-matched control match the raw JSON token counts, `bytes_per_token` (to displayed precision), and `roundtrip_exact` flags. The three SVG figures show the same rounded penalties as the report.

`results/balanced_vocabulary_sharing_v1.json` has the same training hashes and the same 1024 test-token counts as `results/balanced_vocabulary_sweep_v1.json`. It is not an independent run.

## Results to cite

### Tier 1 — one balanced run at each vocabulary size

Source: `results/balanced_vocabulary_sweep_v1.json`. Not replicate means.

| Vocabulary | English | Hindi | Nepali |
| ---: | ---: | ---: | ---: |
| 512 | 15.74% | 14.95% | 17.53% |
| 1024 | 13.65% | 15.41% | 19.58% |
| 2048 | 14.33% | 14.66% | 18.11% |

Training UTF-8 bytes in that selection: English 297,381; Hindi 299,885; Nepali 299,348. The joint model was trained on 896,616 bytes.

### Tier 2 — three resamples, vocabulary 1024, about 300 KB per language

Source: `results/repeated_vocabulary_sharing_v1.json`.

Joint training bytes: 896,306; 895,182; 899,404. Dedicated sums: 896,304; 895,180; 899,402. The joint string is 2 bytes larger because concatenation inserts a newline between language samples.

| Language | Dedicated tokens | Joint tokens | Penalties (%) | Mean ± sample SD |
| --- | --- | --- | --- | --- |
| English | 35,298 / 35,218 / 35,574 | 40,286 / 40,472 / 40,413 | 14.1311122443 / 14.9185075814 / 13.6026311351 | 14.2174169869% ± 0.6621699834 pp |
| Hindi | 44,435 / 44,545 / 44,170 | 51,365 / 51,219 / 51,368 | 15.5958141105 / 14.9826018633 / 16.2961285941 | 15.6248481893% ± 0.6572445142 pp |
| Nepali | 53,593 / 53,864 / 53,466 | 63,858 / 63,636 / 63,382 | 19.1536208087 / 18.1419872271 / 18.5463659148 | 18.6139913169% ± 0.5091959565 pp |

Rounded report form: English 14.22% ± 0.66 pp; Hindi 15.62% ± 0.66 pp; Nepali 18.61% ± 0.51 pp.

### Tier 3 — budget-matched control, vocabulary 1024, about 100 KB per language

Source: `results/total_budget_matched_v1.json`. Summary: `results/total_budget_matched_v1_summary.json`. Manifest: `data/manifests/total_budget_matched_replicates_v1.json`.

Joint training bytes: 297,882; 297,304; 295,867. Dedicated sums: 297,880; 297,302; 295,865. Again, the joint string is 2 bytes larger because of the two inserted newlines. Per-language files are 96,525–99,884 bytes.

| Language | Replicate | Dedicated tokens | Joint tokens | Penalty (%) | Reconstruction |
| --- | ---: | ---: | ---: | ---: | --- |
| English | 1 | 37,210 | 40,901 | 9.9193765117 | exact |
| English | 2 | 36,696 | 41,746 | 13.7617178984 | exact |
| English | 3 | 36,649 | 41,740 | 13.8912385058 | exact |
| Hindi | 1 | 45,874 | 52,042 | 13.4455246981 | exact |
| Hindi | 2 | 45,657 | 51,848 | 13.5598046302 | exact |
| Hindi | 3 | 46,513 | 52,206 | 12.2395889321 | exact |
| Nepali | 1 | 56,362 | 65,924 | 16.9653312516 | exact |
| Nepali | 2 | 55,890 | 64,622 | 15.6235462516 | exact |
| Nepali | 3 | 55,364 | 64,213 | 15.9833104544 | exact |

| Language | Mean ± sample SD |
| --- | --- |
| English | 12.5241109720% ± 2.2566956160 pp |
| Hindi | 13.0816394201% ± 0.7314723125 pp |
| Nepali | 16.1907293192% ± 0.6945240713 pp |

Rounded report form: English 12.52% ± 2.26 pp; Hindi 13.08% ± 0.73 pp; Nepali 16.19% ± 0.69 pp.

Every dedicated and joint evaluation on the matched language reconstructed the held-out text exactly. The saved JSON also records exact reconstruction for every model on every language's test.

### What the two 1024 regimes can support

Both regimes have a positive penalty for every language and every saved resample.

Mean penalties are lower in the matched-budget regime. Absolute test-token counts are higher there for both the dedicated and joint models. Arithmetic means of the three saved counts:

| Language | Dedicated, ~300 KB each | Joint, ~900 KB | Dedicated, ~100 KB | Joint, ~300 KB |
| --- | ---: | ---: | ---: | ---: |
| English | 35,363.33 | 40,390.33 | 36,851.67 | 41,462.33 |
| Hindi | 44,383.33 | 51,317.33 | 46,014.67 | 52,032.00 |
| Nepali | 53,641.00 | 63,625.33 | 55,872.00 | 64,919.67 |

The smaller percentage is therefore not a reduction in joint-model tokens. The comparison changes dedicated sample size, joint sample size, and document composition together. It does not isolate total training bytes. Sample standard deviations describe these three overlapping resamples. They are not confidence intervals or significance tests. English's matched-budget spread (±2.26 pp) overlaps the fixed-per-language English penalties.

## Corpus and budgets

Verified from `src/collect_corpus.py`, `data/manifests/pilot_manifest.json`, and the saved splits:

- Source name recorded in the manifest: Hugging Face dataset `ashtok897/indic-hplt-v2`, streaming `train` split. Collection timestamp: 2026-10-28T06:18:59.246915+00:00.
- No dataset revision, commit, or snapshot hash is stored. Whether a fresh stream would reproduce these files is unverified. The saved files under `data/processed/` are the corpus of record.
- Languages: English `en`, Hindi `hi`, Nepali `ne`.
- Cleaning: NUL bytes replaced, whitespace collapsed, ends stripped. Documents outside 100–20,000 characters were rejected. Exact cleaned-text duplicates were dropped within each language.
- Collection stopped after 1,000,000 characters per language, counted with Python `len` (code points, not UTF-8 bytes).
- Saved pool, from `data/manifests/pilot_manifest.json`: English 102 documents (91 train / 11 test); Hindi 451 (405 / 46); Nepali 538 (484 / 54).
- The split shuffles with `random.Random(42)` and cuts at 90% of the document count. One generator continues from English to Hindi to Nepali. The held-out portion is about 10% of documents, not 10% of bytes.
- A document in later audits and selectors is one nonempty stripped line.

Train/test facts from `results/corpus_audit_v2.json` (created 2026-10-29T06:27:03.759860+00:00):

- Exact line overlap between processed train and test is 0 for every language pair, including cross-language pairs.
- Within-file duplicate excess is 0 for the processed splits, the balanced selection, and the three 300 KB replicates.
- NFC/NFKC collision counts in that audit are 0 for those same files.
- Within-file character-shingle screening at Jaccard threshold 0.90 and shingle size 5 found 0 candidate pairs in those files. The audit states that this does not detect paraphrases.
- The audit's own limitations say hash matches do not prove upstream provenance, and it cannot prove the test set is free of contamination that occurred before this split.

Budget definitions, verified from the preparation scripts and saved byte counts:

- Fixed-per-language resamples (`src/prepare_repeated_corpora.py`): seeds 11, 22, and 33. Lines of `data/processed/{language}/train.txt` are ordered by SHA-256 of `"{seed}:"` plus the line. Whole lines are kept while the file, including newlines, stays within 300,000 UTF-8 bytes. Every saved replicate line is contained in that language's processed train file (`right_fraction_shared = 1` in the audit).
- Budget-matched subsets (`src/prepare_total_budget_matched_corpora.py`): within each of those 300 KB files, lines are ordered by SHA-256 of `total_budget_matched_v1:{seed}:{language}:` plus the line and kept within 100,000 UTF-8 bytes, including newlines. Every 100 KB line set is a subset of its parent 300 KB file and shares no stripped line with that language's test file. The same is true of the parent 300 KB files and the test file.
- Joint training text in both regimes is `"\n".join` of the three language strings. Because each language string already ends in a newline, the joint text is 2 bytes larger than the sum of the three files.
- The single-run sweep uses `src/prepare_balanced_multilingual_corpus.py`. That selector sorts by SHA-256 of the line and counts document bytes only, not separator newlines, against a 300,000-byte cap. The written files still land at the byte counts listed under Tier 1. This cap is not implemented identically to the resample cap.
- "About 300 KB" and "about 900 KB" in the prose are decimal approximations of these UTF-8 byte counts. The integers above are the quantities to cite. They are not kibibytes.

Overlap, from the audit's document Jaccard on the 300 KB replicates:

| Language | Pairwise Jaccard |
| --- | --- |
| English | 0.140000, 0.230769, 0.166667 |
| Hindi | 0.066038, 0.076087, 0.087379 |
| Nepali | 0.103448, 0.064000, 0.046512 |

The report's ranges, English 0.14–0.23, Hindi 0.07–0.09, and Nepali 0.05–0.10, are those values rounded to two decimals. The exact Nepali maximum is 0.103448.

Those ranges do not describe the 100 KB subsets. `corpus_audit_v2.json` does not include them. A direct check of the saved 100 KB files during this handoff found pairwise exact-line Jaccard of 0 for all English pairs and all Hindi pairs. Nepali pairs were 0.107143 (3 shared lines, replicates 1 and 2), 0 (replicates 1 and 3), and 0.027027 (1 shared line, replicates 2 and 3). The parent 300 KB samples still overlap, so these subsets are not independent draws.

## Tokenizer

The reported experiments use `src/tokenizer/tokenizer.py` and `src/tokenizer/bpe.py`.

- Training encodes the string as UTF-8 bytes. The base vocabulary is the 256 byte values.
- Requested vocabulary size `V` schedules `V - 256` merges. Actual vocabulary size is 256 plus the number of merges completed. Every controlled model checked here reached the requested size: 512, 1024, or 2048, with 256, 768, or 1,792 merges.
- At each step the most frequent adjacent pair is merged. If several pairs tie, `max(stats, key=stats.get)` keeps the first pair inserted while scanning the current sequence left to right. The procedure is deterministic. There is no separate random seed inside BPE.
- Encoding repeatedly applies the earliest learned merge that is still present. Decoding concatenates the byte strings and decodes UTF-8 with `errors="replace"`. `roundtrip_exact` means the decoded string equals the original string.
- The same held-out test files are used for the sweep, the duplicate 1024 run, the three resamples, and the budget-matched control. SHA-256 values: English `3ca0d56b55616435f868d83b95b17e08ddae354edc0f2251961fe4944cd3bcfd` (93,262 bytes); Hindi `1a10e42e80775ad75254534e4dae8e455f6208ac7ceeaeb5c7485cfb21239f21` (226,287 bytes); Nepali `c90991323ea059db86845d94ea42f7370dc82acc7d8f9e98b13a83a927973497` (295,837 bytes).

Grapheme and akshara tokenizers are a separate full-corpus baseline at requested vocabulary 4096. They appear in `results/master_results_table.csv` as `contextual_baseline` and in `results/multilingual_tokenizer_baseline_v1.json`. They are not part of the three evidence tiers.

## Repository map

Byte-level BPE evidence:

| File | Role |
| --- | --- |
| `src/tokenizer/bpe.py` | Pair counts, merges, encode, decode |
| `src/tokenizer/tokenizer.py` | UTF-8 byte tokenizer and JSON save/load |
| `src/tokenizer/metrics.py` | Token count, bytes per token, exact reconstruction |
| `src/collect_corpus.py` | Stream and split the source corpus |
| `src/prepare_balanced_multilingual_corpus.py` | One 300 KB-per-language selection for the sweep |
| `src/run_balanced_vocabulary_sweep.py` | Train sweep at 512, 1024, and 2048 |
| `src/run_balanced_vocabulary_experiment.py` | Train the duplicate 1024 run |
| `src/prepare_repeated_corpora.py` | Three 300 KB resamples |
| `src/run_repeated_vocabulary_experiment.py` | Train the fixed-per-language resamples |
| `src/prepare_total_budget_matched_corpora.py` | Three 100 KB matched subsets |
| `src/run_total_budget_matched_experiment.py` | Train the budget-matched control |
| `src/audit_corpus_v2.py` | Saved corpus audit |
| `src/consolidate_results.py` | Rebuild the report, CSV, portfolio summary, and three figures from saved JSON |
| `src/run_multilingual_experiment.py` | Full-corpus 4096 baseline, including grapheme and akshara |

Training scripts refuse to overwrite an existing result file. Do not rerun them for this frozen phase.

`src/tokenizer/grapheme_bpe.py`, `src/tokenizer/akshara_bpe.py`, `src/tokenizer/graphemes.py`, and `src/tokenizer/aksharas.py` support the separate baseline and the unit tests. They are not the implementation behind the three evidence tiers.

## Tests and reproduction

Command run for this handoff, from the repository root:

`.venv/bin/python -m pytest tests -q`

Result: `20 passed in 0.05s`.

Python requirement in `pyproject.toml`: `>=3.12,<3.13`. The byte-BPE trainer, metrics, and `src/consolidate_results.py` use the standard library. Regenerating the report does not train models:

`python src/consolidate_results.py`

`requirements.txt` lists `jupyter`, `ipykernel`, `pytest`, `regex`, and `tiktoken`. The byte-BPE evidence path does not import `tiktoken`. Grapheme segmentation imports `regex`. Corpus collection imports `datasets`, which is not listed in `requirements.txt`. Reproducing collection also requires network access to the named dataset and is not pinned to a revision.

There is no git repository in this directory.

## Final output filenames

Reports and figures:

- `results/master_experiment_report.md`
- `results/master_results_table.csv`
- `results/portfolio_results_summary.md`
- `results/figures/shared_vocabulary_penalty_sweep.svg`
- `results/figures/shared_vocabulary_penalty_resamples.svg`
- `results/figures/shared_vocabulary_penalty_budget_comparison.svg`

Primary result JSON:

- `results/balanced_vocabulary_sweep_v1.json`
- `results/balanced_vocabulary_sharing_v1.json` (same 1024 selection as the sweep)
- `results/repeated_vocabulary_sharing_v1.json`
- `results/repeated_vocabulary_sharing_v1_summary.json`
- `results/total_budget_matched_v1.json`
- `results/total_budget_matched_v1_summary.json`
- `results/multilingual_tokenizer_baseline_v1.json` (contextual baseline only)
- `results/corpus_audit_v2.json`

Manifests for the cited designs:

- `data/manifests/pilot_manifest.json`
- `data/manifests/controlled_multilingual_balanced_v1.json`
- `data/manifests/controlled_multilingual_replicates_v1.json`
- `data/manifests/total_budget_matched_replicates_v1.json`

Trained model JSON files are under `results/models/` and are named `{experiment}_{condition}_vocab_{size}.json`, with a replicate segment for the resampled experiments.

## Claims that should stay narrow

The verified result is an intrinsic token-count comparison for this corpus, this byte-BPE implementation, these budgets, and this vocabulary. A shared vocabulary used more test tokens than the matched dedicated vocabulary in every saved comparison.

It does not establish a general property of multilingual tokenizers, a Devanagari-specific law, downstream task quality, or a causal effect of training bytes alone. The literature memo argues against novelty claims for BPE, vocabulary sharing, and Indic tokenizer design. This handoff did not re-check those papers.
