# Counterfactual vocabulary augmentation

This experiment asks whether the Nepali byte strings missing from the shared vocabulary are sufficient to explain Nepali’s extra Devanagari tokenization penalty relative to Hindi.

It does not retrain a tokenizer. Each intervention appends merges to a copy of a saved `repeated_vocabulary_sharing_v1` shared model. Original model files, test files, the manuscript, and `analysis/nepali_penalty_investigation/` were not modified.

## Augmentation rule

A candidate is one learned merge from the dedicated Hindi or Nepali model whose byte string is absent from that replicate’s shared vocabulary. The split is the dedicated model’s own child pair. Candidates are ranked by how often the dedicated tokenizer emits them on that replicate’s training file. Held-out text is not opened until those lists are frozen. Training-file hashes match `results/repeated_vocabulary_sharing_v1.json`.

A candidate is eligible only if, on the original shared vocabulary:

1. Both child byte strings already exist as exactly one shared token.
2. Encoding the candidate bytes in isolation, with that one new merge appended, returns exactly the new token.

New merge ids start above every original shared id. Lower id still means higher priority, so every original shared merge outranks every added merge. Original merge order is unchanged. If an earlier added merge later prevents isolation construction, the later piece is skipped. In these three replicates that did not happen inside a single-language list: every individually eligible piece was accepted.

Pieces whose children are not already in the shared vocabulary are excluded. Adding them would require new intermediate tokens and a priority the saved shared merges do not determine.

Budgets are 25, 50, 100, 200, and every accepted piece. The random control shuffles the same eligible pool with `Random(20261030 + replicate index)` and uses the same append rule. Combined budgets add the same number from each language, alternating Nepali then Hindi. At the full combined budget, about 45–52 pieces are dropped because the other language has already added the same byte string. No combined drop was caused by a priority conflict.

Per replicate, the Nepali dedicated model is missing about 463 learned strings. About 340 are eligible, about 110 are excluded because a child is absent, and about 13 fail isolation against the original shared merges. Hindi is similar: about 451 missing, 336 eligible, 101 child-absent, 15 isolation failures.

## Validation

- Baseline shared and dedicated token counts match the published JSON on every language and replicate.
- Baseline Devanagari-starting token counts match `script_penalty.csv` from the previous investigation.
- Every evaluated segmentation reconstructs the held-out bytes exactly.
- The original shared merge list is a prefix of every augmented merge list.
- Candidate order was fixed before any test file was read.

Baseline means, which reproduce the published penalties:

| Language | Overall penalty | Devanagari-starting penalty |
| --- | ---: | ---: |
| English | 14.22% | — |
| Hindi | 15.62% | 16.28% |
| Nepali | 18.61% | 25.23% |

The Devanagari penalty is the relative increase in tokens whose first byte lies in a Devanagari character. English has none.

## What restoring Nepali pieces does

Mean over the three replicates. “Pieces” are Nepali strings added to the shared model, in training-frequency order.

| Nepali pieces | Nepali overall penalty | Nepali Devanagari penalty | Hindi Devanagari penalty on the same model | Nepali − Hindi gap |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 18.61% | 25.23% | 16.28% | +8.94 pp |
| 25 | 16.54% | 22.72% | 16.04% | +6.68 pp |
| 50 | 14.80% | 20.44% | 15.39% | +5.05 pp |
| 100 | 11.77% | 16.66% | 14.17% | +2.48 pp |
| 200 | 6.91% | 10.48% | 11.66% | −1.17 pp |
| All eligible (~340) | 1.55% | 3.63% | 8.64% | −5.00 pp |

The decline is in every replicate. At 100 pieces the three Nepali Devanagari penalties are 16.62%, 16.80%, and 16.54%. At the full eligible set they are 3.06%, 3.84%, and 3.99%.

Relative to Hindi’s original 16.28% Devanagari penalty, Nepali’s excess is about gone at 100 pieces (16.66% versus 16.28%). Relative to Hindi tokenized by the same Nepali-augmented model, the gap closes later, between 100 and 200 pieces, because Nepali strings also shorten Hindi a little.

Word-level overlap counts on the Nepali test move with the same intervention:

| Nepali pieces | Words with more shared tokens | Mean extra tokens per word | 90th percentile increase |
| ---: | ---: | ---: | ---: |
| 0 | 46.6% | 0.530 | 66.7% |
| 25 | 42.8% | 0.471 | 64.4% |
| 100 | 34.1% | 0.328 | 50.0% |
| 200 | 24.3% | 0.185 | 50.0% |
| All eligible | 14.2% | 0.026 | 30.6% |

The median word stays at a 0% increase. The tail shrinks.

Boundary accounting against the dedicated Nepali segmentation:

| Nepali pieces | Net extra tokens | Extra cuts inside strings still absent | Extra cuts inside strings now present | Dedicated boundaries crossed by a longer token |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 9,984 | 13,537 | 1,058 | 4,611 |
| 100 | 6,314 | 8,935 | 2,049 | 4,669 |
| All eligible | 829 | 3,063 | 2,662 | 4,896 |

About 91.7% of the original net extra tokens are gone after every eligible Nepali string is restored (9,984 to 829). Cuts inside strings that are still absent fall from 13,537 to 3,063. Aligned dedicated tokens whose bytes are in the augmented vocabulary are still never fragmented: that count stays zero. The rise in “present” extra cuts is unaligned spans, where a higher-priority original merge crosses the dedicated token even though the byte string is now available.

## Hindi control, random control, and combined addition

Adding Hindi’s own eligible strings to the shared model reduces Hindi’s Devanagari penalty from 16.28% to 14.42% at 25 pieces, 9.43% at 100, and −0.95% at the full eligible set (about 336 pieces). The small negative value means the augmented model, which still contains the original shared merges, uses slightly fewer Devanagari-starting tokens than the dedicated Hindi model. Nepali’s full eligible set does not overshoot in that way: its Devanagari penalty remains +3.63%.

The same Hindi strings do much less for Nepali. At 100 Hindi pieces, Nepali’s Devanagari penalty is 23.31%, down 1.9 percentage points from 25.23%. At 100 Nepali pieces, the Nepali penalty is 16.66%, down 8.6 percentage points. Each language’s strings mainly help that language.

Random Nepali pieces move the same curve, more slowly at small budgets. At 25 pieces the Devanagari penalty is 22.72% in frequency order and 23.18% in random order. At 100 pieces the figures are 16.66% and 18.53%. At 200 pieces they are 10.48% and 12.54%. The full sets meet, at 3.63% and 3.62%, because they are the same pieces. Frequency order matters for a fixed small budget. The size of the eligible pool, not the shuffle, determines the endpoint.

Combined addition, with equal counts from both languages, lowers Nepali further than Nepali pieces alone: Devanagari penalty 14.96% at 100 per language, 6.97% at 200, and 0.17% at the full combined set. Most of the early drop is still the Nepali half. From the Nepali-only result at 100 pieces (16.66%) to the combined result (14.96%) is 1.7 percentage points. The Nepali-only step from the baseline to 100 pieces is 8.6 percentage points.

The curves do not flatten early. From 200 Nepali pieces to the remaining ~140 eligible pieces, the Devanagari penalty still falls from 10.48% to 3.63%. Per-piece gains get smaller, and they are still material through the end of the eligible pool.

## Cross-language effects

Token-count change versus the unmodified shared model, frequency interventions, means of three replicates:

| Intervention | English | Hindi | Nepali |
| --- | ---: | ---: | ---: |
| 100 Nepali pieces | −4 | −698 | −3,670 |
| All eligible Nepali pieces | −9 | −2,486 | −9,155 |
| 100 Hindi pieces | −11 | −2,678 | −908 |
| All eligible Hindi pieces | −88 | −6,333 | −3,443 |

Nepali augmentation does not increase Hindi or English token counts. English changes by fewer than 10 tokens until the full Nepali set (−9 on a baseline of 40,390). Hindi gets shorter, not longer. Hindi augmentation likewise shortens Nepali and leaves English almost unchanged (−88 tokens at the full Hindi set). None of these interventions buys one language’s improvement by making another language worse.

## Answers

1. **Does adding missing Nepali pieces reduce Nepali’s Devanagari fragmentation penalty?** Yes. It falls from 25.23% to 22.72% at 25 pieces, 16.66% at 100, 10.48% at 200, and 3.63% when every eligible piece is added. Overall penalty falls from 18.61% to 1.55% at that last point. The share of Nepali word occurrences that get longer falls from 46.6% to 14.2%.

2. **Does it reduce the Nepali–Hindi gap?** Yes. The original Devanagari gap is 8.94 percentage points (25.23% versus 16.28%). Against Hindi’s original penalty, Nepali is level at about 100 pieces. Against Hindi run through the same Nepali-augmented model, Nepali is still 2.48 percentage points higher at 100 pieces and 1.17 percentage points lower at 200. Past that point the original excess is reversed, because the added Nepali strings help Nepali more than they help Hindi.

3. **Does adding Hindi pieces produce a comparable improvement?** On Hindi, yes. One hundred Hindi pieces cut Hindi’s Devanagari penalty by 6.9 percentage points; one hundred Nepali pieces cut Nepali’s by 8.6. On Nepali, Hindi pieces do not: 100 of them cut Nepali’s Devanagari penalty by 1.9 percentage points. The improvement follows the language of the strings.

4. **Do improvements saturate?** Not inside this eligible pool. Gains per piece shrink, and the penalty is still falling between 200 pieces and the full set of about 340. Saturation would require a flat region before the pool runs out. There is not one.

5. **Does Nepali augmentation harm Hindi or English?** No. Hindi token counts fall. English token counts fall by 0 to 9 tokens depending on the budget.

6. **How much of the original penalty can be attributed to missing pieces under this intervention?** Restoring every eligible Nepali string removes 21.6 of the 25.2 Devanagari percentage points, which is 86% of that penalty, and 17.1 of the 18.6 overall percentage points, which is 92% of that penalty. Net extra tokens fall from 9,984 to 829. This attributes the removed share to these strings under append-only, lower-priority merges. It does not attribute the removed share to strings that were excluded.

7. **Does a substantial residual gap remain?** A residual Nepali penalty remains: 3.63% Devanagari and 1.55% overall after every eligible string, with 3,063 extra cuts still inside dedicated strings that were not added. That is much smaller than the original 25.23% and 18.61%, and it is below Hindi’s original Devanagari penalty. About a quarter of the missing Nepali strings were never added, because a child byte string was not in the shared vocabulary. Whether those excluded strings account for the leftover cuts was not tested.

8. **What does this establish, and what does it not establish?** It establishes that the eligible missing Nepali byte strings, inserted below the saved shared merges, are sufficient to remove Nepali’s excess Devanagari penalty relative to Hindi and most of Nepali’s own penalty, without lengthening Hindi or English. The Hindi control shows the same kind of gain from Hindi’s own missing strings. Frequency order beats a random order at small budgets and matches it once the pool is exhausted.

It does not establish what a retrained joint tokenizer would learn. The augmented vocabulary is larger than 1024 and was not produced by the original training objective. It does not establish that the excluded child-absent strings are harmless, or that the residual 3.63% has a morphological cause. It does not say the shared model was failing to use strings it already contained: aligned strings that are present remain unfragmented, before and after augmentation.

## Files

| Path | Contents |
| --- | --- |
| `run_experiment.py` | Selection, augmentation, evaluation, figures |
| `bpe_fast.py` | Encoder used for the copies |
| `configs/augmentation_semantics.json` | Priority and selection rules |
| `configs/replicate_*_{hindi,nepali}_candidates.json` | Accepted order and training counts |
| `models/*.json` | Augmented merge tables; originals are not in this directory |
| `augmentation_results.csv` | Penalties by intervention, budget, language |
| `candidate_pieces.csv` | Every missing piece, with exclusion reason or addition order |
| `boundary_decomposition.csv` | Extra cuts before and after augmentation |
| `word_fragmentation_results.csv` | Word-level increase rates |
| `cross_language_effects.csv` | Token deltas versus the unmodified shared model |
| `figures/devanagari_penalty_vs_budget.svg` | Devanagari penalty against pieces added |
| `figures/overall_penalty_vs_budget.svg` | Overall penalty, including the random and combined controls |
| `figures/cross_language_token_delta.svg` | Effect of each language’s pieces on all three tests |

Reproduce with:

```bash
.venv/bin/python analysis/counterfactual_vocabulary_augmentation/run_experiment.py
```
