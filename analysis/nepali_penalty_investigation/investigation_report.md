# Why Nepali has a higher shared-vocabulary tokenization penalty

This note uses the saved vocabulary-1024 byte-level BPE models, their training files, and the fixed held-out tests. No tokenizer was retrained, and no existing result file was changed. Every published test token count used below was recomputed from the saved merge tables and matched the stored JSON: 72 model/test pairs, with no mismatches.

Two regimes are covered.

- `repeated_vocabulary_sharing_v1`: each dedicated tokenizer is trained on about 300 KB of one language, and the shared tokenizer is trained on the concatenation, about 900 KB. The published mean penalties are English 14.22%, Hindi 15.62%, and Nepali 18.61%.
- `total_budget_matched_v1`: dedicated and shared training use the same approximately 100 KB sample of each language, so the shared model sees about 300 KB in total. The published mean penalties are English 12.52%, Hindi 13.08%, and Nepali 16.19%.

The penalty is `(shared tokens / dedicated tokens) − 1` on `data/processed/{language}/test.txt`. It is a ratio on one fixed text. A language with more bytes can have more tokens without having a higher penalty.

## How pieces were identified

A learned piece is the byte string produced by a merge, not its integer id. Integer ids are not comparable across models. A byte string that is not valid UTF-8 is kept; several frequent merges are prefixes of Devanagari characters. In the glosses below, `⟨e0⟩⟨a4⟩` means the two bytes `E0 A4`, which are the UTF-8 lead of U+0900–U+093F. `⟨e0⟩⟨a5⟩` is the lead of U+0940–U+097F.

A dedicated merge is counted as present in the shared tokenizer in two ways:

- **Type coverage:** the resulting byte string occurs somewhere in the shared vocabulary.
- **Rule coverage:** the shared model contains a merge with the same two child byte strings, in the same order.

Usefulness is the number of times the shared tokenizer emits the piece on a training file or on the held-out test. A separate raw count asks only whether the byte string occurs in the training file, whether or not BPE emits it. The emission class uses an 80% threshold on the three training files of that replicate: exclusive if only one language emits it, predominant if one language emits at least 80%, otherwise cross-lingual.

The boundary account is an identity, not a model. On one text the dedicated and shared segmentations are two partitions of the same bytes. The difference in token counts equals the number of shared cuts that the dedicated segmentation does not make, minus the dedicated cuts that the shared segmentation does not make. The second set are places where a shared token crosses a dedicated token boundary.

## Validation

Re-encoding with the saved merges reproduced the published token counts for every dedicated and shared model on every held-out language, in both regimes. The recomputed penalties match the published means to the reported precision: 14.22%, 15.62%, 18.61% and 12.52%, 13.08%, 16.19%.

In all six shared models, the first four merges are the same: `E0 A4`, `E0 A5`, space plus `E0 A4`, and `E0 A4 BE` (the bytes of Devanagari vowel sign AA). Joint training therefore spends its first merges on the Devanagari UTF-8 frame that Hindi and Nepali share, and on a space before that frame.

## 1. Shared vocabulary allocation

Of the 768 learned merges in the 900 KB regime, the mean allocation by training emission is:

| Class | Mean learned tokens | Share |
| --- | ---: | ---: |
| English, exclusive or predominant | 385.4 | 50.2% |
| Cross-lingual | 263.3 | 34.3% |
| Nepali, exclusive or predominant | 64.0 | 8.3% |
| Hindi, exclusive or predominant | 55.3 | 7.2% |

Nepali’s language-specific slot count is slightly higher than Hindi’s, not lower. English occupies about half of the learned vocabulary. Those English pieces are rarely emitted on Indic text. On the Nepali held-out test, 80.8% of the shared tokenizer’s learned-token emissions are cross-lingual pieces, 16.4% are Nepali-predominant or Nepali-exclusive, and English-predominant pieces are 1.5%. On the Hindi test the split is 81.6% cross-lingual and 15.5% Hindi-specific. On the English test, 99.2% of learned emissions are English-exclusive or English-predominant.

Raw presence in the training bytes tells the same story from the other direction. In the 900 KB regime, a mean of 335.7 learned strings occur in Hindi and Nepali but not English, 184.7 occur in all three, and only 13.7 occur in Nepali alone. Hindi-only strings average 21.7. Most Indic learned strings are shared by the two Devanagari languages. Language-exclusive Nepali strings are a small minority of the vocabulary.

By byte structure, the 900 KB shared vocabulary averages 391.3 ASCII strings, 294.7 valid Devanagari strings, and 78.0 strings that are not valid UTF-8. The invalid strings are mostly the shared Devanagari prefixes. Classifying them by whether they “look like” a language would drop the pieces both Indic languages actually use.

The 300 KB regime moves more learned types into the English-exclusive class (mean 249.0, versus 105.7 in the 900 KB regime) and leaves Nepali-specific and Hindi-specific counts in the same narrow band, about 67–75 tokens combined with their predominant partners. The Nepali penalty stays the highest in that regime as well. A shortage of Nepali-labeled slots, relative to Hindi, is not present in either saved regime.

**Measured:** English takes about half of the learned types. Hindi and Nepali are allocated similar and much smaller language-specific sets. At emission time, both Indic tests are tokenized mainly by cross-lingual pieces.

**Interpretation:** The shared vocabulary is competitive, but the competition that shows up in the slot counts is English versus a shared Devanagari block. It does not show Nepali losing slots to Hindi.

## 2. Token fragmentation on the held-out texts

Whitespace-delimited words were scored from the full-document segmentation, so a token that crosses a space is counted in each word it touches. That overlap count is a description of words. The official penalty remains the document token count. Examples below are taken from `repeated_vocabulary_sharing_v1` by a fixed rule: word types with at least five held-out occurrences and at least one letter; rank by mean extra overlapping tokens across the three replicates; then, separately, types nearest the median percent increase among types that increase. The displayed segmentation is the replicate-1 occurrence closest to that type’s median extra, with each token clipped to the word. `…` means the token continues outside the word.

Across word occurrences, the mean percent increase and the share of occurrences that grow are:

| Language | Mean increase per word | Share of words with more shared tokens | Median increase | 90th percentile |
| --- | ---: | ---: | ---: | ---: |
| English | 12.5% | 23.7% | 0% | 50% |
| Hindi | 15.4% | 33.1% | 0% | 55.6% |
| Nepali | 22.0% | 46.6% | 0% | 66.7% |

The median word is unchanged in every language. The higher Nepali mean comes from a larger affected share and a heavier upper tail. The same ordering appears in the 300 KB regime (mean increases 12.0%, 13.9%, and 19.1%).

Extra shared boundaries that fall inside words are spread across the word rather than piled at the end. In the 900 KB regime, Nepali’s in-word extra cuts are distributed through the ten relative-position bins at roughly 3–14% each. Hindi has one mid-word peak: about 25% of its in-word extra cuts fall at 50–60% of the word.

Illustrative rows from the systematic selection:

| Language | Rule | Span | Occurrences | Tokens per occurrence, dedicated → shared | Increase | Replicate-1 pieces |
| --- | --- | ---: | --- | --- | ---: | --- |
| English | highest total extra | mill | 55 | 1.34 → 2.00 | +49% | `mill` → `m \| ill` |
| English | highest total extra | between | 15 | 1.33 → 3.33 | +150% | `between` → `b \| et \| we \| en` |
| English | median increase | work | 12 | 1.33 → 1.67 | +25% | `work` → `wor \| k` |
| Hindi | highest total extra | करने | 66 | 1.10 → 2.14 | +94% | `करने` → `कर \| ने` |
| Hindi | highest total extra | बाद | 58 | 1.40 → 2.30 | +65% | `बाद` → `ब \| ाद` |
| Hindi | median increase | क्षमता | 8 | 4.12 → 4.88 | +18% | `क \| ्ष \| म \| ता` → adds a cut before `ा` |
| Nepali | highest total extra | पनि | 179 | 1.61 → 2.40 | +49% | `पनि` → `प \| नि` |
| Nepali | highest total extra | गरेको | 78 | 1.46 → 2.56 | +76% | `गरेको` → `गर \| ेको` |
| Nepali | highest total extra | गर्ने | 87 | 1.42 → 2.36 | +66% | `गर्ने` → `गर्न \| े` |
| Nepali | highest rate | रुपैयाँ | 14 | 2.00 → 5.00 | +150% | one dedicated piece → `र \| ुप \| ै \| य \| ाँ` |
| Nepali | median increase | लागू | 13 | 2.46 → 3.00 | +22% | `लाग \| ू` → `ल \| ाग \| ू` |

The full selected set is in `fragmentation_examples.csv`. Several high-impact dedicated pieces continue past the word and include `⟨e0⟩⟨a4⟩`, the lead of the next character. Those are byte-level merges across a word boundary, not dictionary words.

**Measured:** Nepali word occurrences are more often lengthened, and the lengthening is distributed through the word. The same pattern appears for English and Hindi at lower rates. Some of the largest Nepali examples are frequent function-like forms (`पनि`, `गरेको`, `गर्ने`) and some are longer words (`रुपैयाँ`, `काठमाडौं`).

## 3. Unicode and UTF-8

On the fixed held-out tests:

| Quantity | English | Hindi | Nepali |
| --- | ---: | ---: | ---: |
| UTF-8 bytes | 93,262 | 226,287 | 295,837 |
| Codepoints | 92,916 | 90,844 | 112,800 |
| Bytes per codepoint | 1.004 | 2.491 | 2.623 |
| Devanagari codepoints | 0% | 74.4% | 81.0% |
| Latin letters | 80.6% | 2.5% | 2.3% |
| Codepoints per grapheme | 1.00 | 1.47 | 1.55 |
| Graphemes with more than one codepoint | 0% | 35.7% | 40.0% |
| Vowel signs per 1,000 Devanagari letters | — | 488.0 | 450.6 |
| Virama per 1,000 Devanagari letters | — | 74.4 | 128.6 |
| Anusvara per 1,000 letters | — | 61.9 | 8.3 |
| Candrabindu per 1,000 letters | — | 0.6 | 9.3 |
| Nukta per 1,000 letters | — | 7.4 | 0.0 |

Devanagari letters themselves are three UTF-8 bytes in both languages. The higher Nepali bytes-per-codepoint figure is the mixture: Nepali’s test is 81% Devanagari codepoints and Hindi’s is 74%, with a little less ASCII around the Nepali. Total variation distance between the Hindi and Nepali codepoint distributions is 0.166. The largest single gaps are virama (higher in Nepali), anusvara (higher in Hindi), vowel sign AA (higher in Nepali), and the letter NA (higher in Nepali).

Inside Devanagari runs, byte-trigram entropy is 6.81 bits for Hindi and 6.73 bits for Nepali. Six-byte entropy is 10.45 versus 10.32. Nepali’s Devanagari byte stream in this test is slightly more repetitive, not more diverse. Unique byte trigrams per 1,000 Devanagari bytes are 1.11 for Hindi and 0.86 for Nepali.

Token boundaries do split UTF-8 characters and Devanagari dependent signs, in both the dedicated and the shared models. In the 900 KB regime the mean fraction of internal boundaries that fall inside a UTF-8 character is 12.2% for shared Hindi and 15.2% for shared Nepali, against 15.0% and 15.2% for the dedicated models. The fraction that falls immediately before a Devanagari dependent sign is 34.7% (shared Hindi) and 36.5% (shared Nepali), against 35.3% and 34.8% for the dedicated models. The shared Nepali segmentation puts a slightly larger share of its cuts before dependent signs than its dedicated model does. The Hindi shared model does not.

A direct partition of the official token difference, by the script of the codepoint at each token’s first byte, sums to the published gap. Means over the three 900 KB replicates:

| Script at token start | English extra tokens | Hindi extra tokens | Nepali extra tokens |
| --- | ---: | ---: | ---: |
| Devanagari | 0 | +5,132 | +10,341 |
| Latin letters | +5,144 | −250 | −993 |
| Whitespace | −161 | +1,545 | +557 |
| Other | +44 | +507 | +80 |
| Net | +5,027 | +6,934 | +9,984 |

Relative to dedicated tokens that start on a Devanagari character, the extra rate is 15.2–17.7% for Hindi and 24.7–25.8% for Nepali, in every 900 KB replicate. In the 300 KB regime the corresponding means are 13.5% for Hindi and 21.2% for Nepali.

Latin spans move the other way. The shared vocabulary reduces Latin-starting tokens by about 14% inside the Hindi test and by about 39% inside the Nepali test. The Nepali dedicated model, trained without an English sample, spends about one token per embedded Latin letter; the shared model uses the English merges and shortens those spans. That saving lowers Nepali’s published penalty relative to the Devanagari-only rate.

**Measured:** Hindi and Nepali differ in virama rate, anusvara versus candrabindu, and the amount of embedded Latin. They do not differ in bytes per Devanagari character. Devanagari n-gram entropy is slightly lower for Nepali. The published penalty mixes a large positive Devanagari term with a negative Latin term, and the Latin saving is larger in the Nepali test.

**Interpretation:** UTF-8 expansion is a property of the script shared by Hindi and Nepali. It raises absolute token counts for both dedicated and shared models. It does not, by itself, account for Nepali’s higher ratio. The ratio gap is concentrated on tokens that start in Devanagari, and the headline percentages compress that gap because the shared model helps the Latin passages in the Nepali test.

## 4. Endings, word identity, and virama

Word-final two- and three-codepoint strings were counted on held-out words of at least four codepoints, with a minimum of 40 occurrences. For each ending, the rate of a token boundary at the start of that ending was compared with the rate of a token boundary at a typical codepoint boundary inside the same words. The quantity of interest is the excess: how much the shared model increases the ending boundary, beyond how much it increases boundaries in general.

For endings with at least 80 occurrences, the median excess is −0.046 in Nepali (47 endings) and −0.037 in Hindi (26 endings). The frequent Nepali endings `को` (excess +0.005), `मा` (−0.067), `का` (−0.094), and `लाई` (−0.047) sit near or below zero. `ले` is +0.057. A few endings are far above the center: Nepali `ेको` has excess +0.335 and `ेका` has +0.273 (331 and 170 occurrences). Those are measured outliers. The center of the distribution does not put extra cuts specifically at endings.

Exact word types were also split by whether they occur in the other Indic language’s training file from the same replicate, within character-length bins. In the 900 KB regime, Nepali words absent from the Hindi training file are most of the test and show overlap-token increases of about 14–17% in every length bin. Nepali words that also occur in the Hindi training file are fewer; the one- and two-character ones are shortened by the shared model (−8.6%). Longer shared types are not systematically easier than Nepali-only types. This comparison is an exact string match, so inflected forms of a shared lemma count as absent.

Words containing U+094D virama were scored the same way, on Devanagari-containing words only (`virama_stratification.csv`):

| Language | No virama | With virama |
| --- | ---: | ---: |
| Hindi, all lengths | +11.7% (13,809 occ.) | +7.3% (2,661 occ.) |
| Nepali, all lengths | +16.6% (11,061 occ.) | +19.3% (5,946 occ.) |

Inside Hindi, virama words have the smaller increase. Inside Nepali, virama words have a modestly larger increase. The Nepali–Hindi gap is present in the no-virama rows in every length bin from 3–4 characters upward (for example, 5–8 characters: Hindi +10.7%, Nepali +17.1%). It is wider among virama words (5–8 characters: Hindi +7.7%, Nepali +19.3%).

**Measured:** Extra boundaries are not concentrated at word ends. The typical frequent ending does not attract an excess boundary. Nepali’s higher word-level increase appears with and without virama. Virama words widen the gap; they do not create it.

**What is not established:** A morphological cause. A boundary before `ेको` is compatible with an inflectional cut, and also with a frequent character sequence that lost a merge. The design has no morphological parse, so the ending result stays descriptive.

## 5. Dedicated merges versus the shared vocabulary

This is the comparison that can be tied directly to the token-count gap.

Mean retention of each language’s 768 dedicated learned pieces by the shared vocabulary:

| Regime | Language | Type coverage | Rule coverage | Held-out emission coverage | Top-100 emission coverage | Penalty |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 900 KB | English | 49.2% | 48.0% | 79.0% | 97.2% | 14.22% |
| 900 KB | Hindi | 41.3% | 38.5% | 73.9% | 96.4% | 15.62% |
| 900 KB | Nepali | 39.8% | 37.5% | 73.8% | 97.5% | 18.61% |
| 300 KB | English | 46.9% | 45.1% | 77.0% | 94.0% | 12.52% |
| 300 KB | Hindi | 37.9% | 35.7% | 76.0% | 95.9% | 13.08% |
| 300 KB | Nepali | 37.3% | 34.6% | 75.4% | 97.3% | 16.19% |

Nepali retains slightly fewer dedicated types than Hindi. The fraction of held-out dedicated emissions whose byte string is in the shared vocabulary is the same for Hindi and Nepali in the 900 KB regime (73.9% and 73.8%) and nearly the same in the 300 KB regime (76.0% and 75.4%). The top 100 dedicated pieces by held-out frequency are almost all kept, and Nepali’s top-100 emission coverage is the highest of the three languages. The missing mass is in the middle and the tail, not in the most frequent dedicated merges.

On the held-out text, learned pieces are 76% of English dedicated tokens, 82% of Hindi, and 83% of Nepali. The missing learned emissions are 21.0%, 26.1%, and 26.2% of learned emissions in the 900 KB regime. Each missing emission produces, on average, 1.04, 1.07, and 1.16 extra shared boundaries. Nepali’s missing pieces are split a little finer. Hindi then offsets more of those extra cuts by tokens that cross a dedicated boundary, so its net extra tokens per missing emission are 0.73, against 0.86 for Nepali and 0.89 for English.

Where the extra cuts sit, 900 KB means:

| | English | Hindi | Nepali |
| --- | ---: | ---: | ---: |
| Net extra tokens | 5,027 | 6,934 | 9,984 |
| Extra shared boundaries | 6,214 | 10,967 | 14,595 |
| Of which, inside a dedicated token absent from the shared vocabulary | 5,854 | 10,172 | 13,537 |
| Of which, inside a dedicated token that is present | 360 | 795 | 1,058 |
| Dedicated boundaries crossed by a longer shared token | 1,187 | 4,033 | 4,611 |
| Aligned dedicated tokens that are present in the shared vocabulary and still fragmented | 0 | 0 | 0 |

The last row is zero in every language, every replicate, and both regimes. Whenever a dedicated token’s bytes are in the shared vocabulary and the shared segmentation respects that token’s endpoints, the shared model emits that token whole. About 93% of the extra boundaries fall inside dedicated tokens whose byte string was not learned by the shared model. The penalty is absence of those byte strings. It is not a failure to use strings the shared vocabulary already contains. The remaining few percent are unaligned spans, where a shared token crosses the dedicated token and the dedicated string, although present, is not the one selected.

The highest-frequency absent dedicated pieces are often short and often end in `⟨e0⟩⟨a4⟩`. On replicate 1 of the 900 KB Nepali model, the most frequent missing held-out pieces include `ि ⟨e0⟩⟨a4⟩` (176 emissions), `ित ⟨e0⟩⟨a4⟩` (134), `ीले ⟨e0⟩⟨a4⟩` (116), and the valid string `हरु` (85), which the shared model emits as `हर | ु`. Hindi’s list has the same shape: ` में ⟨e0⟩⟨a4⟩`, `ार ⟨e0⟩⟨a4⟩`, `ने` split into `न | े`. English’s absent pieces are ordinary strings such as `ence `, `mill `, and `tions `. The Indic dedicated models are learning “this character sequence, then the UTF-8 lead of whatever character comes next.” The shared model learns the lead itself as one of its first merges, and it does not keep every language-specific sequence glued to that lead.

Frequency-weighted coverage of training-byte n-grams inside Devanagari runs, 900 KB regime, shows the same length effect. At 3 bytes, shared coverage is 39.7% for Hindi and 37.7% for Nepali, close to each dedicated model. At 9 bytes, dedicated coverage is 5.0% (Hindi) and 7.5% (Nepali), while shared coverage falls to 2.6% and 3.6%. The dedicated-minus-shared gap at 9 bytes is 2.4 percentage points for Hindi and 3.9 for Nepali. Nepali’s dedicated tokenizer captures more of its longer Devanagari sequences, and the shared vocabulary returns less of that coverage.

**Measured:** Hindi and Nepali lose almost the same fraction of dedicated emission mass. Nepali’s lost pieces break into slightly more tokens, and Hindi recoups more of the gross cuts through crossing merges. The very frequent dedicated merges are kept for all three languages. Longer Devanagari sequences show a larger coverage gap for Nepali.

**Interpretation:** “Nepali loses a larger share of its high-frequency dedicated merges” does not fit these models. The penalty gap sits in how the missing mid-frequency strings are spread through a larger Devanagari token mass, how finely they split, and how much crossing structure the shared model still shares with Hindi.

## Answers

### 1. What measurable differences line up with Nepali’s higher penalty?

Four differences are measured in both regimes and line up with the ordering of the penalties.

- Tokens that start on a Devanagari character increase by about 25% for Nepali and about 16% for Hindi under the 900 KB models (about 21% and 14% under the 300 KB models). Every replicate keeps that order.
- The fraction of word occurrences that become longer is 47% for Nepali, 33% for Hindi, and 24% for English.
- Missing dedicated pieces produce slightly more extra cuts per missing emission in Nepali (1.16 versus 1.07).
- Nine-byte Devanagari sequences are covered more by the dedicated Nepali model than by the dedicated Hindi model, and the shared model gives a larger fraction of that coverage back to Hindi.

A fifth measured factor changes the headline in the other direction. Embedded Latin in the Nepali test is shortened by about 39% under the shared model, against about 14% in the Hindi test. The published 18.61% versus 15.62% gap is smaller than the Devanagari-only gap.

Nepali also has more virama per letter (129 versus 74 per 1,000). Words that contain virama have a wider cross-language gap than words that do not. Words without virama still show a Nepali excess in every length bin above two characters.

### 2. Is the difference vocabulary competition, fragmentation, UTF-8, or a combination?

It is a combination, and the parts are not equal.

Vocabulary competition with English is real at the level of slot counts: about half of the learned types are English-specific, and the first shared merges are the shared Devanagari UTF-8 leads. Once those leads exist, Hindi and Nepali draw on the same cross-lingual block. Nepali does not receive fewer language-specific slots than Hindi, and it does not lose a larger share of its frequent dedicated emissions.

Fragmentation is the direct form of the penalty. About 93% of extra boundaries lie in dedicated byte strings the shared vocabulary lacks. Aligned strings that the shared vocabulary does contain are not fragmented. Nepali’s fragmentation rate on Devanagari-starting tokens is higher, and it affects more words.

UTF-8 byte length does not explain the ratio. Devanagari is three bytes per character for both languages. The small difference in overall bytes per codepoint is the share of Devanagari versus ASCII in the test file. Both the dedicated and the shared tokenizer see that file. Boundary rates inside UTF-8 characters are similar for the two dedicated Indic models.

The combination that the artifacts support is: shared training keeps the common Devanagari frame and the highest-frequency language-specific pieces; it drops mid-frequency multi-byte sequences; those drops add tokens on Devanagari text; Nepali’s held-out Devanagari is lengthened more than Hindi’s; Latin passages in the tests move the published ratios toward each other.

### 3. Which findings are read directly from the saved artifacts?

- The published token counts and penalties, recomputed from the saved merges.
- The first four shared merges, identical in all six shared models.
- Allocation of learned byte strings by emission and by raw presence.
- Type, rule, and emission coverage of dedicated merges, including the top 50, 100, and 200.
- The boundary identity: where extra tokens are added and where shared tokens cross dedicated tokens, including the zero count of aligned present-but-fragmented tokens.
- Script composition, dependent-sign rates, n-gram entropy, and the script partition of the token difference.
- Word-level increase rates, the position of extra cuts, ending-boundary excesses, and the virama split.

### 4. Which explanations remain hypotheses?

- That Nepali morphology, as morphology, causes the penalty. The ending analysis does not show a general excess at frequent endings. A few sequences such as `ेको` are more often separated. That is compatible with inflection and compatible with an ordinary missing merge.
- That the higher virama rate is the main cause. It widens the gap and leaves most of the no-virama gap standing. The artifacts do not identify which conjuncts are responsible, and within Hindi the virama words are the less penalized group.
- That English “uses up” the merges Nepali needs, in a stronger sense than slot counts. English-specific pieces are rarely emitted on Nepali text. Showing that those slots would have become Nepali merges under a different training mixture would require a new training run. These files do not contain that run.
- Any claim about downstream model quality. These experiments count tokens only.

### 5. The most useful next experiment

Keep these saved models fixed and add the missing dedicated Nepali byte strings back into a copy of the shared merge table, in frequency order, without retraining. Re-encode the same held-out tests after each addition. If restoring Nepali’s missing mid-frequency strings removes the Devanagari gap relative to Hindi, absence of those strings is sufficient. If a gap remains after the string sets match, the remaining difference is in merge priority and in tokens that cross the original dedicated boundaries. A retraining run would answer a different question, about which strings joint training would have chosen instead, and these artifacts cannot answer it.

Matching the held-out samples on word length and virama is worth doing in that same re-encode, because the current tests are not matched: the Nepali test has more Devanagari letters, a higher virama rate, and a larger Latin saving under the shared model. The script partition already shows that the Devanagari gap survives that mixture. A matched slice would make the residual numerical comparison cleaner.

## Files

| Path | Contents |
| --- | --- |
| `run_analysis.py` | Loads the saved models and writes the tables and figures |
| `bpe_fast.py` | Encoder checked against the published token counts |
| `virama_stratification.py` | Virama and length split for the 900 KB models |
| `vocabulary_allocation.csv` | Shared-vocabulary class counts |
| `vocabulary_allocation_tokens.csv` | One row per learned shared token |
| `merge_coverage.csv` | Dedicated versus shared coverage |
| `merge_coverage_patterns.csv` | Highest-frequency missing dedicated pieces |
| `fragmentation_examples.csv` | Systematically selected word examples |
| `fragmentation_summary.csv` | Word-level aggregates and position bins |
| `unicode_analysis.csv` | Corpus profile and boundary rates |
| `penalty_decomposition.csv` | Boundary accounting |
| `script_penalty.csv` | Extra tokens by script at token start |
| `suffix_boundaries.csv` | Ending-boundary rates |
| `lexicon_fragmentation.csv` | Words present or absent in the other training file |
| `ngram_coverage.csv` | Byte n-gram coverage |
| `virama_stratification.csv` | Word increase by virama and length |
| `measured_summary.json` | Method record and held-out Unicode profile |
| `figures/*.svg` | Allocation, coverage, boundary account, script contribution, cut position, script mix, dependent signs |

Reproduce with:

```bash
.venv/bin/python analysis/nepali_penalty_investigation/run_analysis.py
.venv/bin/python analysis/nepali_penalty_investigation/virama_stratification.py
```
