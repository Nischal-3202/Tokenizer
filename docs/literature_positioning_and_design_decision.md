# Literature Positioning and Research-Design Decision

**Status:** literature review completed 2026-10-29; no new tokenizer training run.

## Current evidence (this project)

This project evaluates a custom byte-level BPE implementation at fixed requested
vocabulary sizes. Each dedicated tokenizer is trained on approximately 300 KB of
one language (English, Hindi, or Nepali); the shared tokenizer is trained on the
concatenation, approximately 900 KB, with the same requested vocabulary size.
On the fixed held-out sets, the shared 1024-token vocabulary used more tokens in
every saved resample: mean penalties were 14.22% (English), 15.62% (Hindi), and
18.61% (Nepali), with descriptive sample SDs of 0.66, 0.66, and 0.51 percentage
points respectively. See `results/master_experiment_report.md` for exact counts,
resampling caveats, and reconstruction checks.

This is an intrinsic compression result for this corpus and implementation. It
does not test downstream language-model quality, transfer, morphology, or a
general property of multilingual tokenizers.

## What prior work establishes

1. BPE is a standard subword method for open-vocabulary NLP, but it is not itself
   a new research contribution here [Sennrich et al., 2016](https://aclanthology.org/P16-1162/).
2. Multilingual vocabulary capacity and language-specific coverage are established
   issues. Clustered vocabularies can improve multilingual task results without
   increasing model or data size [Chung et al., 2020](https://aclanthology.org/2020.emnlp-main.367/);
   VoCap explicitly allocates capacity by language [Zheng et al., 2021](https://aclanthology.org/2021.emnlp-main.257/);
   and XLM-V argues that weak allocation and indiscriminate sharing create a
   vocabulary bottleneck [Liang et al., 2023](https://aclanthology.org/2023.emnlp-main.813/).
3. Vocabulary overlap has task-dependent effects rather than a universal benefit
   or harm: it may aid NER and sentence-level cross-lingual tasks while harming
   POS and dependency-labeling outcomes [Limisiewicz et al., 2023](https://aclanthology.org/2023.findings-acl.350/).
   Conversely, targeted overlap among related languages can improve transfer
   [Patil et al., 2022](https://aclanthology.org/2022.acl-long.18/).
4. The Indic-specific gap is narrower than it first appears. A 2026 study over 17
   Indic languages already evaluates BPE versus Unigram, normalization,
   vocabulary size, and joint versus cluster-based construction; it reports
   improvements from script-specific normalization and cluster-based construction
   in downstream tasks [Brahma et al., 2026](https://aclanthology.org/2026.findings-acl.1632/).

## Positioning

The project should **not** claim that shared multilingual vocabularies, vocabulary
allocation, byte-level BPE, or Indic tokenizer design are novel. A defensible
narrow contribution is a reproducible, small-budget English--Hindi--Nepali case
study: under a fixed small vocabulary and approximately equal *per-language*
training text, a fully joint byte-BPE vocabulary compressed each evaluated
language less efficiently than a dedicated vocabulary. The Hindi--Nepali
Devanagari pairing makes this a useful negative result about script sharing alone:
shared script is not sufficient to guarantee lower intrinsic token counts.

That framing remains corpus-, implementation-, budget-, and metric-specific. It
should be presented as evidence complementary to, not competing with, large-scale
downstream studies. In particular, intrinsic token count cannot predict transfer
or task quality by itself; this is consistent with the task-dependent findings of
Limisiewicz et al. (2023) and the tokenizer/downstream mismatch documented by
[Ali et al., 2024](https://aclanthology.org/2024.findings-naacl.247/).

## Design decision: add one budget-matched control before paper claims

**Decision: necessary for any claim that contrasts shared and dedicated training
under a fair total tokenizer-training-data budget.** The current main comparison
holds per-language bytes approximately constant, but triples the shared
tokenizer's total training bytes (about 900 KB versus 300 KB). It therefore
isolates neither vocabulary sharing nor total training-data exposure. The current
result is still valid for its stated fixed-per-language setup, and should remain
reported rather than replaced.

### Minimal follow-up (do not run until approved)

At vocabulary 1024, repeat the existing three-resample design with matched
100 KB source samples for both conditions per replicate:

| Condition | English | Hindi | Nepali | Total |
| --- | ---: | ---: | ---: | ---: |
| Dedicated, matched (new) | 100 KB | -- | -- | 100 KB per model |
| Joint, existing | 300 KB | 300 KB | 300 KB | 900 KB |
| Joint, total-budget matched (new) | 100 KB | 100 KB | 100 KB | 300 KB |

Use the same preprocessing, byte-BPE code, requested vocabulary, and fixed
held-out tests. Generate deterministic document-boundary 100 KB subsets from each
existing 300 KB training selection, save hashes and selection manifests, and
train each dedicated tokenizer on its language's same subset. The planned control
is intentionally only at 1024 with the existing three resamples; a full new
512/1024/2048 sweep is not needed for the decision.

### Interpretation rule

The joint tokenizer holds **total** training text constant at approximately 300
KB. Each corresponding dedicated model uses the joint condition's exact source
sample for its language (approximately 100 KB). This does not make every notion
of fairness identical: the dedicated condition is three separate 100 KB models,
while the joint condition is one 300 KB model. Its purpose is precise: establish
whether the observed shared-vocabulary penalty persists without unmatched source
samples. The existing 900 KB joint condition answers the complementary
fixed-per-language question. Report both conditions and do not attribute a change
between them solely to vocabulary sharing.

If the project remains a portfolio case study with the narrow fixed-per-language
claim above, the existing results suffice. If it is framed as a research paper or
as evidence about shared versus dedicated tokenizers generally, run this minimal
control first; it is the highest-value remaining experiment. A downstream model
experiment is a separate, substantially broader study and is not required for
this intrinsic-tokenization paper.

## References

- Ali et al. (2024). [Tokenizer Choice For LLM Training: Negligible or Crucial?](https://aclanthology.org/2024.findings-naacl.247/)
- Brahma et al. (2026). [Multilingual Tokenization through the Lens of Indian Languages: Challenges and Insights](https://aclanthology.org/2026.findings-acl.1632/)
- Chung et al. (2020). [Improving Multilingual Models with Language-Clustered Vocabularies](https://aclanthology.org/2020.emnlp-main.367/)
- Liang et al. (2023). [XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models](https://aclanthology.org/2023.emnlp-main.813/)
- Limisiewicz, Balhar, and Marecek (2023). [Tokenization Impacts Multilingual Language Modeling](https://aclanthology.org/2023.findings-acl.350/)
- Patil, Talukdar, and Sarawagi (2022). [Overlap-based Vocabulary Generation Improves Cross-lingual Transfer Among Related Languages](https://aclanthology.org/2022.acl-long.18/)
- Sennrich, Haddow, and Birch (2016). [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)
- Zheng et al. (2021). [Allocating Large Vocabulary Capacity for Cross-Lingual Language Model Pre-Training](https://aclanthology.org/2021.emnlp-main.257/)
