# Portfolio Results Summary: Fixed-Vocabulary Multilingual BPE

## Result

At vocabulary 1024, three saved approximately 300 KB-per-language resamples show shared-vocabulary token-count penalties of **14.22% ± 0.66 pp** (English), **15.62% ± 0.66 pp** (Hindi), and **18.61% ± 0.51 pp** (Nepali). In that regime each dedicated tokenizer trains on about 300 KB and the joint tokenizer on about 900 KB. Every saved resample produced more tokens with the shared vocabulary on its matched held-out test set.

The total-budget-matched control uses the same approximately 100 KB source sample for each corresponding dedicated and joint condition. Its joint tokenizer totals approximately 300 KB. Mean penalties remain positive: **12.52% ± 2.26 pp** (English), **13.08% ± 0.73 pp** (Hindi), and **16.19% ± 0.69 pp** (Nepali). These means are lower than in the approximately 900 KB joint regime, but absolute test-token counts are higher for both dedicated and joint models. The smaller percentage does not mean the joint tokenizer used fewer tokens, and the two regimes do not isolate training-budget size from sample size or composition.

The separate single-run 512/1024/2048 sweep also has positive penalties in all nine language-by-size observations. The sweep, the three-resample summary, and the budget comparison are separate evidence tiers.

This is a controlled empirical tokenization trade-off for this corpus and implementation, not evidence of statistical significance, algorithmic novelty, downstream-model impact, or universal multilingual behavior. Because resamples overlap, the reported standard deviations are descriptive only.

See `master_experiment_report.md`, `master_results_table.csv`, and `figures/` for the complete evidence.
