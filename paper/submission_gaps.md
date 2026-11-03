# Submission gaps

This note records decisions that are still open after the 29 October 2026 revision of `paper/manuscript.tex`. The compiled file is `paper/manuscript.pdf` (7 pages, generic 10pt two-column article). It is not submission-ready for a specific venue.

## Author decisions

- `paper/manuscript.tex` and `paper/manuscript.pdf` are the named mentor-review draft: Nischal Poudel, BMS Institute of Technology and Management, Bengaluru, Karnataka, India, undergraduate student in Computer Science and Engineering (B.E., expected 2027). The title page says so. No email, ORCID, supervisor, co-author, funding source, or acknowledgement was added.
- `paper/manuscript_anonymized.tex` and `paper/manuscript_anonymized.pdf` keep the earlier placeholder author block. Use that copy for an anonymized submission, not the named draft.
- A venue is still unselected, so neither file is a conference submission.

## Venue-dependent requirements

No target conference has been selected. The manuscript uses a generic `article` class, not a venue style file. Before submission, the author still needs to settle:

- the style file and citation format required by that venue
- the page limit, and whether references count toward it
- anonymity, a responsible-NLP or ethics checklist, and any required limitations or broader-impact section
- supplementary-material rules, since the saved JSON, manifests, and figures live in the project directory rather than in a formatted appendix
- whether the current 7-page PDF still fits after the venue style is applied

The body runs through the start of page 6, and the references continue through page 7. That split is an observation about this PDF only.

## Citations

Every key in `paper/references.bib` is cited, and the compiled PDF shows references [1]–[10] with no undefined citations. `paper/references.bib` was not changed in this revision.

- Mahajan (reference [6]) is the citation requested by the Hugging Face dataset card for `ashtok897/indic-hplt-v2`, retrieved on 29 October 2026. It is not a peer-reviewed paper. No proceedings record was verified for it.
- HPLT 3.0 (reference [7]) is the arXiv preprint arXiv:2511.01066 (2025). This bibliography does not assign it a conference venue or a DOI.
- The other eight entries were checked against ACL Anthology records on 29 October 2026 and were left unchanged. This revision did not re-fetch those records.

## Methodology points that remain in the paper

These are stated limitations, not unresolved numerical conflicts.

- At vocabulary size 1,024 the shared condition is one 1,024-entry vocabulary. The three dedicated tokenizers together contain 3,072 entries. The experiment matches vocabulary size per tokenizer. An equal aggregate vocabulary budget was not run, so the results do not isolate sharing under that budget. The same per-tokenizer rule applies to the single-run sizes 512 and 2,048.
- The three evidence tiers stay separate: one vocabulary sweep at 512, 1,024, and 2,048; three resamples of about 300,000 bytes per language; and the total-budget-matched control of about 100,000 bytes per language.
- Percentage penalties and absolute token counts answer different questions. In the matched-budget regime the percentage penalty is smaller while both token counts are larger.
- The resamples overlap. The sample standard deviations are descriptive. No significance test is reported.
- The two budget regimes differ in sample size and in which documents are included.
- The 28 October 2026 stream has no stored revision, commit, or snapshot hash. `data/processed/` is the corpus of record.
- The cross-lingual-merge account in the introduction is offered as a possible explanation. The study did not test that mechanism.
