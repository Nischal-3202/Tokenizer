# Tokenizer Research

A research implementation of Byte-Pair Encoding (BPE) tokenization for **English**, **Hindi**, and **Nepali**, exploring how vocabulary sharing between languages affects tokenization efficiency.

## Overview

This project investigates the *tokenization penalty* — the increase in token count when a single shared (joint) tokenizer is used instead of dedicated per-language tokenizers. A key finding is that Nepali consistently suffers the largest penalty (~16–19%) due to Devanagari script fragmentation, while English sees the smallest (~12–14%).

Three tokenizer variants are implemented:

| Tokenizer | Segmentation unit | Best suited for |
|---|---|---|
| `Tokenizer` (byte-BPE) | Raw UTF-8 bytes | General / English |
| `GraphemeBPE` | Unicode grapheme clusters | Script-aware multilingual |
| `AksharaBPE` | Devanagari akshara units | Hindi / Nepali |

## Repository Layout

```
Tokenizer/
├── src/
│   ├── tokenizer/              # Core library
│   │   ├── tokenizer.py        # Byte-BPE Tokenizer class
│   │   ├── bpe.py              # BPE train / encode / decode primitives
│   │   ├── grapheme_bpe.py     # GraphemeBPE — grapheme-cluster segmentation
│   │   ├── akshara_bpe.py      # AksharaBPE — Devanagari akshara segmentation
│   │   ├── aksharas.py         # Devanagari akshara splitter
│   │   ├── graphemes.py        # Unicode grapheme cluster splitter
│   │   ├── metrics.py          # Evaluation helpers (tokens, bytes/token, roundtrip)
│   │   └── unicode_utils.py    # Unicode utility functions
│   ├── run_baseline.py                        # Single-language baseline experiments
│   ├── run_joint_experiment.py                # Joint (shared) vocabulary experiments
│   ├── run_balanced_vocabulary_experiment.py  # Balanced corpus vocabulary experiments
│   ├── run_balanced_vocabulary_sweep.py       # Vocabulary size sweep (512–2048)
│   ├── run_total_budget_matched_experiment.py # Budget-controlled resampling
│   ├── compare_baselines.py                   # Cross-tokenizer comparison
│   ├── demo_tokenization.py                   # Interactive tokenization demo
│   ├── collect_corpus.py                      # Corpus collection utilities
│   └── ...                                    # Other analysis/preparation scripts
├── analysis/
│   ├── nepali_penalty_investigation/          # Root-cause analysis of Nepali penalty
│   └── counterfactual_vocabulary_augmentation/# Counterfactual augmentation experiments
├── results/                                   # Saved experiment outputs (JSON/CSV/MD)
├── tests/                                     # Pytest test suite
├── paper/                                     # LaTeX manuscript and figures
├── requirements.txt
└── pyproject.toml
```

## Installation

**Python 3.12 is required.**

```bash
# Clone the repository
git clone <repo-url>
cd Tokenizer

# Install in editable mode (recommended)
pip install -e .

# Install additional research dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic byte-BPE tokenizer

```python
from tokenizer.tokenizer import Tokenizer

tok = Tokenizer()
tok.train("your training text here", vocab_size=1024)

ids = tok.encode("Hello, world!")
print(ids)

text = tok.decode(ids)
print(text)  # Hello, world!

# Save and reload
tok.save("my_tokenizer.json")
tok.load("my_tokenizer.json")
```

### Akshara-aware tokenizer for Nepali/Hindi

```python
from tokenizer.akshara_bpe import AksharaBPE

tok = AksharaBPE()
tok.train("नेपाली पाठ यहाँ राख्नुहोस्", vocab_size=512)

ids = tok.encode("नेपाली")
print(tok.decode(ids))  # नेपाली
```

### Grapheme-cluster tokenizer

```python
from tokenizer.grapheme_bpe import GraphemeBPE

tok = GraphemeBPE()
tok.train(training_text, vocab_size=1024)
ids = tok.encode("text")
```

### Evaluate a tokenizer

```python
from tokenizer.metrics import evaluate_tokenizer

results = evaluate_tokenizer(tok, test_text)
# {
#   "characters": 500,
#   "utf8_bytes": 1500,
#   "tokens": 320,
#   "bytes_per_token": 4.69,
#   "tokens_per_1000_chars": 640.0,
#   "roundtrip_exact": True
# }
```

## Running the Demo

```bash
python src/demo_tokenization.py
```

This displays side-by-side tokenization of English, Hindi, and Nepali sentences using pre-trained vocab-1024 models, showing each token's ID, byte length, and whether it was a base byte or a learned merge.

## Running Experiments

```bash
# Baseline — dedicated per-language tokenizers
python src/run_baseline.py

# Joint tokenizer — shared vocabulary across all three languages
python src/run_joint_experiment.py

# Vocabulary size sweep (512, 1024, 2048)
python src/run_balanced_vocabulary_sweep.py

# Budget-matched control (same training bytes for dedicated vs. joint)
python src/run_total_budget_matched_experiment.py
```

Results are written to `results/` as JSON files and summarized in `results/master_experiment_report.md`.

## Running Tests

```bash
pytest tests/
```

The test suite covers BPE primitives, akshara segmentation, grapheme splitting, and Unicode utilities.

## Key Results

| Language | Mean vocabulary-sharing penalty | Notes |
|---|---|---|
| **English** | ~12.5% | Smallest penalty |
| **Hindi** | ~13.1% | Mid-range |
| **Nepali** | ~16.2% | Largest penalty — Devanagari fragmentation |

*Penalty = (joint tokens − dedicated tokens) / dedicated tokens × 100*

The penalty is consistent across vocabulary sizes (512–2048) and budget regimes. See [`results/master_experiment_report.md`](results/master_experiment_report.md) and the [`analysis/nepali_penalty_investigation/`](analysis/nepali_penalty_investigation/) directory for a detailed root-cause breakdown.

## Paper

The LaTeX manuscript and figures are in [`paper/`](paper/). The anonymized submission PDF is `paper/manuscript_anonymized.pdf`.

## Dependencies

| Package | Purpose |
|---|---|
| `regex` | Unicode-aware grapheme cluster splitting |
| `tiktoken` | Baseline comparison tokenizer |
| `pytest` | Test runner |
| `jupyter` / `ipykernel` | Interactive analysis notebooks |

## License

See repository root for license information.
