"""Stratify held-out word token counts by virama and word length.

Uses the saved repeated_vocabulary_sharing_v1 tokenizers. A word is a
whitespace-delimited span that contains at least one Devanagari codepoint.
Token counts are overlapping pieces of the full-document segmentation.
This does not retrain anything.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bpe_fast import encode_bytes, token_spans
from tokenizer.tokenizer import Tokenizer

OUT = Path(__file__).resolve().parent / "virama_stratification.csv"
LANGS = ("hindi", "nepali")
REPS = ("replicate_1", "replicate_2", "replicate_3")


def word_spans(text: str):
    spans = []
    byte_i = 0
    buf = []
    start = 0
    for ch in text:
        size = len(ch.encode("utf-8"))
        if ch.isspace():
            if buf:
                spans.append((start, byte_i, "".join(buf)))
                buf = []
        else:
            if not buf:
                start = byte_i
            buf.append(ch)
        byte_i += size
    if buf:
        spans.append((start, byte_i, "".join(buf)))
    return spans


def overlap_counts(words, spans):
    counts = []
    j = 0
    for start, end, _word in words:
        while j < len(spans) and spans[j][1] <= start:
            j += 1
        k = j
        n = 0
        while k < len(spans) and spans[k][0] < end:
            n += 1
            k += 1
        counts.append(n)
    return counts


def length_bin(n: int) -> str:
    if n <= 2:
        return "1-2"
    if n <= 4:
        return "3-4"
    if n <= 8:
        return "5-8"
    return "9+"


def main():
    rows = []
    tests = {
        lang: (ROOT / "data" / "processed" / lang / "test.txt").read_text(encoding="utf-8")
        for lang in LANGS
    }
    for rep in REPS:
        dedicated = {}
        for lang in LANGS:
            path = (
                ROOT
                / "results"
                / "models"
                / f"repeated_vocabulary_sharing_v1_{rep}_{lang}_separate_vocab_1024.json"
            )
            tok = Tokenizer()
            tok.load(path)
            dedicated[lang] = tok
        shared = Tokenizer()
        shared.load(
            ROOT
            / "results"
            / "models"
            / f"repeated_vocabulary_sharing_v1_{rep}_shared_multilingual_vocab_1024.json"
        )
        for lang in LANGS:
            text = tests[lang]
            data = text.encode("utf-8")
            d_spans = token_spans(encode_bytes(data, dedicated[lang].merges), dedicated[lang].vocab)
            s_spans = token_spans(encode_bytes(data, shared.merges), shared.vocab)
            words = word_spans(text)
            d_counts = overlap_counts(words, d_spans)
            s_counts = overlap_counts(words, s_spans)
            acc = defaultdict(lambda: [0, 0, 0])
            for (_s, _e, word), d_n, s_n in zip(words, d_counts, s_counts):
                if not any("\u0900" <= ch <= "\u097f" for ch in word):
                    continue
                key = (
                    length_bin(len(word)),
                    "with_virama" if "\u094d" in word else "no_virama",
                )
                acc[key][0] += 1
                acc[key][1] += d_n
                acc[key][2] += s_n
            for (bin_name, virama), (n, d_n, s_n) in sorted(acc.items()):
                rows.append(
                    {
                        "experiment": "repeated_vocabulary_sharing_v1",
                        "replicate": rep,
                        "language": lang,
                        "length_bin": bin_name,
                        "virama": virama,
                        "word_occurrences": n,
                        "dedicated_overlap_tokens": d_n,
                        "shared_overlap_tokens": s_n,
                        "percent_increase": (s_n - d_n) / d_n * 100 if d_n else "",
                    }
                )
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
