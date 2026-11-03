"""
demo_tokenization.py
====================
Shows how a sentence is broken into BPE tokens for English, Hindi, and Nepali.
Uses the saved balanced-vocabulary-sweep vocab-1024 models (dedicated per-language).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tokenizer.tokenizer import Tokenizer

MODEL_DIR = ROOT / "results" / "models"

# ── model files ──────────────────────────────────────────────────────────────
MODELS = {
    "English": "balanced_vocabulary_sweep_v1_english_separate_vocab_1024.json",
    "Hindi":   "balanced_vocabulary_sweep_v1_hindi_separate_vocab_1024.json",
    "Nepali":  "balanced_vocabulary_sweep_v1_nepali_separate_vocab_1024.json",
}

# ── demo sentences ───────────────────────────────────────────────────────────
SENTENCES = {
    "English": "The language model tokenizes text into subword units.",
    "Hindi":   "भाषा मॉडल पाठ को उपशब्द इकाइयों में विभाजित करता है।",
    "Nepali":  "भाषा मोडेलले पाठलाई उपशब्द एकाइहरूमा विभाजित गर्दछ।",
}

SEP = "─" * 72


def fmt_token(tok_bytes: bytes) -> str:
    try:
        text = tok_bytes.decode("utf-8")
        # show spaces visually
        return repr(text)
    except UnicodeDecodeError:
        return tok_bytes.hex()


def demo(language: str, model_file: str, sentence: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {language.upper()}")
    print(f"{'═' * 72}")

    tok = Tokenizer()
    tok.load(MODEL_DIR / model_file)

    print(f"\n  Input sentence:\n    {sentence!r}")
    print(f"\n  UTF-8 bytes : {len(sentence.encode('utf-8'))}")
    print(f"  Characters : {len(sentence)}")
    print(f"  Vocab size : {tok.vocab_size}")

    # ── encode ───────────────────────────────────────────────────────────────
    token_ids = tok.encode(sentence)
    token_bytes = [tok.vocab[tid] for tid in token_ids]

    print(f"\n  Token count : {len(token_ids)}")
    print(f"  Bytes/token : {len(sentence.encode('utf-8')) / len(token_ids):.2f}")

    # ── show each token ───────────────────────────────────────────────────────
    print(f"\n  {'#':<5} {'ID':<8} {'Bytes':>6}  Token")
    print(f"  {SEP[:65]}")
    for i, (tid, tb) in enumerate(zip(token_ids, token_bytes), 1):
        kind = "merged" if tid >= 256 else "byte"
        print(f"  {i:<5} {tid:<8} {len(tb):>6}  {fmt_token(tb)}  [{kind}]")

    # ── reconstruct ──────────────────────────────────────────────────────────
    decoded = tok.decode(token_ids)
    match = decoded == sentence
    print(f"\n  Roundtrip exact: {match}")
    if not match:
        print(f"  Decoded: {decoded!r}")


def main():
    print("\nBPE TOKENIZATION DEMO  —  English · Hindi · Nepali  (vocab 1024)")
    for lang, model_file in MODELS.items():
        demo(lang, model_file, SENTENCES[lang])
    print(f"\n{'═' * 72}\n")


if __name__ == "__main__":
    main()
