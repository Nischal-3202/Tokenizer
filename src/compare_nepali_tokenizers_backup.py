
import json
from pathlib import Path

from tokenizer.tokenizer import Tokenizer
from tokenizer.grapheme_bpe import GraphemeBPE

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

VOCAB_SIZE = 4096
GRAPHEME_VOCAB_SIZE = 4096
LANGUAGE = "nepali"


def read_text(filename):
    return (DATA_DIR / LANGUAGE / filename).read_text(encoding="utf-8")


def evaluate(tokenizer, text):
    token_ids = tokenizer.encode(text)
    original_bytes = len(text.encode("utf-8"))

    return {
        "characters": len(text),
        "utf8_bytes": original_bytes,
        "tokens": len(token_ids),
        "bytes_per_token": (
            original_bytes / len(token_ids) if token_ids else 0
        ),
        "tokens_per_1000_chars": (
            len(token_ids) / len(text) * 1000 if text else 0
        ),
        "roundtrip_exact": tokenizer.decode(token_ids) == text,
    }


def main():
    train_text = read_text("train.txt")
    test_text = read_text("test.txt")

    results = {
        "language": LANGUAGE,
        "byte_requested_vocab_size": VOCAB_SIZE,
        "grapheme_requested_vocab_size": GRAPHEME_VOCAB_SIZE,
        "models": {},
    }

    print("Training byte-level BPE...")
    byte_tokenizer = Tokenizer()
    byte_tokenizer.train(train_text, vocab_size=VOCAB_SIZE)

    results["models"]["byte_bpe"] = {
        "actual_vocab_size": byte_tokenizer.vocab_size,
        "merge_count": len(byte_tokenizer.merges),
        "test": evaluate(byte_tokenizer, test_text),
    }

    print("Training grapheme-level BPE...")
    grapheme_tokenizer = GraphemeBPE()
    grapheme_tokenizer.train(train_text, vocab_size=GRAPHEME_VOCAB_SIZE)

    results["models"]["grapheme_bpe"] = {
        "actual_vocab_size": grapheme_tokenizer.vocab_size,
        "base_vocab_size": len(grapheme_tokenizer.base_tokens),
        "merge_count": len(grapheme_tokenizer.merges),
        "test": evaluate(grapheme_tokenizer, test_text),
    }

    for name, model in results["models"].items():
        print(f"\n{name.upper()}")
        print("Actual vocabulary:", model["actual_vocab_size"])
        print("Learned merges:", model["merge_count"])
        if "base_vocab_size" in model:
            print("Base grapheme vocabulary:", model["base_vocab_size"])
        print("Test metrics:", model["test"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "nepali_byte_vs_grapheme_bpe.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
