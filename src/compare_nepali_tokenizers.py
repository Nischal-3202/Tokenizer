
import json
from pathlib import Path

from tokenizer.tokenizer import Tokenizer
from tokenizer.grapheme_bpe import GraphemeBPE
from tokenizer.akshara_bpe import AksharaBPE

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

VOCAB_SIZES = [4096,8192,16384]
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
        "vocab_sizes": VOCAB_SIZES,
        "models": {},
    }

    tokenizer_classes = {
        "byte_bpe": Tokenizer,
        "grapheme_bpe": GraphemeBPE,
        "akshara_bpe": AksharaBPE,
    }

    for name, tokenizer_class in tokenizer_classes.items():
        results["models"][name] = {}

        for vocab_size in VOCAB_SIZES:
            print(f"\nTraining {name} at vocab size {vocab_size}...")

            tokenizer = tokenizer_class()
            tokenizer.train(train_text, vocab_size=vocab_size)

            model_result = {
                "requested_vocab_size": vocab_size,
                "actual_vocab_size": tokenizer.vocab_size,
                "merge_count": len(tokenizer.merges),
                "test": evaluate(tokenizer, test_text),
            }

            if hasattr(tokenizer, "base_tokens"):
                model_result["base_vocab_size"] = len(tokenizer.base_tokens)

            results["models"][name][str(vocab_size)] = model_result

            print("Actual vocabulary:", tokenizer.vocab_size)
            print("Learned merges:", len(tokenizer.merges))
            print("Test metrics:", model_result["test"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "nepali_vocab_size_experiment.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
