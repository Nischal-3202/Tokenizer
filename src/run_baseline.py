
import json
from pathlib import Path

from tokenizer.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]

LANGUAGES = {
    "en": "english",
    "hi": "hindi",
    "ne": "nepali",
}

VOCAB_SIZE = 2048


def read_text(path):
    return path.read_text(encoding="utf-8")


def evaluate(tokenizer, text):
    token_ids = tokenizer.encode(text)
    original_bytes = len(text.encode("utf-8"))

    return {
        "characters": len(text),
        "utf8_bytes": original_bytes,
        "tokens": len(token_ids),
        "bytes_per_token": (
            original_bytes / len(token_ids)
            if token_ids else 0
        ),
        "tokens_per_1000_chars": (
            len(token_ids) / len(text) * 1000
            if text else 0
        ),
        "roundtrip_exact": tokenizer.decode(token_ids) == text,
    }


def main():
    results = {}

    for code, language in LANGUAGES.items():
        print(f"\nTraining {language} tokenizer...")

        folder = ROOT / "data" / "processed" / language
        train_text = read_text(folder / "train.txt")
        test_text = read_text(folder / "test.txt")

        tokenizer = Tokenizer()
        tokenizer.train(train_text, vocab_size=VOCAB_SIZE)

        train_metrics = evaluate(tokenizer, train_text)
        test_metrics = evaluate(tokenizer, test_text)

        results[code] = {
            "language": language,
            "requested_vocab_size": VOCAB_SIZE,
            "actual_vocab_size": tokenizer.vocab_size,
            "merge_count": len(tokenizer.merges),
            "train": train_metrics,
            "test": test_metrics,
        }

        print("Actual vocabulary size:", tokenizer.vocab_size)
        print("Learned merges:", len(tokenizer.merges))
        print("Test metrics:", test_metrics)

        model_dir = ROOT / "results" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(model_dir / f"{language}_bpe.json")

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "baseline_results.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
