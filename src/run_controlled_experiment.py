
import json
from pathlib import Path

from tokenizer.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
CONTROLLED_DIR = ROOT / "data" / "controlled"
RESULTS_DIR = ROOT / "results"

VOCAB_SIZE = 1024

LANGUAGES = ["hindi", "nepali"]


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


def train_and_evaluate(name, training_text):
    print(f"\nTraining {name}...")

    tokenizer = Tokenizer()
    tokenizer.train(training_text, vocab_size=VOCAB_SIZE)

    print("Vocabulary size:", tokenizer.vocab_size)
    print("Learned merges:", len(tokenizer.merges))

    results = {
        "actual_vocab_size": tokenizer.vocab_size,
        "merge_count": len(tokenizer.merges),
        "test": {},
    }

    for language in LANGUAGES:
        # A separate tokenizer is evaluated only on its own language.
        if name == "hindi_separate" and language != "hindi":
            continue
        if name == "nepali_separate" and language != "nepali":
            continue

        test_text = read_text(
            DATA_DIR / language / "test.txt"
        )
        metrics = evaluate(tokenizer, test_text)
        results["test"][language] = metrics

        print(f"{language.upper()} test metrics:", metrics)

    model_dir = RESULTS_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{name}_vocab_{VOCAB_SIZE}.json"
    tokenizer.save(model_path)

    print("Model saved to:", model_path)
    return results


def main():
    hindi_train = read_text(
        CONTROLLED_DIR / "hindi_train.txt"
    )
    nepali_train = read_text(
        CONTROLLED_DIR / "nepali_train.txt"
    )

    results = {
        "experiment": "controlled_hindi_nepali",
        "vocab_size": VOCAB_SIZE,
        "training": {
            "hindi_utf8_bytes": len(hindi_train.encode("utf-8")),
            "nepali_utf8_bytes": len(nepali_train.encode("utf-8")),
        },
        "models": {},
    }

    results["models"]["hindi_separate"] = train_and_evaluate(
        "hindi_separate",
        hindi_train,
    )

    results["models"]["nepali_separate"] = train_and_evaluate(
        "nepali_separate",
        nepali_train,
    )

    joint_train = hindi_train + "\n" + nepali_train

    results["training"]["joint_utf8_bytes"] = len(
        joint_train.encode("utf-8")
    )

    results["models"]["joint"] = train_and_evaluate(
        "hindi_nepali_joint",
        joint_train,
    )

    output_path = RESULTS_DIR / "controlled_hindi_nepali_vocab_1024.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
