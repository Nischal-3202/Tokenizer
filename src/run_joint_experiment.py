
import json
from pathlib import Path

from tokenizer.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 1024

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

LANGUAGES = {
    "hindi": "hi",
    "nepali": "ne",
}


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
    print("Loading Hindi and Nepali training data...")

    hindi_train = read_text(DATA_DIR / "hindi" / "train.txt")
    nepali_train = read_text(DATA_DIR / "nepali" / "train.txt")

    # Train one tokenizer on both languages.
    joint_train = hindi_train + "\n" + nepali_train

    print("Hindi training characters:", len(hindi_train))
    print("Nepali training characters:", len(nepali_train))
    print("Combined training characters:", len(joint_train))

    print(f"\nTraining joint tokenizer (vocab size {VOCAB_SIZE})...")

    tokenizer = Tokenizer()
    tokenizer.train(joint_train, vocab_size=VOCAB_SIZE)

    print("Actual vocabulary size:", tokenizer.vocab_size)
    print("Learned merges:", len(tokenizer.merges))

    model_dir = RESULTS_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "hindi_nepali_joint_bpe_1024.json"
    tokenizer.save(model_path)

    results = {
        "experiment": "joint_hindi_nepali",
        "requested_vocab_size": VOCAB_SIZE,
        "actual_vocab_size": tokenizer.vocab_size,
        "merge_count": len(tokenizer.merges),
        "training_characters": {
            "hindi": len(hindi_train),
            "nepali": len(nepali_train),
            "combined": len(joint_train),
        },
        "test": {},
    }

    for language, code in LANGUAGES.items():
        test_path = DATA_DIR / language / "test.txt"
        test_text = read_text(test_path)

        metrics = evaluate(tokenizer, test_text)
        results["test"][language] = metrics

        print(f"\n{language.upper()} test metrics:")
        print(metrics)

    output_path = RESULTS_DIR / "joint_hindi_nepali_vocab_1024.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nModel saved to:", model_path)
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()
