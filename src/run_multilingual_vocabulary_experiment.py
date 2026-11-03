
import json
import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from tokenizer.tokenizer import Tokenizer
from tokenizer.metrics import evaluate_tokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
CONTROLLED_DIR = ROOT / "data" / "controlled_multilingual"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = RESULTS_DIR / "models"

LANGUAGES = ["english", "hindi", "nepali"]
LANGUAGE_CODES = {
    "english": "en",
    "hindi": "hi",
    "nepali": "ne",
}

VOCAB_SIZE = 1024
EXPERIMENT_NAME = "multilingual_vocabulary_sharing_v1"

OUTPUT_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}.json"


def read_text(path):
    return path.read_text(encoding="utf-8")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def train_model(name, training_text):
    print(f"\nTraining {name}...")

    tokenizer = Tokenizer()
    tokenizer.train(training_text, vocab_size=VOCAB_SIZE)

    print("Actual vocabulary size:", tokenizer.vocab_size)
    print("Merge count:", len(tokenizer.merges))

    return tokenizer


def evaluate_model(tokenizer, language):
    test_path = DATA_DIR / language / "test.txt"
    test_text = read_text(test_path)

    metrics = evaluate_tokenizer(tokenizer, test_text)

    return {
        "test_sha256": sha256_text(test_text),
        "metrics": metrics,
    }


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite {OUTPUT_PATH}. "
            "Change EXPERIMENT_NAME to create a new output."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_data = {
        language: read_text(
            CONTROLLED_DIR / f"{language}_train.txt"
        )
        for language in LANGUAGES
    }

    results = {
        "experiment": EXPERIMENT_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "algorithm": "byte_level_bpe",
        "requested_vocab_size": VOCAB_SIZE,
        "training_data": {},
        "models": {},
    }

    for language, text in training_data.items():
        results["training_data"][language] = {
            "code_points": len(text),
            "utf8_bytes": len(text.encode("utf-8")),
            "sha256": sha256_text(text),
        }

    joint_text = "\n".join(
        training_data[language] for language in LANGUAGES
    )

    training_conditions = {
        "english_separate": training_data["english"],
        "hindi_separate": training_data["hindi"],
        "nepali_separate": training_data["nepali"],
        "shared_multilingual": joint_text,
    }

    for name, train_text in training_conditions.items():
        tokenizer = train_model(name, train_text)

        model_path = MODEL_DIR / (
            f"{EXPERIMENT_NAME}_{name}_vocab_{VOCAB_SIZE}.json"
        )
        tokenizer.save(model_path)

        model_results = {
            "actual_vocab_size": tokenizer.vocab_size,
            "merge_count": len(tokenizer.merges),
            "training_code_points": len(train_text),
            "training_utf8_bytes": len(train_text.encode("utf-8")),
            "training_sha256": sha256_text(train_text),
            "model_file": str(model_path.relative_to(ROOT)),
            "test": {},
        }

        # Every model is evaluated on all three languages.
        for language in LANGUAGES:
            evaluation = evaluate_model(tokenizer, language)
            model_results["test"][language] = evaluation

            print(
                f"{name} -> {language}:",
                evaluation["metrics"],
            )

        results["models"][name] = model_results

    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nExperiment complete.")
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
