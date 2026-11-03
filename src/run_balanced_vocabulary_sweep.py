
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from tokenizer.tokenizer import Tokenizer
from tokenizer.metrics import evaluate_tokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
CONTROLLED_DIR = ROOT / "data" / "controlled_multilingual_balanced"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = RESULTS_DIR / "models"

LANGUAGES = ["english", "hindi", "nepali"]
VOCAB_SIZES = [512, 1024, 2048]

EXPERIMENT_NAME = "balanced_vocabulary_sweep_v1"
OUTPUT_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}.json"


def read_text(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def train_model(name, text, vocab_size):
    print(f"\nTraining {name}, vocabulary={vocab_size}...")

    tokenizer = Tokenizer()
    tokenizer.train(text, vocab_size=vocab_size)

    print("Actual vocabulary size:", tokenizer.vocab_size)
    print("Merge count:", len(tokenizer.merges))

    return tokenizer


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite {OUTPUT_PATH}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_data = {
        language: read_text(
            CONTROLLED_DIR / f"{language}_train.txt"
        )
        for language in LANGUAGES
    }

    test_data = {
        language: read_text(
            DATA_DIR / language / "test.txt"
        )
        for language in LANGUAGES
    }

    joint_text = "\n".join(
        training_data[language] for language in LANGUAGES
    )

    conditions = {
        "english_separate": training_data["english"],
        "hindi_separate": training_data["hindi"],
        "nepali_separate": training_data["nepali"],
        "shared_multilingual": joint_text,
    }

    planned_paths = []
    for vocab_size in VOCAB_SIZES:
        for name in conditions:
            planned_paths.append(
                MODEL_DIR
                / f"{EXPERIMENT_NAME}_{name}_vocab_{vocab_size}.json"
            )

    existing = [path for path in planned_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing model files:\n"
            + "\n".join(str(path) for path in existing)
        )

    results = {
        "experiment": EXPERIMENT_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "algorithm": "byte_level_bpe",
        "vocab_sizes_requested": VOCAB_SIZES,
        "selection_manifest": (
            "data/manifests/"
            "controlled_multilingual_balanced_v1.json"
        ),
        "training_data": {},
        "test_data": {},
        "models": {},
    }

    for language, text in training_data.items():
        results["training_data"][language] = {
            "utf8_bytes": len(text.encode("utf-8")),
            "code_points": len(text),
            "sha256": sha256_text(text),
        }

    for language, text in test_data.items():
        results["test_data"][language] = {
            "utf8_bytes": len(text.encode("utf-8")),
            "code_points": len(text),
            "sha256": sha256_text(text),
        }

    for vocab_size in VOCAB_SIZES:
        results["models"][str(vocab_size)] = {}

        for name, train_text in conditions.items():
            tokenizer = train_model(name, train_text, vocab_size)

            model_path = (
                MODEL_DIR
                / f"{EXPERIMENT_NAME}_{name}_vocab_{vocab_size}.json"
            )
            tokenizer.save(model_path)

            model_result = {
                "requested_vocab_size": vocab_size,
                "actual_vocab_size": tokenizer.vocab_size,
                "merge_count": len(tokenizer.merges),
                "training_utf8_bytes": len(train_text.encode("utf-8")),
                "training_sha256": sha256_text(train_text),
                "model_file": str(model_path.relative_to(ROOT)),
                "test": {},
            }

            for language, test_text in test_data.items():
                metrics = evaluate_tokenizer(tokenizer, test_text)

                model_result["test"][language] = {
                    "test_sha256": sha256_text(test_text),
                    "metrics": metrics,
                }

                print(
                    f"{name} -> {language} "
                    f"(vocab={vocab_size}): {metrics}"
                )

            results["models"][str(vocab_size)][name] = model_result

    # Exclusive creation protects existing results.
    with OUTPUT_PATH.open("x", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("\nVocabulary sweep complete.")
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
