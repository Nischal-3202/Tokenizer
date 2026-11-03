
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from tokenizer.tokenizer import Tokenizer
from tokenizer.metrics import evaluate_tokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
CORPUS_DIR = ROOT / "data" / "controlled_multilingual_replicates"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = RESULTS_DIR / "models"

LANGUAGES = ["english", "hindi", "nepali"]
REPLICATES = ["replicate_1", "replicate_2", "replicate_3"]
VOCAB_SIZE = 1024

EXPERIMENT_NAME = "repeated_vocabulary_sharing_v1"
OUTPUT_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}.json"
SUMMARY_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}_summary.json"


def read_text(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
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


def main():
    for path in [OUTPUT_PATH, SUMMARY_PATH]:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load fixed held-out test sets once.
    test_data = {
        language: read_text(DATA_DIR / language / "test.txt")
        for language in LANGUAGES
    }

    results = {
        "experiment": EXPERIMENT_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "algorithm": "byte_level_bpe",
        "requested_vocab_size": VOCAB_SIZE,
        "replicates": {},
        "test_data": {},
    }

    for language, text in test_data.items():
        results["test_data"][language] = {
            "utf8_bytes": len(text.encode("utf-8")),
            "code_points": len(text),
            "sha256": sha256_text(text),
        }

    # Store token counts for the statistical summary.
    token_counts = {
        replicate: {}
        for replicate in REPLICATES
    }

    # Check all planned model paths before expensive training.
    planned_paths = []
    for replicate in REPLICATES:
        for condition in [
            "english_separate",
            "hindi_separate",
            "nepali_separate",
            "shared_multilingual",
        ]:
            planned_paths.append(
                MODEL_DIR
                / (
                    f"{EXPERIMENT_NAME}_{replicate}_"
                    f"{condition}_vocab_{VOCAB_SIZE}.json"
                )
            )

    existing = [path for path in planned_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing models:\n"
            + "\n".join(str(path) for path in existing)
        )

    for replicate in REPLICATES:
        print(f"\n{'=' * 60}\n{replicate}\n{'=' * 60}")

        training_data = {
            language: read_text(
                CORPUS_DIR / replicate / f"{language}_train.txt"
            )
            for language in LANGUAGES
        }

        conditions = {
            "english_separate": training_data["english"],
            "hindi_separate": training_data["hindi"],
            "nepali_separate": training_data["nepali"],
            "shared_multilingual": "\n".join(
                training_data[language] for language in LANGUAGES
            ),
        }

        results["replicates"][replicate] = {
            "training_data": {},
            "models": {},
        }

        for language, text in training_data.items():
            results["replicates"][replicate]["training_data"][language] = {
                "utf8_bytes": len(text.encode("utf-8")),
                "code_points": len(text),
                "sha256": sha256_text(text),
            }

        token_counts[replicate] = {}

        for condition, training_text in conditions.items():
            tokenizer = train_model(condition, training_text)

            model_path = (
                MODEL_DIR
                / (
                    f"{EXPERIMENT_NAME}_{replicate}_"
                    f"{condition}_vocab_{VOCAB_SIZE}.json"
                )
            )
            tokenizer.save(model_path)

            model_result = {
                "actual_vocab_size": tokenizer.vocab_size,
                "merge_count": len(tokenizer.merges),
                "training_utf8_bytes": len(
                    training_text.encode("utf-8")
                ),
                "training_sha256": sha256_text(training_text),
                "model_file": str(model_path.relative_to(ROOT)),
                "test": {},
            }

            token_counts[replicate][condition] = {}

            for language, test_text in test_data.items():
                metrics = evaluate_tokenizer(tokenizer, test_text)

                model_result["test"][language] = {
                    "test_sha256": sha256_text(test_text),
                    "metrics": metrics,
                }

                token_counts[replicate][condition][language] = (
                    metrics["tokens"]
                )

                print(
                    f"{condition} -> {language}: "
                    f"{metrics['tokens']:,} tokens; "
                    f"roundtrip={metrics['roundtrip_exact']}"
                )

            results["replicates"][replicate]["models"][
                condition
            ] = model_result

    # Calculate shared-model penalty against the matching dedicated model.
    summary = {
        "experiment": EXPERIMENT_NAME,
        "vocab_size": VOCAB_SIZE,
        "replicates": {},
        "per_language": {},
    }

    for replicate in REPLICATES:
        summary["replicates"][replicate] = {}

        for language in LANGUAGES:
            dedicated_condition = f"{language}_separate"
            dedicated_tokens = token_counts[replicate][
                dedicated_condition
            ][language]
            shared_tokens = token_counts[replicate][
                "shared_multilingual"
            ][language]

            penalty = (
                (shared_tokens - dedicated_tokens)
                / dedicated_tokens
                * 100
            )

            summary["replicates"][replicate][language] = {
                "dedicated_tokens": dedicated_tokens,
                "shared_tokens": shared_tokens,
                "penalty_percent": penalty,
            }

    for language in LANGUAGES:
        penalties = [
            summary["replicates"][replicate][language][
                "penalty_percent"
            ]
            for replicate in REPLICATES
        ]

        summary["per_language"][language] = {
            "penalties_percent": penalties,
            "mean_penalty_percent": statistics.mean(penalties),
            "sample_stddev_percentage_points": (
                statistics.stdev(penalties)
                if len(penalties) > 1
                else 0.0
            ),
        }

    # Exclusive creation prevents accidental overwrites.
    with OUTPUT_PATH.open("x", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
        file.write("\n")

    with SUMMARY_PATH.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("\nRepeated experiment complete.")
    print("Full results:", OUTPUT_PATH)
    print("Summary:", SUMMARY_PATH)

    print("\nShared-vocabulary penalty summary:")
    for language, data in summary["per_language"].items():
        print(
            f"{language}: "
            f"{data['mean_penalty_percent']:.2f}% "
            f"± {data['sample_stddev_percentage_points']:.2f} "
            "(mean ± sample SD)"
        )


if __name__ == "__main__":
    main()
