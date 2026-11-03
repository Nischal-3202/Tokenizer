
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from tokenizer.tokenizer import Tokenizer
from tokenizer.grapheme_bpe import GraphemeBPE
from tokenizer.akshara_bpe import AksharaBPE
from tokenizer.metrics import evaluate_tokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# Start small. Expand after the smoke test passes.
VOCAB_SIZES = [4096]
LANGUAGES = ["english", "hindi", "nepali"]

# ByteBPE and GraphemeBPE apply to all three languages.
# AksharaBPE is evaluated only on Hindi and Nepali.
TOKENIZERS = {
    "byte_bpe": Tokenizer,
    "grapheme_bpe": GraphemeBPE,
    "akshara_bpe": AksharaBPE,
}

EXPERIMENT_NAME = "multilingual_tokenizer_baseline_v1"
OUTPUT_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}.json"


def read_text(path):
    return path.read_text(encoding="utf-8")


def get_fallback_count(tokenizer, token_ids):
    """Count byte-fallback IDs for tokenizers that use this convention."""
    fallback_base = getattr(tokenizer, "BYTE_FALLBACK_BASE", None)

    # The current grapheme and akshara implementations define this
    # constant at module level rather than on the class.
    if fallback_base is None:
        if isinstance(tokenizer, (GraphemeBPE, AksharaBPE)):
            fallback_base = 1_000_000
        else:
            return None

    return sum(token_id >= fallback_base for token_id in token_ids)


def evaluate(tokenizer, text):
    metrics = evaluate_tokenizer(tokenizer, text)
    token_ids = tokenizer.encode(text)

    metrics["fallback_id_count"] = get_fallback_count(
        tokenizer, token_ids
    )
    return metrics


def train_tokenizer(tokenizer_class, train_text, vocab_size):
    tokenizer = tokenizer_class()
    tokenizer.train(train_text, vocab_size=vocab_size)
    return tokenizer


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {OUTPUT_PATH}\n"
            "Choose a new EXPERIMENT_NAME before running again."
        )

    results = {
        "experiment": EXPERIMENT_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "languages": LANGUAGES,
        "vocabulary_budgets": VOCAB_SIZES,
        "metrics": [
            "characters",
            "utf8_bytes",
            "tokens",
            "bytes_per_token",
            "tokens_per_1000_chars",
            "fallback_id_count",
            "roundtrip_exact",
        ],
        "runs": [],
    }

    for language in LANGUAGES:
        print(f"\n{'=' * 65}")
        print(f"LANGUAGE: {language.upper()}")
        print("=" * 65)

        language_dir = DATA_DIR / language
        train_text = read_text(language_dir / "train.txt")
        test_text = read_text(language_dir / "test.txt")

        print("Training code points:", len(train_text))
        print("Training UTF-8 bytes:", len(train_text.encode("utf-8")))
        print("Test code points:", len(test_text))
        print("Test UTF-8 bytes:", len(test_text.encode("utf-8")))

        for vocab_size in VOCAB_SIZES:
            for tokenizer_name, tokenizer_class in TOKENIZERS.items():

                if tokenizer_name == "akshara_bpe" and language == "english":
                    continue

                print(
                    f"\nTraining {tokenizer_name} | "
                    f"{language} | target vocab={vocab_size}"
                )

                tokenizer = train_tokenizer(
                    tokenizer_class, train_text, vocab_size
                )

                train_metrics = evaluate(tokenizer, train_text)
                test_metrics = evaluate(tokenizer, test_text)

                run = {
                    "language": language,
                    "tokenizer": tokenizer_name,
                    "requested_vocab_size": vocab_size,
                    "actual_vocab_size": tokenizer.vocab_size,
                    "merge_count": len(tokenizer.merges),
                    "training_code_points": len(train_text),
                    "training_utf8_bytes": len(train_text.encode("utf-8")),
                    "test": test_metrics,
                    "train": train_metrics,
                }

                results["runs"].append(run)

                print("Actual vocabulary:", tokenizer.vocab_size)
                print("Merge count:", len(tokenizer.merges))
                print("Test metrics:", test_metrics)

    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("Total runs:", len(results["runs"]))


if __name__ == "__main__":
    main()
