
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

VOCAB_SIZES = [512, 1024, 2048]
LANGUAGES = ["english", "hindi", "nepali"]


def load_results(vocab_size):
    path = RESULTS_DIR / f"baseline_results_vocab_{vocab_size}.json"

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    results = {
        size: load_results(size)
        for size in VOCAB_SIZES
    }

    print("\nHELD-OUT TEST RESULTS")
    print("=" * 100)

    header = (
        f"{'Language':<10}"
        f"{'Vocab':>8}"
        f"{'Tokens':>12}"
        f"{'Bytes/token':>15}"
        f"{'Tokens/1000 chars':>21}"
        f"{'Roundtrip':>12}"
    )
    print(header)
    print("-" * 100)

    for language in LANGUAGES:
        for size in VOCAB_SIZES:
            entry = results[size][
                {"english": "en", "hindi": "hi", "nepali": "ne"}[language]
            ]
            metrics = entry["test"]

            print(
                f"{language:<10}"
                f"{size:>8}"
                f"{metrics['tokens']:>12,}"
                f"{metrics['bytes_per_token']:>15.3f}"
                f"{metrics['tokens_per_1000_chars']:>21.2f}"
                f"{str(metrics['roundtrip_exact']):>12}"
            )

        print()

    print("\nTOKEN COUNT REDUCTION FROM VOCABULARY 512 TO 2048")
    print("=" * 65)

    for language in LANGUAGES:
        code = {"english": "en", "hindi": "hi", "nepali": "ne"}[language]

        tokens_512 = results[512][code]["test"]["tokens"]
        tokens_2048 = results[2048][code]["test"]["tokens"]

        reduction = (tokens_512 - tokens_2048) / tokens_512 * 100

        print(
            f"{language.capitalize():<10}"
            f"{tokens_512:>12,} -> {tokens_2048:<12,}"
            f"Reduction: {reduction:.2f}%"
        )


if __name__ == "__main__":
    main()
