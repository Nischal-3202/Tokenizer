
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LANGUAGES = ["english", "hindi", "nepali"]


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    for language in LANGUAGES:
        folder = ROOT / "data" / "processed" / language

        train_path = folder / "train.txt"
        test_path = folder / "test.txt"

        train = read_lines(train_path)
        test = read_lines(test_path)

        train_hashes = {fingerprint(text) for text in train}
        test_hashes = {fingerprint(text) for text in test}

        overlap = train_hashes & test_hashes

        print(f"\n{language.upper()}")
        print("Train lines:", len(train))
        print("Test lines:", len(test))
        print("Exact overlap:", len(overlap))
        print(
            "Train UTF-8 bytes:",
            sum(len(text.encode("utf-8")) for text in train),
        )
        print(
            "Test UTF-8 bytes:",
            sum(len(text.encode("utf-8")) for text in test),
        )

        assert train, f"{language}: training split is empty"
        assert test, f"{language}: test split is empty"
        assert not overlap, f"{language}: train/test overlap detected"

    print("\nAll corpus validation checks passed.")


if __name__ == "__main__":
    main()
