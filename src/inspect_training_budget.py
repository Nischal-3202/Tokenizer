
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

LANGUAGES = ["english", "hindi", "nepali"]


def main():
    print("\nTRAINING CORPUS BUDGET")
    print("=" * 75)

    for language in LANGUAGES:
        path = DATA_DIR / language / "train.txt"
        text = path.read_text(encoding="utf-8")

        print(f"\n{language.upper()}")
        print("Documents:", sum(1 for line in text.splitlines() if line.strip()))
        print("Unicode code points:", len(text))
        print("UTF-8 bytes:", len(text.encode("utf-8")))


if __name__ == "__main__":
    main()
