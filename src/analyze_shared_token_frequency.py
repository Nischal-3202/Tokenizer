
from collections import Counter
from pathlib import Path

from tokenizer.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "results" / "models"
DATA_DIR = ROOT / "data" / "processed"


def load_tokenizer(filename):
    tokenizer = Tokenizer()
    tokenizer.load(MODEL_DIR / filename)
    return tokenizer


def read_text(language):
    path = DATA_DIR / language / "train.txt"
    return path.read_text(encoding="utf-8")


def count_token_bytes(tokenizer, text):
    token_ids = tokenizer.encode(text)
    counts = Counter(token_ids)

    # Convert token IDs to the byte sequences they represent.
    return Counter({
        tokenizer.vocab[token_id]: count
        for token_id, count in counts.items()
    })


def main():
    hindi = load_tokenizer("hindi_separate_vocab_1024.json")
    nepali = load_tokenizer("nepali_separate_vocab_1024.json")

    hindi_text = read_text("hindi")
    nepali_text = read_text("nepali")

    hindi_counts = count_token_bytes(hindi, hindi_text)
    nepali_counts = count_token_bytes(nepali, nepali_text)

    hindi_learned = {
        token_bytes
        for token_id, token_bytes in hindi.vocab.items()
        if token_id >= 256
    }
    nepali_learned = {
        token_bytes
        for token_id, token_bytes in nepali.vocab.items()
        if token_id >= 256
    }

    shared = hindi_learned & nepali_learned

    rows = []

    for token_bytes in shared:
        try:
            token = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue

        if not any("\u0900" <= char <= "\u097f" for char in token):
            continue

        rows.append({
            "token": token,
            "hindi_frequency": hindi_counts[token_bytes],
            "nepali_frequency": nepali_counts[token_bytes],
        })

    rows.sort(
        key=lambda row: min(
            row["hindi_frequency"],
            row["nepali_frequency"],
        ),
        reverse=True,
    )

    print("\nSHARED DEVANAGARI TOKEN FREQUENCIES")
    print("=" * 80)
    print(
        f"{'Token':<18}"
        f"{'Hindi frequency':>20}"
        f"{'Nepali frequency':>22}"
    )
    print("-" * 80)

    for row in rows[:30]:
        print(
            f"{row['token']!r:<18}"
            f"{row['hindi_frequency']:>20,}"
            f"{row['nepali_frequency']:>22,}"
        )


if __name__ == "__main__":
    main()
