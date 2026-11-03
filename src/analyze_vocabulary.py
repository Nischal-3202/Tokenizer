
from pathlib import Path

from tokenizer.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "results" / "models"


def load_tokenizer(filename):
    tokenizer = Tokenizer()
    tokenizer.load(MODEL_DIR / filename)
    return tokenizer


def learned_tokens(tokenizer):
    return {
        token_id: token_bytes
        for token_id, token_bytes in tokenizer.vocab.items()
        if token_id >= 256
    }



def main():
    hindi = load_tokenizer("hindi_separate_vocab_1024.json")
    nepali = load_tokenizer("nepali_separate_vocab_1024.json")

    hindi_tokens = set(learned_tokens(hindi).values())
    nepali_tokens = set(learned_tokens(nepali).values())

    shared = hindi_tokens & nepali_tokens

    devanagari_shared = []

    for token_bytes in shared:
        try:
            token_text = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue

        if any("\u0900" <= char <= "\u097f" for char in token_text):
            devanagari_shared.append((token_text, token_bytes))

    # Sort by Unicode code-point length, longest first.
    devanagari_shared.sort(
        key=lambda item: len(item[0]),
        reverse=True,
    )

    print("\nLONGER SHARED DEVANAGARI TOKENS")
    print("=" * 60)
    print("Shared learned tokens overall:", len(shared))
    print("Shared tokens containing Devanagari:", len(devanagari_shared))

    print("\nTop 30 by code-point length:")

    for token_text, token_bytes in devanagari_shared[:30]:
        print(
            repr(token_text),
            "| code points:", len(token_text),
            "| UTF-8 bytes:", len(token_bytes),
        )


if __name__ == "__main__":
    main()

    hindi = load_tokenizer("hindi_separate_vocab_1024.json")
    nepali = load_tokenizer("nepali_separate_vocab_1024.json")
    joint = load_tokenizer("hindi_nepali_joint_vocab_1024.json")

    hindi_tokens = set(learned_tokens(hindi).values())
    nepali_tokens = set(learned_tokens(nepali).values())
    joint_tokens = set(learned_tokens(joint).values())

    shared = hindi_tokens & nepali_tokens

    print("\nLEARNED VOCABULARY OVERLAP")
    print("=" * 50)
    print("Hindi learned tokens:", len(hindi_tokens))
    print("Nepali learned tokens:", len(nepali_tokens))
    print("Joint learned tokens:", len(joint_tokens))
    print("Shared Hindi/Nepali tokens:", len(shared))

    print(
        "Share of Hindi vocabulary also in Nepali:",
        f"{len(shared) / len(hindi_tokens) * 100:.2f}%"
    )
    print(
        "Share of Nepali vocabulary also in Hindi:",
        f"{len(shared) / len(nepali_tokens) * 100:.2f}%"
    )

    print("\nEXAMPLES OF SHARED TOKENS")
    shown = 0

    for token_bytes in sorted(shared, key=lambda item: (len(item), item)):
        token_text = token_bytes.decode("utf-8", errors="replace")

        if any("\u0900" <= char <= "\u097f" for char in token_text):
            print(repr(token_text))
            shown += 1

        if shown >= 20:
            break


if __name__ == "__main__":
    main()
