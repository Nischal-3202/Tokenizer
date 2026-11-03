
from collections import Counter
from pathlib import Path

from tokenizer.grapheme_bpe import GraphemeBPE
from tokenizer.akshara_bpe import AksharaBPE
from tokenizer.graphemes import split_graphemes
from tokenizer.aksharas import split_aksharas


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "nepali"
TRAIN_FILE = DATA_DIR / "train.txt"
TEST_FILE = DATA_DIR / "test.txt"

VOCAB_SIZES = [4096, 8192, 16384]


def boundary_positions(units):
    """Return code-point offsets at which segmentation units end."""
    positions = set()
    offset = 0

    for unit in units[:-1]:
        offset += len(unit)
        positions.add(offset)

    return positions


def emitted_token_strings(tokenizer, token_ids):
    """Return strings for ordinary vocabulary IDs actually emitted."""
    return [
        tokenizer.vocab[token_id]
        for token_id in token_ids
        if token_id in tokenizer.vocab
    ]


def print_token_frequency(title, tokens, token_counts, limit=10):
    """Print the most frequently emitted tokens from a given set."""
    print(f"\n{title}")

    ranked_tokens = sorted(
        tokens,
        key=lambda token: (-token_counts[token], token),
    )

    for token in ranked_tokens[:limit]:
        print(f"  {token!r}: {token_counts[token]}")


def main():
    train_text = TRAIN_FILE.read_text(encoding="utf-8")
    test_text = TEST_FILE.read_text(encoding="utf-8")

    grapheme_units = split_graphemes(test_text)
    akshara_units = split_aksharas(test_text)

    grapheme_boundaries = boundary_positions(grapheme_units)
    akshara_boundaries = boundary_positions(akshara_units)

    print("=== TEST SET SEGMENTATION ===")
    print("Code points:", len(test_text))
    print("Grapheme units:", len(grapheme_units))
    print("Akshara units:", len(akshara_units))
    print("Identical unit sequences:", grapheme_units == akshara_units)
    print(
        "Boundaries unique to graphemes:",
        len(grapheme_boundaries - akshara_boundaries),
    )
    print(
        "Boundaries unique to aksharas:",
        len(akshara_boundaries - grapheme_boundaries),
    )

    print("\n=== TOKENIZER COMPARISON ===")

    for vocab_size in VOCAB_SIZES:
        print(f"\n{'=' * 50}")
        print(f"Vocabulary size: {vocab_size}")
        print("=" * 50)

        grapheme_model = GraphemeBPE()
        akshara_model = AksharaBPE()

        grapheme_model.train(train_text, vocab_size)
        akshara_model.train(train_text, vocab_size)

        # Compare token strings, not token IDs.
        grapheme_tokens = set(grapheme_model.vocab.values())
        akshara_tokens = set(akshara_model.vocab.values())

        shared = grapheme_tokens & akshara_tokens
        only_grapheme = grapheme_tokens - akshara_tokens
        only_akshara = akshara_tokens - grapheme_tokens

        print("\n--- Vocabulary overlap ---")
        print("Grapheme vocabulary size:", len(grapheme_tokens))
        print("Akshara vocabulary size:", len(akshara_tokens))
        print("Shared token strings:", len(shared))
        print("Only in grapheme vocabulary:", len(only_grapheme))
        print("Only in akshara vocabulary:", len(only_akshara))
        print("Sample grapheme-only tokens:", sorted(only_grapheme)[:10])
        print("Sample akshara-only tokens:", sorted(only_akshara)[:10])

        # Encode the same test text with both tokenizers.
        grapheme_ids = grapheme_model.encode(test_text)
        akshara_ids = akshara_model.encode(test_text)

        print("\n--- Tokenization metrics ---")
        print("Grapheme token count:", len(grapheme_ids))
        print("Akshara token count:", len(akshara_ids))
        print("Same token count:", len(grapheme_ids) == len(akshara_ids))
        print("Identical token-ID sequences:", grapheme_ids == akshara_ids)

        # Count ordinary vocabulary tokens actually emitted.
        grapheme_emitted = emitted_token_strings(
            grapheme_model, grapheme_ids
        )
        akshara_emitted = emitted_token_strings(
            akshara_model, akshara_ids
        )

        grapheme_token_counts = Counter(grapheme_emitted)
        akshara_token_counts = Counter(akshara_emitted)

        # Count IDs that are not in the ordinary vocabulary.
        grapheme_fallback_count = sum(
            token_id not in grapheme_model.vocab
            for token_id in grapheme_ids
        )
        akshara_fallback_count = sum(
            token_id not in akshara_model.vocab
            for token_id in akshara_ids
        )

        print("\n--- Fallback diagnostics ---")
        print("Grapheme fallback IDs:", grapheme_fallback_count)
        print("Akshara fallback IDs:", akshara_fallback_count)

        print_token_frequency(
            "Most frequent grapheme-only tokens",
            only_grapheme,
            grapheme_token_counts,
        )

        print_token_frequency(
            "Most frequent akshara-only tokens",
            only_akshara,
            akshara_token_counts,
        )

        # Verify lossless encoding and decoding.
        print("\n--- Round-trip validation ---")
        print(
            "Grapheme exact round-trip:",
            grapheme_model.decode(grapheme_ids) == test_text,
        )
        print(
            "Akshara exact round-trip:",
            akshara_model.decode(akshara_ids) == test_text,
        )


if __name__ == "__main__":
    main()
