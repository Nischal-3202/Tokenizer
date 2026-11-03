
from pathlib import Path
from collections import Counter

from tokenizer.graphemes import split_graphemes
from tokenizer.aksharas import split_aksharas


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "processed" / "nepali" / "train.txt"


def find_segmentation_differences(text, max_differences=20):
    graphemes = split_graphemes(text)
    aksharas = split_aksharas(text)

    def boundaries(units):
        positions = set()
        offset = 0

        for unit in units:
            offset += len(unit)
            positions.add(offset)

        return positions

    g_boundaries = boundaries(graphemes)
    a_boundaries = boundaries(aksharas)

    only_grapheme = sorted(g_boundaries - a_boundaries)
    only_akshara = sorted(a_boundaries - g_boundaries)

    print(
        "Boundaries unique to grapheme segmentation:",
        len(only_grapheme),
    )
    print(
        "Boundaries unique to akshara heuristic:",
        len(only_akshara),
    )

    positions = sorted(set(only_grapheme + only_akshara))
    differences = []

    for position in positions[:max_differences]:
        start = max(0, position - 15)
        end = min(len(text), position + 15)

        differences.append({
            "position": position,
            "context": text[start:end],
            "grapheme_boundary": position in g_boundaries,
            "akshara_boundary": position in a_boundaries,
        })

    return differences


def main():
    text = TRAIN_FILE.read_text(encoding="utf-8")

    graphemes = split_graphemes(text)
    aksharas = split_aksharas(text)

    print("=== SEGMENTATION COMPARISON ===")
    print("Training file:", TRAIN_FILE)
    print("Text code points:", len(text))
    print("Grapheme units:", len(graphemes))
    print("Akshara units:", len(aksharas))
    print("Same unit count:", len(graphemes) == len(aksharas))
    print("Same segmentation:", graphemes == aksharas)

    grapheme_counts = Counter(graphemes)
    akshara_counts = Counter(aksharas)

    print("\nDistinct grapheme types:", len(grapheme_counts))
    print("Distinct akshara types:", len(akshara_counts))

    print("\n=== EXAMPLE WORDS ===")

    words = [
        "नेपाल",
        "नमस्ते",
        "क्षेत्र",
        "विद्यालय",
        "प्रकाश",
        "कर्म",
        "संस्कृति",
        "दृष्टि",
    ]

    for word in words:
        g = split_graphemes(word)
        a = split_aksharas(word)

        print(f"\n{word}")
        print("  Graphemes:", g)
        print("  Aksharas: ", a)
        print("  Same:", g == a)

    print("\n=== FIRST CORPUS DIFFERENCES ===")

    differences = find_segmentation_differences(text)

    if not differences:
        print("No differing boundaries found.")
    else:
        for item in differences:
            print(f"\nPosition: {item['position']}")
            print("Context:", repr(item["context"]))
            print(
                "Grapheme boundary:",
                item["grapheme_boundary"],
            )
            print(
                "Akshara boundary:",
                item["akshara_boundary"],
            )

    print("\nDifferences shown:", len(differences))


if __name__ == "__main__":
    main()
