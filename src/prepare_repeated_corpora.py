
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "controlled_multilingual_replicates"
MANIFEST_PATH = (
    ROOT / "data" / "manifests"
    / "controlled_multilingual_replicates_v1.json"
)

LANGUAGES = ["english", "hindi", "nepali"]
REPLICATE_SEEDS = {
    "replicate_1": 11,
    "replicate_2": 22,
    "replicate_3": 33,
}
BYTE_BUDGET = 300_000


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def select_documents(source_text, seed, byte_budget):
    documents = [
        line.strip()
        for line in source_text.splitlines()
        if line.strip()
    ]

    # Seed-specific deterministic order, independent of source ordering.
    documents.sort(
        key=lambda doc: hashlib.sha256(
            f"{seed}:".encode("utf-8") + doc.encode("utf-8")
        ).hexdigest()
    )

    selected = []

    for document in documents:
        candidate = selected + [document]
        candidate_text = "\n".join(candidate) + "\n"
        candidate_bytes = len(candidate_text.encode("utf-8"))

        if candidate_bytes <= byte_budget:
            selected.append(document)

    output_text = "\n".join(selected) + "\n"
    return selected, output_text


def main():
    if OUTPUT_DIR.exists() or MANIFEST_PATH.exists():
        raise FileExistsError(
            "Output already exists. Refusing to overwrite it."
        )

    manifest = {
        "experiment": "repeated_corpus_sampling_v1",
        "selection_method": (
            "Sort complete source lines by SHA-256 of seed plus document; "
            "greedily select documents within the byte budget."
        ),
        "byte_budget_per_language_per_replicate": BYTE_BUDGET,
        "seeds": REPLICATE_SEEDS,
        "replicates": {},
    }

    # Validate all source files before creating output.
    source_texts = {
        language: (
            SOURCE_DIR / language / "train.txt"
        ).read_text(encoding="utf-8")
        for language in LANGUAGES
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    for replicate, seed in REPLICATE_SEEDS.items():
        replicate_dir = OUTPUT_DIR / replicate
        replicate_dir.mkdir(parents=True, exist_ok=False)

        manifest["replicates"][replicate] = {
            "seed": seed,
            "languages": {},
        }

        for language in LANGUAGES:
            selected, output_text = select_documents(
                source_texts[language],
                seed,
                BYTE_BUDGET,
            )

            output_path = replicate_dir / f"{language}_train.txt"
            output_path.write_text(output_text, encoding="utf-8")

            output_bytes = len(output_text.encode("utf-8"))

            manifest["replicates"][replicate]["languages"][language] = {
                "path": str(output_path.relative_to(ROOT)),
                "selected_documents": len(selected),
                "utf8_bytes_including_newlines": output_bytes,
                "code_points": len(output_text),
                "sha256": sha256_bytes(output_text.encode("utf-8")),
            }

            print(
                f"{replicate} | {language}: "
                f"{len(selected)} documents, "
                f"{output_bytes:,} bytes"
            )

            if output_bytes > BYTE_BUDGET:
                raise ValueError(
                    f"{replicate}/{language} exceeds the byte budget."
                )

    with MANIFEST_PATH.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("\nManifest saved to:", MANIFEST_PATH)
    print("Repeated corpus preparation complete.")


if __name__ == "__main__":
    main()
