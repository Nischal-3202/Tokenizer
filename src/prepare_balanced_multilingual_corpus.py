
import hashlib
import json
from pathlib import Path


LANGUAGES = ["english", "hindi", "nepali"]
BYTE_BUDGET = 300_000

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/controlled_multilingual_balanced")
MANIFEST_PATH = Path(
    "data/manifests/controlled_multilingual_balanced_v1.json"
)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_documents(text, byte_budget):
    """Select complete documents deterministically within a byte budget."""
    documents = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Stable hash ordering avoids depending on the original document order.
    documents.sort(key=sha256_text)

    selected = []
    selected_bytes = 0

    for document in documents:
        document_bytes = len(document.encode("utf-8"))

        if selected_bytes + document_bytes <= byte_budget:
            selected.append(document)
            selected_bytes += document_bytes

    return selected, selected_bytes, len(documents)


def main():
    if OUTPUT_DIR.exists() or MANIFEST_PATH.exists():
        raise FileExistsError(
            "Output already exists. Refusing to overwrite existing data."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "balanced_multilingual_corpus_v1",
        "selection_method": (
            "SHA-256 hash ordering; retain complete documents "
            "while staying within the byte budget"
        ),
        "byte_budget_per_language": BYTE_BUDGET,
        "languages": {},
    }

    for language in LANGUAGES:
        source_path = DATA_DIR / language / "train.txt"

        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing training corpus: {source_path}"
            )

        source_text = source_path.read_text(encoding="utf-8")

        selected, selected_bytes, available_documents = (
            select_documents(source_text, BYTE_BUDGET)
        )

        output_text = "\n".join(selected) + "\n"
        output_path = OUTPUT_DIR / f"{language}_train.txt"
        output_path.write_text(output_text, encoding="utf-8")

        output_bytes = len(output_text.encode("utf-8"))

        manifest["languages"][language] = {
            "source_path": str(source_path),
            "output_path": str(output_path),
            "available_documents": available_documents,
            "selected_documents": len(selected),
            "selected_document_bytes": selected_bytes,
            "output_bytes_including_newlines": output_bytes,
            "output_sha256": hashlib.sha256(
                output_text.encode("utf-8")
            ).hexdigest(),
        }

        print(
            f"{language}: "
            f"{len(selected)} documents, "
            f"{output_bytes:,} output bytes"
        )

        if output_bytes > BYTE_BUDGET:
            raise ValueError(
                f"{language} exceeds the byte budget: {output_bytes}"
            )

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nManifest saved to: {MANIFEST_PATH}")
    print("Balanced multilingual corpus preparation complete.")


if __name__ == "__main__":
    main()
