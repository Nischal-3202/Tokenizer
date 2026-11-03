
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "controlled_multilingual"
MANIFEST_DIR = ROOT / "data" / "manifests"

LANGUAGES = ["english", "hindi", "nepali"]
BYTE_BUDGET = 900_000


def read_documents(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_documents(documents, byte_budget):
    selected = []
    total_bytes = 0

    for document in documents:
        document_bytes = len(document.encode("utf-8"))

        if total_bytes + document_bytes > byte_budget:
            continue

        selected.append(document)
        total_bytes += document_bytes

        if total_bytes == byte_budget:
            break

    return selected, total_bytes


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "controlled_multilingual_corpus_v1",
        "byte_budget_per_language": BYTE_BUDGET,
        "selection_method": (
            "Deterministic source-order selection of complete "
            "documents without exceeding the UTF-8 byte budget."
        ),
        "languages": {},
    }

    for language in LANGUAGES:
        source = DATA_DIR / language / "train.txt"
        documents = read_documents(source)

        selected, total_bytes = select_documents(
            documents, BYTE_BUDGET
        )

        output_path = OUTPUT_DIR / f"{language}_train.txt"
        output_path.write_text(
            "\n".join(selected) + "\n",
            encoding="utf-8",
        )

        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()

        manifest["languages"][language] = {
            "source_documents": len(documents),
            "selected_documents": len(selected),
            "selected_utf8_bytes": total_bytes,
            "selected_code_points": sum(map(len, selected)),
            "sha256": digest,
            "output_file": str(output_path.relative_to(ROOT)),
        }

        print(f"\n{language.upper()}")
        print("Selected documents:", len(selected))
        print("Selected UTF-8 bytes:", total_bytes)
        print("Selected code points:", sum(map(len, selected)))
        print("SHA-256:", digest)

    manifest_path = (
        MANIFEST_DIR / "controlled_multilingual_corpus_v1.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nManifest saved to:", manifest_path)


if __name__ == "__main__":
    main()
