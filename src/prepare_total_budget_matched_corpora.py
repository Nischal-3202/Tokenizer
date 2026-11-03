"""Create deterministic 100 KB-per-language subsets of saved 300 KB resamples."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "controlled_multilingual_replicates"
OUTPUT_DIR = ROOT / "data" / "total_budget_matched_replicates"
MANIFEST_PATH = ROOT / "data" / "manifests" / "total_budget_matched_replicates_v1.json"
LANGUAGES = ("english", "hindi", "nepali")
REPLICATES = ("replicate_1", "replicate_2", "replicate_3")
SEEDS = {"replicate_1": 11, "replicate_2": 22, "replicate_3": 33}
BYTE_BUDGET = 100_000


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def select_documents(source_text, selection_seed):
    documents = [line.strip() for line in source_text.splitlines() if line.strip()]
    documents.sort(key=lambda document: hashlib.sha256(f"{selection_seed}:".encode("utf-8") + document.encode("utf-8")).hexdigest())
    selected = []
    for document in documents:
        candidate = "\n".join([*selected, document]) + "\n"
        if len(candidate.encode("utf-8")) <= BYTE_BUDGET:
            selected.append(document)
    return selected, "\n".join(selected) + "\n"


def main():
    if OUTPUT_DIR.exists() or MANIFEST_PATH.exists():
        raise FileExistsError("Refusing to overwrite existing matched-budget data.")
    source_texts = {replicate: {language: (SOURCE_DIR / replicate / f"{language}_train.txt").read_text(encoding="utf-8") for language in LANGUAGES} for replicate in REPLICATES}
    manifest = {"experiment": "total_budget_matched_corpus_sampling_v1", "source_manifest": "data/manifests/controlled_multilingual_replicates_v1.json", "selection_method": "Within each saved 300 KB replicate/language partition, sort complete lines by SHA-256 of the documented replicate seed, experiment label, language, and line; greedily select within 100,000 UTF-8 bytes.", "byte_budget_per_language": BYTE_BUDGET, "replicates": {}}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    for replicate in REPLICATES:
        replicate_dir = OUTPUT_DIR / replicate
        replicate_dir.mkdir(exist_ok=False)
        manifest["replicates"][replicate] = {"source_seed": SEEDS[replicate], "languages": {}}
        for language in LANGUAGES:
            source_text = source_texts[replicate][language]
            selection_seed = f"total_budget_matched_v1:{SEEDS[replicate]}:{language}"
            selected, output_text = select_documents(source_text, selection_seed)
            output_path = replicate_dir / f"{language}_train.txt"
            output_path.write_text(output_text, encoding="utf-8")
            output_bytes = output_text.encode("utf-8")
            manifest["replicates"][replicate]["languages"][language] = {"source_path": str((SOURCE_DIR / replicate / f"{language}_train.txt").relative_to(ROOT)), "source_sha256": sha256_bytes(source_text.encode("utf-8")), "selection_seed": selection_seed, "selected_documents": len(selected), "utf8_bytes_including_newlines": len(output_bytes), "code_points": len(output_text), "sha256": sha256_bytes(output_bytes), "path": str(output_path.relative_to(ROOT))}
            print(f"{replicate} | {language}: {len(selected)} documents, {len(output_bytes):,} bytes")
    with MANIFEST_PATH.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
