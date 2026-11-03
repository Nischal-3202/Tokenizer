
import hashlib
import json
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

OUTPUT_PATH = RESULTS_DIR / "corpus_audit_v1.json"

LANGUAGES = ["english", "hindi", "nepali"]
REPLICATES = ["replicate_1", "replicate_2", "replicate_3"]

CORPUS_FILES = [
    DATA_DIR / "processed" / language / f"{split}.txt"
    for language in LANGUAGES
    for split in ["train", "test"]
]

CONTROLLED_DIRS = [
    DATA_DIR / "controlled_multilingual",
    DATA_DIR / "controlled_multilingual_balanced",
    DATA_DIR / "controlled_multilingual_replicates",
]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path):
    return path.read_bytes()


def read_text(path):
    return read_bytes(path).decode("utf-8", errors="strict")


def documents(text):
    # Matches the line-based document interpretation used in sampling.
    return [line.strip() for line in text.splitlines() if line.strip()]


def document_stats(text):
    lines = text.splitlines()
    docs = documents(text)
    counts = Counter(unicodedata.category(ch) for ch in text)

    return {
        "code_points": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "line_count_including_empty": len(lines),
        "nonempty_documents": len(docs),
        "unique_documents": len(set(docs)),
        "duplicate_document_excess": len(docs) - len(set(docs)),
        "empty_lines": sum(not line.strip() for line in lines),
        "replacement_character_count": text.count("\uFFFD"),
        "nul_character_count": text.count("\x00"),
        "control_character_count": sum(
            count for category, count in counts.items()
            if category == "Cc" and "\n" not in text
        ),
        "unicode_categories": dict(counts),
        "sha256": sha256_bytes(text.encode("utf-8")),
    }


def load_corpus_file(path):
    if not path.exists():
        return None

    text = read_text(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "stats": document_stats(text),
        "documents": documents(text),
        "text": text,
    }


def overlap_stats(left_docs, right_docs):
    left = set(left_docs)
    right = set(right_docs)
    intersection = left & right
    union = left | right

    return {
        "left_unique_documents": len(left),
        "right_unique_documents": len(right),
        "shared_unique_documents": len(intersection),
        "left_fraction_shared": (
            len(intersection) / len(left) if left else 0.0
        ),
        "right_fraction_shared": (
            len(intersection) / len(right) if right else 0.0
        ),
        "jaccard_similarity": (
            len(intersection) / len(union) if union else 0.0
        ),
        "shared_examples": sorted(intersection)[:5],
    }


def walk_manifest_records(obj, parent_key=""):
    """Find path/hash/byte-count records in different manifest schemas."""
    if isinstance(obj, dict):
        path_value = obj.get("path") or obj.get("output_path")
        hash_value = obj.get("sha256") or obj.get("output_sha256")

        if path_value and hash_value:
            yield {
                "path": path_value,
                "sha256": hash_value,
                "expected_bytes": (
                    obj.get("utf8_bytes_including_newlines")
                    or obj.get("output_bytes_including_newlines")
                    or obj.get("utf8_bytes")
                    or obj.get("bytes")
                ),
                "context": parent_key,
            }

        for key, value in obj.items():
            yield from walk_manifest_records(value, str(key))

    elif isinstance(obj, list):
        for item in obj:
            yield from walk_manifest_records(item, parent_key)


def audit_manifests():
    manifest_dir = DATA_DIR / "manifests"
    reports = []

    if not manifest_dir.exists():
        return reports

    for path in sorted(manifest_dir.glob("*.json")):
        try:
            manifest = json.loads(read_text(path))
        except Exception as exc:
            reports.append({
                "manifest": str(path.relative_to(ROOT)),
                "error": str(exc),
            })
            continue

        checks = []

        for record in walk_manifest_records(manifest):
            candidate = Path(record["path"])
            if not candidate.is_absolute():
                candidate = ROOT / candidate

            item = {
                "path": record["path"],
                "expected_sha256": record["sha256"],
                "context": record["context"],
            }

            if not candidate.exists():
                item["status"] = "missing_file"
            else:
                raw = read_bytes(candidate)
                actual_hash = sha256_bytes(raw)
                item["actual_sha256"] = actual_hash
                item["hash_matches"] = actual_hash == record["sha256"]

                if record["expected_bytes"] is not None:
                    item["expected_bytes"] = record["expected_bytes"]
                    item["actual_bytes"] = len(raw)
                    item["byte_count_matches"] = (
                        len(raw) == record["expected_bytes"]
                    )

                item["status"] = (
                    "ok" if item["hash_matches"] else "hash_mismatch"
                )

            checks.append(item)

        reports.append({
            "manifest": str(path.relative_to(ROOT)),
            "record_count": len(checks),
            "checks": checks,
        })

    return reports


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite {OUTPUT_PATH}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    audit = {
        "audit": "corpus_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "document_definition": (
            "Each nonempty stripped line is treated as one document."
        ),
        "corpora": {},
        "duplicate_checks": {},
        "replicate_overlap": {},
        "manifest_checks": [],
        "limitations": [
            "Exact duplicate checks do not detect paraphrases.",
            "Line-based document boundaries may not match source documents.",
            "Hash ordering is deterministic but does not guarantee "
            "statistically independent samples.",
            "Matching hashes establish file integrity, not dataset provenance.",
            "The audit cannot prove that the source test set is free of "
            "upstream training-data contamination."
        ],
    }

    loaded = {}

    # Audit original train/test files.
    for path in CORPUS_FILES:
        if not path.exists():
            audit["corpora"][str(path.relative_to(ROOT))] = {
                "status": "missing_file"
            }
            continue

        try:
            item = load_corpus_file(path)
            loaded[str(path.relative_to(ROOT))] = item

            audit["corpora"][item["path"]] = {
                "status": "ok",
                **item["stats"],
            }
        except UnicodeDecodeError as exc:
            audit["corpora"][str(path.relative_to(ROOT))] = {
                "status": "invalid_utf8",
                "error": str(exc),
            }

    # Audit controlled corpora.
    controlled_docs = {}

    for directory in CONTROLLED_DIRS:
        if not directory.exists():
            continue

        for path in sorted(directory.rglob("*.txt")):
            try:
                item = load_corpus_file(path)
                key = str(path.relative_to(ROOT))
                loaded[key] = item
                controlled_docs[key] = item["documents"]

                audit["corpora"][key] = {
                    "status": "ok",
                    **item["stats"],
                }
            except UnicodeDecodeError as exc:
                audit["corpora"][str(path.relative_to(ROOT))] = {
                    "status": "invalid_utf8",
                    "error": str(exc),
                }

    # Within-file duplicate documents.
    audit["duplicate_checks"]["within_file"] = {
        path: {
            "document_count": len(item["documents"]),
            "unique_count": len(set(item["documents"])),
            "duplicate_excess": (
                len(item["documents"]) - len(set(item["documents"]))
            ),
        }
        for path, item in loaded.items()
    }

    # Exact train/test overlap, both within and across languages.
    train_docs = {
        language: loaded.get(
            f"data/processed/{language}/train.txt", {}
        ).get("documents", [])
        for language in LANGUAGES
    }
    test_docs = {
        language: loaded.get(
            f"data/processed/{language}/test.txt", {}
        ).get("documents", [])
        for language in LANGUAGES
    }

    leakage = {}
    for train_language in LANGUAGES:
        for test_language in LANGUAGES:
            leakage[
                f"train_{train_language}__test_{test_language}"
            ] = overlap_stats(
                train_docs[train_language],
                test_docs[test_language],
            )

    audit["duplicate_checks"]["train_test_exact_overlap"] = leakage

    # Compare the three replicate samples for each language.
    for language in LANGUAGES:
        pairs = {}

        for left, right in combinations(REPLICATES, 2):
            left_path = (
                f"data/controlled_multilingual_replicates/"
                f"{left}/{language}_train.txt"
            )
            right_path = (
                f"data/controlled_multilingual_replicates/"
                f"{right}/{language}_train.txt"
            )

            left_docs = loaded.get(left_path, {}).get("documents", [])
            right_docs = loaded.get(right_path, {}).get("documents", [])

            pairs[f"{left}__{right}"] = overlap_stats(
                left_docs, right_docs
            )

        audit["replicate_overlap"][language] = pairs

    # Compare each replicate with its original-language training corpus.
    audit["duplicate_checks"]["replicate_vs_source_train"] = {}

    for replicate in REPLICATES:
        for language in LANGUAGES:
            replicate_path = (
                f"data/controlled_multilingual_replicates/"
                f"{replicate}/{language}_train.txt"
            )
            source_path = f"data/processed/{language}/train.txt"

            audit["duplicate_checks"]["replicate_vs_source_train"][
                f"{replicate}_{language}"
            ] = overlap_stats(
                loaded.get(source_path, {}).get("documents", []),
                loaded.get(replicate_path, {}).get("documents", []),
            )

    audit["manifest_checks"] = audit_manifests()

    # Simple automated warnings; these do not replace manual review.
    warnings = []

    for path, item in audit["corpora"].items():
        if item.get("status") != "ok":
            warnings.append(f"Corpus file needs attention: {path}")
            continue

        if item.get("replacement_character_count", 0):
            warnings.append(f"Replacement characters found in {path}")

        if item.get("nul_character_count", 0):
            warnings.append(f"NUL characters found in {path}")

        if item.get("duplicate_document_excess", 0):
            warnings.append(f"Duplicate lines found in {path}")

    for manifest in audit["manifest_checks"]:
        for check in manifest.get("checks", []):
            if check.get("status") != "ok":
                warnings.append(
                    f"Manifest mismatch: {manifest['manifest']} "
                    f"-> {check['path']}"
                )
            if check.get("byte_count_matches") is False:
                warnings.append(
                    f"Manifest byte-count mismatch: {check['path']}"
                )

    audit["warnings"] = sorted(set(warnings))

    with OUTPUT_PATH.open("x", encoding="utf-8") as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("\nCorpus audit complete.")
    print("Corpus files checked:", len(audit["corpora"]))
    print("Manifest files checked:", len(audit["manifest_checks"]))
    print("Warnings:", len(audit["warnings"]))

    for warning in audit["warnings"]:
        print("WARNING:", warning)

    print("\nExact train/test overlaps:")
    for pair, result in leakage.items():
        if result["shared_unique_documents"]:
            print(
                pair,
                "shared documents:",
                result["shared_unique_documents"],
            )

    print("\nReplicate overlap:")
    for language, pairs in audit["replicate_overlap"].items():
        for pair, result in pairs.items():
            print(
                language,
                pair,
                f"Jaccard={result['jaccard_similarity']:.4f}",
                f"shared={result['shared_unique_documents']}",
            )

    print("\nAudit report saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
