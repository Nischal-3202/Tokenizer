
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "ashtok897/indic-hplt-v2"

LANGUAGES = {
    "en": "english",
    "hi": "hindi",
    "ne": "nepali",
}

MAX_CHARS_PER_LANGUAGE = 1_000_000
TEST_FRACTION = 0.10
RANDOM_SEED = 42
MIN_CHARS = 100
MAX_CHARS_PER_DOCUMENT = 20_000


def clean_text(text):
    """Apply conservative, language-independent whitespace cleanup."""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stable_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    manifest_dir = ROOT / "data" / "manifests"

    for folder in (raw_dir, processed_dir, manifest_dir):
        folder.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_ID} in streaming mode...")

    stream = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
    )

    records = {lang: [] for lang in LANGUAGES}
    seen_hashes = {lang: set() for lang in LANGUAGES}
    stats = {
        lang: Counter()
        for lang in LANGUAGES
    }

    for row in stream:
        lang = row.get("lang")

        if lang not in LANGUAGES:
            continue

        if stats[lang]["characters_kept"] >= MAX_CHARS_PER_LANGUAGE:
            if all(
                stats[code]["characters_kept"] >= MAX_CHARS_PER_LANGUAGE
                for code in LANGUAGES
            ):
                break
            continue

        text = row.get("text", "")
        stats[lang]["records_seen"] += 1

        if not isinstance(text, str):
            stats[lang]["invalid_text"] += 1
            continue

        text = clean_text(text)

        if not MIN_CHARS <= len(text) <= MAX_CHARS_PER_DOCUMENT:
            stats[lang]["length_rejected"] += 1
            continue

        digest = stable_hash(text)

        if digest in seen_hashes[lang]:
            stats[lang]["duplicates"] += 1
            continue

        seen_hashes[lang].add(digest)

        record = {
            "text": text,
            "doc_id": row.get("doc_id"),
            "url": row.get("url"),
            "sha256": digest,
            "char_count": len(text),
            "utf8_bytes": len(text.encode("utf-8")),
        }

        records[lang].append(record)
        stats[lang]["documents_kept"] += 1
        stats[lang]["characters_kept"] += len(text)
        stats[lang]["utf8_bytes_kept"] += len(
            text.encode("utf-8")
        )

        if stats[lang]["records_seen"] % 1000 == 0:
            print(
                LANGUAGES[lang],
                "records scanned:",
                stats[lang]["records_seen"],
                "characters kept:",
                stats[lang]["characters_kept"],
            )

    manifest = {
        "dataset_id": DATASET_ID,
        "split": "train",
        "collection_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "random_seed": RANDOM_SEED,
        "test_fraction": TEST_FRACTION,
        "max_characters_per_language": MAX_CHARS_PER_LANGUAGE,
        "cleaning": (
            "Replace NUL characters, collapse whitespace, strip ends; "
            "reject texts outside configured length bounds; "
            "deduplicate exact cleaned text within each language."
        ),
        "languages": {},
    }

    rng = random.Random(RANDOM_SEED)

    for lang, language_name in LANGUAGES.items():
        language_records = records[lang]
        rng.shuffle(language_records)

        split_index = int(
            len(language_records) * (1 - TEST_FRACTION)
        )

        train_records = language_records[:split_index]
        test_records = language_records[split_index:]

        language_raw_dir = raw_dir / language_name
        language_processed_dir = processed_dir / language_name

        language_raw_dir.mkdir(parents=True, exist_ok=True)
        language_processed_dir.mkdir(parents=True, exist_ok=True)

        # Keep the original record text and provenance in JSONL.
        with open(
            language_raw_dir / "sample.jsonl",
            "w",
            encoding="utf-8",
        ) as f:
            for record in language_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        for filename, split_records in (
            ("train.txt", train_records),
            ("test.txt", test_records),
        ):
            with open(
                language_processed_dir / filename,
                "w",
                encoding="utf-8",
            ) as f:
                for record in split_records:
                    f.write(record["text"] + "\n")

        manifest["languages"][lang] = {
            "directory": language_name,
            "documents": len(language_records),
            "train_documents": len(train_records),
            "test_documents": len(test_records),
            "characters": sum(
                r["char_count"] for r in language_records
            ),
            "utf8_bytes": sum(
                r["utf8_bytes"] for r in language_records
            ),
            "collection_stats": dict(stats[lang]),
        }

        print(f"\n{language_name.upper()}")
        print("Documents:", len(language_records))
        print("Train:", len(train_records))
        print("Test:", len(test_records))
        print(
            "Characters:",
            manifest["languages"][lang]["characters"],
        )
        print(
            "UTF-8 bytes:",
            manifest["languages"][lang]["utf8_bytes"],
        )

    manifest_path = manifest_dir / "pilot_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nManifest saved to:", manifest_path)


if __name__ == "__main__":
    main()
