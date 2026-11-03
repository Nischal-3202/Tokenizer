
from datasets import load_dataset

DATASET_ID = "ashtok897/indic-hplt-v2"

language_names = {
    "en": "English",
    "hi": "Hindi",
    "ne": "Nepali",
}

dataset = load_dataset(
    DATASET_ID,
    split="train",
    streaming=True,
)

samples = {}

print("Searching the stream for all three languages...")

for row in dataset:
    lang = row.get("lang")

    if lang in language_names and lang not in samples:
        samples[lang] = row

        print(f"\nFound {language_names[lang]} ({lang})")
        print("Document ID:", row.get("doc_id"))
        print("Characters:", row.get("char_count"))
        print("Text preview:", row.get("text", "")[:250])

    if len(samples) == len(language_names):
        break

print("\n--- Summary ---")

for lang, name in language_names.items():
    print(f"{name}: {'FOUND' if lang in samples else 'NOT FOUND'}")

if len(samples) != len(language_names):
    print("Not all languages were found in the scanned stream.")
