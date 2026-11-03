
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

for language in ["hindi", "nepali"]:
    path = DATA_DIR / language / "train.txt"
    documents = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    sizes = [len(doc.encode("utf-8")) for doc in documents]

    print(f"\n{language.upper()}")
    print("Documents:", len(documents))
    print("Smallest document (bytes):", min(sizes))
    print("Largest document (bytes):", max(sizes))
    print("Documents over 900,000 bytes:", sum(size > 900_000 for size in sizes))
