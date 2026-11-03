"""Train matched 100 KB dedicated and joint byte-BPE tokenizers for three resamples."""
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from tokenizer.metrics import evaluate_tokenizer
from tokenizer.tokenizer import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
CORPUS_DIR = ROOT / "data" / "total_budget_matched_replicates"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = RESULTS_DIR / "models"
LANGUAGES = ("english", "hindi", "nepali")
REPLICATES = ("replicate_1", "replicate_2", "replicate_3")
VOCAB_SIZE = 1024
EXPERIMENT_NAME = "total_budget_matched_v1"
OUTPUT_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}.json"
SUMMARY_PATH = RESULTS_DIR / f"{EXPERIMENT_NAME}_summary.json"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path.read_text(encoding="utf-8")


def train_model(training_text):
    tokenizer = Tokenizer()
    tokenizer.train(training_text, vocab_size=VOCAB_SIZE)
    return tokenizer


def main():
    if OUTPUT_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment result.")
    planned_paths = [MODEL_DIR / f"{EXPERIMENT_NAME}_{replicate}_{condition}_vocab_{VOCAB_SIZE}.json" for replicate in REPLICATES for condition in (*[f"{language}_separate" for language in LANGUAGES], "shared_multilingual")]
    existing = [path for path in planned_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite model files: " + ", ".join(map(str, existing)))
    test_data = {language: read_text(DATA_DIR / language / "test.txt") for language in LANGUAGES}
    results = {"experiment": EXPERIMENT_NAME, "created_at_utc": datetime.now(timezone.utc).isoformat(), "python_version": sys.version, "platform": platform.platform(), "algorithm": "byte_level_bpe", "requested_vocab_size": VOCAB_SIZE, "selection_manifest": "data/manifests/total_budget_matched_replicates_v1.json", "design": "Each dedicated and joint source language sample is matched within replicate at approximately 100 KB; the joint tokenizer totals approximately 300 KB.", "test_data": {language: {"utf8_bytes": len(text.encode("utf-8")), "code_points": len(text), "sha256": sha256_text(text)} for language, text in test_data.items()}, "replicates": {}}
    penalties = {language: [] for language in LANGUAGES}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for replicate in REPLICATES:
        training_data = {language: read_text(CORPUS_DIR / replicate / f"{language}_train.txt") for language in LANGUAGES}
        conditions = {**{f"{language}_separate": training_data[language] for language in LANGUAGES}, "shared_multilingual": "\n".join(training_data.values())}
        replicate_result = {"training_data": {language: {"utf8_bytes": len(text.encode("utf-8")), "code_points": len(text), "sha256": sha256_text(text)} for language, text in training_data.items()}, "models": {}}
        for condition, training_text in conditions.items():
            tokenizer = train_model(training_text)
            model_path = MODEL_DIR / f"{EXPERIMENT_NAME}_{replicate}_{condition}_vocab_{VOCAB_SIZE}.json"
            tokenizer.save(model_path)
            model_result = {"actual_vocab_size": tokenizer.vocab_size, "merge_count": len(tokenizer.merges), "training_utf8_bytes": len(training_text.encode("utf-8")), "training_sha256": sha256_text(training_text), "model_file": str(model_path.relative_to(ROOT)), "test": {}}
            for language, test_text in test_data.items():
                model_result["test"][language] = {"test_sha256": sha256_text(test_text), "metrics": evaluate_tokenizer(tokenizer, test_text)}
            replicate_result["models"][condition] = model_result
        for language in LANGUAGES:
            dedicated = replicate_result["models"][f"{language}_separate"]["test"][language]["metrics"]["tokens"]
            joint = replicate_result["models"]["shared_multilingual"]["test"][language]["metrics"]["tokens"]
            penalties[language].append((joint - dedicated) / dedicated * 100)
        results["replicates"][replicate] = replicate_result
    summary = {"experiment": EXPERIMENT_NAME, "vocab_size": VOCAB_SIZE, "replicates": {replicate: {language: {"dedicated_tokens": results["replicates"][replicate]["models"][f"{language}_separate"]["test"][language]["metrics"]["tokens"], "shared_tokens": results["replicates"][replicate]["models"]["shared_multilingual"]["test"][language]["metrics"]["tokens"], "penalty_percent": penalties[language][index]} for language in LANGUAGES} for index, replicate in enumerate(REPLICATES)}, "per_language": {language: {"penalties_percent": penalties[language], "mean_penalty_percent": statistics.mean(penalties[language]), "sample_stddev_percentage_points": statistics.stdev(penalties[language])} for language in LANGUAGES}}
    with OUTPUT_PATH.open("x", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with SUMMARY_PATH.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
