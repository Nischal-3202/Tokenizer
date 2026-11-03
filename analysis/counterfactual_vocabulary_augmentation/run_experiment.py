"""
Counterfactual vocabulary augmentation on saved shared byte-level BPE models.

Regime: repeated_vocabulary_sharing_v1, three replicates, vocabulary 1024.
No retraining. Original model files are read and never written.

Augmentation semantics
----------------------
A candidate is one learned merge from a dedicated tokenizer whose resulting
byte string is absent from the matching shared vocabulary. Its construction is
the dedicated merge's own child pair, not a new split.

A candidate is individually eligible only when all of the following hold
against the original shared model:

1. Both child byte strings exist as exactly one token in the shared vocabulary.
2. The child pair is not already a shared merge.
3. Encoding the candidate byte string in isolation, after appending that one
   merge at the next free id, returns exactly the new id.

New ids start at max(shared ids)+1 and increase in acceptance order. Lower id
still means higher priority, so every original shared merge outranks every
added merge. Original merge order is not edited.

Frequency interventions accept eligible candidates in descending training-emission
order. Emissions are counts from the dedicated tokenizer on that replicate's
training file. Held-out text is not read until every candidate list is frozen.
If an earlier added merge stops a later candidate from being built in isolation,
that candidate is skipped and recorded as blocked. Budgets are prefixes of the
accepted sequence: 25, 50, 100, 200, and all accepted pieces.

The random control shuffles the same eligible pool with Random(20261030 +
replicate index) and then applies the same cumulative rule. Combined
interventions alternate Nepali and Hindi pieces from those accepted sequences,
Nepali first in each pair, and drop a piece if it no longer builds in isolation
once the other language's merges are present. Combined budgets are per language.

Pieces whose dedicated children are not already in the shared vocabulary are
excluded. Introducing them would require new intermediate tokens and a priority
decision the saved shared merges do not determine.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bpe_fast import decode_bytes, encode_bytes, token_spans
from tokenizer.tokenizer import Tokenizer

OUT = Path(__file__).resolve().parent
MODEL_OUT = OUT / "models"
CONFIG = OUT / "configs"
FIG = OUT / "figures"
REPS = ("replicate_1", "replicate_2", "replicate_3")
LANGS = ("english", "hindi", "nepali")
BUDGETS = (25, 50, 100, 200, "all")
SEED = 20261030
SOURCE_EXPERIMENT = "repeated_vocabulary_sharing_v1"


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return path.read_text(encoding="utf-8")


def load_tokenizer(path: Path) -> Tokenizer:
    if not path.exists():
        raise FileNotFoundError(f"Missing required model: {path}")
    tok = Tokenizer()
    tok.load(path)
    return tok


def gloss(blob: bytes) -> str:
    out = []
    i = 0
    while i < len(blob):
        decoded = None
        for length in (4, 3, 2, 1):
            if i + length > len(blob):
                continue
            chunk = blob[i : i + length]
            try:
                text = chunk.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if text.encode("utf-8") == chunk:
                decoded = text
                i += length
                break
        if decoded is None:
            out.append(f"⟨{blob[i]:02x}⟩")
            i += 1
        else:
            out.append(decoded)
    return "".join(out).replace("\n", "\\n").replace("\r", "\\r")


def quantile(values: list[float], p: float):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def word_spans(text: str):
    spans = []
    byte_i = 0
    buf: list[str] = []
    start = 0
    for ch in text:
        size = len(ch.encode("utf-8"))
        if ch.isspace():
            if buf:
                spans.append((start, byte_i))
                buf = []
        else:
            if not buf:
                start = byte_i
            buf.append(ch)
        byte_i += size
    if buf:
        spans.append((start, byte_i))
    return spans


def overlap_counts(words, spans):
    counts = []
    j = 0
    n = len(spans)
    for start, end in words:
        while j < n and spans[j][1] <= start:
            j += 1
        k = j
        pieces = 0
        while k < n and spans[k][0] < end:
            pieces += 1
            k += 1
        counts.append(pieces)
    return counts


def script_labels(text: str) -> list[str]:
    labels = []
    for ch in text:
        if "\u0900" <= ch <= "\u097f":
            lab = "devanagari"
        elif ch.isascii() and ch.isalpha():
            lab = "latin"
        elif ch.isspace():
            lab = "whitespace"
        elif ch.isascii():
            lab = "ascii_other"
        else:
            lab = "other"
        labels.extend([lab] * len(ch.encode("utf-8")))
    return labels


def count_script(spans, labels, name: str) -> int:
    n = 0
    limit = len(labels)
    for start, _, _ in spans:
        if start < limit and labels[start] == name:
            n += 1
    return n


def boundary_account(ded_spans, ded_vocab, aug_spans, aug_bytes):
    import bisect

    d_bounds = {0}
    for _, end, _ in ded_spans:
        d_bounds.add(end)
    s_bounds = {0}
    for _, end, _ in aug_spans:
        s_bounds.add(end)
    extra = s_bounds - d_bounds
    crossed = d_bounds - s_bounds
    net = len(aug_spans) - len(ded_spans)
    if net != len(extra) - len(crossed):
        raise RuntimeError("Boundary accounting identity failed")
    aug_starts = [start for start, _, _ in aug_spans]
    extra_sorted = sorted(extra)
    extra_missing = extra_present = 0
    aligned_intact = aligned_frag_missing = aligned_frag_present = unaligned = 0
    for start, end, tid in ded_spans:
        blob = ded_vocab[tid]
        present = blob in aug_bytes
        if start in s_bounds and end in s_bounds:
            i0 = bisect.bisect_left(aug_starts, start)
            i1 = bisect.bisect_left(aug_starts, end)
            internal = max(0, (i1 - i0) - 1)
            if internal == 0 and (i1 - i0) == 1:
                aligned_intact += 1
            elif present:
                aligned_frag_present += 1
            else:
                aligned_frag_missing += 1
            if present:
                extra_present += internal
            else:
                extra_missing += internal
        else:
            unaligned += 1
            lo = bisect.bisect_right(extra_sorted, start)
            hi = bisect.bisect_left(extra_sorted, end)
            inside = hi - lo
            if present:
                extra_present += inside
            else:
                extra_missing += inside
    if extra_missing + extra_present != len(extra):
        raise RuntimeError("Extra-cut attribution failed")
    return {
        "net_extra_tokens": net,
        "extra_boundaries": len(extra),
        "extra_boundaries_inside_absent_dedicated_strings": extra_missing,
        "extra_boundaries_inside_present_dedicated_strings": extra_present,
        "dedicated_boundaries_crossed_by_longer_token": len(crossed),
        "aligned_intact": aligned_intact,
        "aligned_fragmented_absent": aligned_frag_missing,
        "aligned_fragmented_present": aligned_frag_present,
        "unaligned_dedicated_tokens": unaligned,
    }


class MergeState:
    """Copy of a shared merge table plus append-only lower-priority merges."""

    def __init__(self, tokenizer: Tokenizer):
        self.merges = dict(tokenizer.merges)
        self.vocab = dict(tokenizer.vocab)
        self.original_pairs = list(tokenizer.merges.items())
        self.by_bytes: dict[bytes, int] = {}
        for tid, blob in self.vocab.items():
            if blob in self.by_bytes:
                raise RuntimeError("Shared vocabulary has duplicate byte strings")
            self.by_bytes[blob] = tid
        self.added: list[dict] = []

    def snapshot(self) -> "MergeState":
        other = MergeState.__new__(MergeState)
        other.merges = dict(self.merges)
        other.vocab = dict(self.vocab)
        other.original_pairs = self.original_pairs
        other.by_bytes = dict(self.by_bytes)
        other.added = list(self.added)
        return other

    def original_order_intact(self) -> bool:
        current = list(self.merges.items())
        return current[: len(self.original_pairs)] == self.original_pairs

    def try_add(self, cand: dict) -> tuple[bool, str]:
        left = cand["left"]
        right = cand["right"]
        whole = cand["whole"]
        if whole in self.by_bytes:
            return False, "byte_string_already_present"
        if left not in self.by_bytes or right not in self.by_bytes:
            return False, "child_byte_string_absent"
        pair = (self.by_bytes[left], self.by_bytes[right])
        if pair in self.merges:
            return False, "pair_already_merged"
        new_id = max(self.vocab) + 1
        self.merges[pair] = new_id
        self.vocab[new_id] = whole
        built = encode_bytes(whole, self.merges)
        if built != [new_id]:
            del self.merges[pair]
            del self.vocab[new_id]
            return False, "not_constructed_given_higher_priority_merges"
        self.by_bytes[whole] = new_id
        record = {
            "new_id": new_id,
            "left_id": pair[0],
            "right_id": pair[1],
            "whole": whole,
            "train_emissions": cand["train_emissions"],
            "frequency_rank": cand["frequency_rank"],
            "source_language": cand["source_language"],
            "hex": cand["hex"],
        }
        self.added.append(record)
        if not self.original_order_intact():
            raise RuntimeError("Original merge order changed")
        return True, "added"


def save_model(path: Path, state: MergeState):
    payload = {
        "vocab_size": len(state.vocab),
        "base_experiment": SOURCE_EXPERIMENT,
        "augmentation": "append_only_lower_priority",
        "original_merge_count": len(state.original_pairs),
        "added_merge_count": len(state.added),
        "merges": [[list(pair), idx] for pair, idx in state.merges.items()],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def individually_eligible(shared: MergeState, cand: dict) -> tuple[bool, str]:
    """Test one candidate against the original shared table only."""
    trial = shared.snapshot()
    return trial.try_add(cand)


def accept_sequence(shared: MergeState, ordered: list[dict]) -> tuple[list[dict], list[tuple[dict, str]]]:
    state = shared.snapshot()
    accepted = []
    blocked = []
    for cand in ordered:
        ok, reason = state.try_add(cand)
        if ok:
            accepted.append(cand)
        else:
            blocked.append((cand, reason))
    return accepted, blocked


def budget_sizes(accepted: list[dict]) -> list[tuple[str, int]]:
    sizes = []
    for budget in BUDGETS:
        if budget == "all":
            sizes.append(("all", len(accepted)))
        else:
            sizes.append((str(budget), min(int(budget), len(accepted))))
    # Unique prefix lengths, keeping the nominal label of the first budget that hits it.
    return sizes


def interleave(shared: MergeState, nepali_seq: list[dict], hindi_seq: list[dict], per_language: int):
    state = shared.snapshot()
    i = j = 0
    got_ne: list[dict] = []
    got_hi: list[dict] = []
    dropped = []
    while (len(got_ne) < per_language or len(got_hi) < per_language) and (
        i < len(nepali_seq) or j < len(hindi_seq)
    ):
        if len(got_ne) < per_language and i < len(nepali_seq):
            cand = nepali_seq[i]
            i += 1
            ok, reason = state.try_add(cand)
            if ok:
                got_ne.append(cand)
            else:
                dropped.append((cand, reason))
        if len(got_hi) < per_language and j < len(hindi_seq):
            cand = hindi_seq[j]
            j += 1
            ok, reason = state.try_add(cand)
            if ok:
                got_hi.append(cand)
            else:
                dropped.append((cand, reason))
        if i >= len(nepali_seq) and len(got_ne) < per_language and j >= len(hindi_seq):
            break
        if i >= len(nepali_seq) and j >= len(hindi_seq):
            break
    return state, got_ne, got_hi, dropped


def svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_lines(path: Path, title: str, subtitle: str, series, xlabel: str, ylabel: str, caption: str):
    width, height = 760, 460
    left, right, top, bottom = 72, 24, 78, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = sorted({x for _, data in series for x, _ in data})
    if not xs:
        return
    xmin, xmax = min(xs), max(xs)
    if xmin == xmax:
        xmax = xmin + 1
    ymax = max((y for _, data in series for _, y in data), default=1)
    ymin = min((y for _, data in series for _, y in data), default=0)
    if ymax == ymin:
        ymax += 1
    pad = (ymax - ymin) * 0.12
    ymin -= pad
    ymax += pad

    def xy(x, y):
        px = left + plot_w * ((x - xmin) / (xmax - xmin))
        py = top + plot_h * (1 - (y - ymin) / (ymax - ymin))
        return px, py

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-family="Georgia, serif" font-size="16">{svg_escape(title)}</text>',
        f'<text x="{left}" y="48" font-family="Helvetica, sans-serif" font-size="11" fill="#444">{svg_escape(subtitle)}</text>',
    ]
    for frac in (0, 0.5, 1):
        yval = ymin + (ymax - ymin) * frac
        _, py = xy(xs[0], yval)
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left+plot_w}" y2="{py:.1f}" stroke="#e6e6e6"/>')
        parts.append(
            f'<text x="{left-8}" y="{py+4:.1f}" text-anchor="end" font-family="Helvetica, sans-serif" font-size="10">{yval:.1f}</text>'
        )
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>')
    parts.append(
        f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" font-family="Helvetica, sans-serif" font-size="11">{svg_escape(ylabel)}</text>'
    )
    colors = ["#009E73", "#E69F00", "#0072B2", "#D55E00", "#CC79A7", "#56B4E9"]
    for s_i, (name, data) in enumerate(series):
        color = colors[s_i % len(colors)]
        coords = []
        for x, y in data:
            if x not in xs:
                continue
            coords.append(xy(x, y))
        points = " ".join(f"{px:.1f},{py:.1f}" for px, py in coords)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        for px, py in coords:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>')
        parts.append(f'<rect x="{left + (s_i % 3) * 230}" y="{height - 68 + (s_i // 3) * 16}" width="10" height="10" fill="{color}"/>')
        parts.append(
            f'<text x="{left + 14 + (s_i % 3) * 230}" y="{height - 58 + (s_i // 3) * 16}" font-family="Helvetica, sans-serif" font-size="11">{svg_escape(name)}</text>'
        )
    tick_max = xmax
    ticks = [0]
    for tick in (50, 100, 200, 300):
        if xmin <= tick <= xmax:
            ticks.append(tick)
    if tick_max not in ticks:
        ticks.append(int(round(tick_max)))
    for tick in ticks:
        px, _ = xy(tick, ymin)
        parts.append(
            f'<text x="{px:.1f}" y="{top+plot_h+16}" text-anchor="middle" font-family="Helvetica, sans-serif" font-size="10">{tick}</text>'
        )
    parts.append(
        f'<text x="{left}" y="{height-14}" font-family="Helvetica, sans-serif" font-size="10" fill="#444">{svg_escape(caption)}</text>'
    )
    parts.append(
        f'<text x="{left+plot_w/2}" y="{height-30}" text-anchor="middle" font-family="Helvetica, sans-serif" font-size="11">{svg_escape(xlabel)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def fnum(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return value


def main():
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    published = json.loads(
        (ROOT / "results" / "repeated_vocabulary_sharing_v1.json").read_text(encoding="utf-8")
    )
    prior = list(
        csv.DictReader(
            (ROOT / "analysis" / "nepali_penalty_investigation" / "script_penalty.csv").open(encoding="utf-8")
        )
    )

    # Selection uses training files only. Test files are opened later.
    trains = {}
    specs = {}
    candidate_rows = []
    for r_i, rep in enumerate(REPS):
        print(f"Selecting candidates {rep}", flush=True)
        block = published["replicates"][rep]
        shared_path = ROOT / block["models"]["shared_multilingual"]["model_file"]
        shared_tok = load_tokenizer(shared_path)
        shared = MergeState(shared_tok)
        specs[rep] = {"shared": shared, "shared_path": str(shared_path.relative_to(ROOT)), "languages": {}}
        for lang in ("hindi", "nepali"):
            model_key = f"{lang}_separate"
            ded_path = ROOT / block["models"][model_key]["model_file"]
            ded = load_tokenizer(ded_path)
            train_path = ROOT / "data" / "controlled_multilingual_replicates" / rep / f"{lang}_train.txt"
            train = read_text(train_path)
            digest = hashlib.sha256(train.encode("utf-8")).hexdigest()
            if digest != block["training_data"][lang]["sha256"]:
                raise RuntimeError(f"Training hash mismatch for {rep} {lang}")
            trains[(rep, lang)] = train
            counts = Counter(encode_bytes(train.encode("utf-8"), ded.merges))
            missing = []
            for (left_id, right_id), idx in ded.merges.items():
                whole = ded.vocab[idx]
                if whole in shared.by_bytes:
                    continue
                missing.append(
                    {
                        "source_language": lang,
                        "dedicated_id": idx,
                        "train_emissions": counts[idx],
                        "left": ded.vocab[left_id],
                        "right": ded.vocab[right_id],
                        "whole": whole,
                        "hex": whole.hex(),
                        "byte_length": len(whole),
                    }
                )
            missing.sort(key=lambda c: (-c["train_emissions"], c["byte_length"], c["hex"]))
            for rank, cand in enumerate(missing, start=1):
                cand["frequency_rank"] = rank
            eligible = []
            for cand in missing:
                ok, reason = individually_eligible(shared, cand)
                cand["individually_eligible"] = ok
                cand["individual_reason"] = reason if ok else reason
                if ok:
                    eligible.append(cand)
                else:
                    cand["individual_reason"] = reason
            freq_accepted, freq_blocked = accept_sequence(shared, eligible)
            rng = random.Random(SEED + r_i)
            shuffled = list(eligible)
            rng.shuffle(shuffled)
            rand_accepted, rand_blocked = accept_sequence(shared, shuffled)
            blocked_freq = {id(c): reason for c, reason in freq_blocked}
            blocked_rand = {id(c): reason for c, reason in rand_blocked}
            freq_pos = {id(c): i + 1 for i, c in enumerate(freq_accepted)}
            rand_pos = {id(c): i + 1 for i, c in enumerate(rand_accepted)}
            specs[rep]["languages"][lang] = {
                "dedicated": ded,
                "ded_path": str(ded_path.relative_to(ROOT)),
                "eligible": eligible,
                "freq_accepted": freq_accepted,
                "rand_accepted": rand_accepted,
                "missing_count": len(missing),
                "eligible_count": len(eligible),
            }
            for cand in missing:
                candidate_rows.append(
                    {
                        "replicate": rep,
                        "source_language": lang,
                        "frequency_rank": cand["frequency_rank"],
                        "train_emissions": cand["train_emissions"],
                        "byte_length": cand["byte_length"],
                        "hex": cand["hex"],
                        "gloss": gloss(cand["whole"]),
                        "left_hex": cand["left"].hex(),
                        "right_hex": cand["right"].hex(),
                        "left_gloss": gloss(cand["left"]),
                        "right_gloss": gloss(cand["right"]),
                        "individually_eligible": int(cand["individually_eligible"]),
                        "exclusion_or_status": cand["individual_reason"],
                        "frequency_addition_order": freq_pos.get(id(cand), ""),
                        "frequency_cumulative_block": blocked_freq.get(id(cand), ""),
                        "random_addition_order": rand_pos.get(id(cand), ""),
                        "random_cumulative_block": blocked_rand.get(id(cand), ""),
                    }
                )
            config = {
                "replicate": rep,
                "source_language": lang,
                "training_file": str(train_path.relative_to(ROOT)),
                "training_sha256": digest,
                "missing_dedicated_strings": len(missing),
                "individually_eligible": len(eligible),
                "frequency_accepted": len(freq_accepted),
                "random_accepted": len(rand_accepted),
                "random_seed": SEED + r_i,
                "frequency_order": [
                    {
                        "addition_order": i + 1,
                        "frequency_rank": c["frequency_rank"],
                        "train_emissions": c["train_emissions"],
                        "hex": c["hex"],
                        "gloss": gloss(c["whole"]),
                    }
                    for i, c in enumerate(freq_accepted)
                ],
                "random_order": [
                    {
                        "addition_order": i + 1,
                        "frequency_rank": c["frequency_rank"],
                        "train_emissions": c["train_emissions"],
                        "hex": c["hex"],
                    }
                    for i, c in enumerate(rand_accepted)
                ],
            }
            (CONFIG / f"{rep}_{lang}_candidates.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"  {lang}: missing {len(missing)}, eligible {len(eligible)}, "
                f"frequency-accepted {len(freq_accepted)}, random-accepted {len(rand_accepted)}",
                flush=True,
            )

    semantics = {
        "experiment": "counterfactual_vocabulary_augmentation",
        "source": SOURCE_EXPERIMENT,
        "priority": "Original shared merge ids are unchanged and always outrank added merges. Added ids increase in acceptance order.",
        "candidate_identity": "Dedicated merge parent byte string, with that merge's own child byte strings.",
        "frequency": "Dedicated-tokenizer emission count on the replicate training file. Test text is not used.",
        "eligibility": "Both children exist uniquely in the original shared vocabulary, and isolation encoding returns the new id.",
        "cumulative_block": "A later eligible piece is skipped if higher-priority added merges prevent isolation construction.",
        "budgets": ["25", "50", "100", "200", "all"],
        "combined_budget": "Per language. Turns alternate, Nepali then Hindi.",
        "random_seed": "random.Random(20261030 + replicate_index), replicate_1 index 0.",
        "excluded_without_shared_children": "Not added. A prerequisite merge would be a new token the frequency rank did not independently justify, and it would need its own priority.",
        "not_a_retrained_tokenizer": True,
    }
    (CONFIG / "augmentation_semantics.json").write_text(
        json.dumps(semantics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Evaluation starts only after candidate lists are frozen.
    print("Loading held-out tests", flush=True)
    tests = {lang: read_text(ROOT / "data" / "processed" / lang / "test.txt") for lang in LANGS}
    for lang, text in tests.items():
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        recorded = published["test_data"][lang]["sha256"]
        if digest != recorded:
            raise RuntimeError(f"Held-out hash mismatch for {lang}")
    labels = {lang: script_labels(tests[lang]) for lang in LANGS}
    words = {lang: word_spans(tests[lang]) for lang in LANGS}
    test_bytes = {lang: tests[lang].encode("utf-8") for lang in LANGS}

    # Dedicated encodings once.
    dedicated_cache = {}
    for rep in REPS:
        block = published["replicates"][rep]
        for lang in LANGS:
            ded = load_tokenizer(ROOT / block["models"][f"{lang}_separate"]["model_file"])
            ids = encode_bytes(test_bytes[lang], ded.merges)
            expect = block["models"][f"{lang}_separate"]["test"][lang]["metrics"]["tokens"]
            if len(ids) != expect:
                raise RuntimeError(f"Dedicated baseline mismatch {rep} {lang}: {len(ids)} != {expect}")
            if decode_bytes(ids, ded.vocab) != test_bytes[lang]:
                raise RuntimeError(f"Dedicated roundtrip failed {rep} {lang}")
            spans = token_spans(ids, ded.vocab)
            dedicated_cache[(rep, lang)] = {
                "tokenizer": ded,
                "ids": ids,
                "spans": spans,
                "tokens": len(ids),
                "devanagari": count_script(spans, labels[lang], "devanagari"),
            }
            prior_row = [
                r
                for r in prior
                if r["experiment"] == SOURCE_EXPERIMENT
                and r["replicate"] == rep
                and r["language"] == lang
                and r["script_at_token_start"] == "devanagari"
            ]
            if prior_row and int(prior_row[0]["dedicated_tokens"]) != dedicated_cache[(rep, lang)]["devanagari"]:
                raise RuntimeError(f"Devanagari dedicated count drifted for {rep} {lang}")

    def evaluate_state(rep, state: MergeState, intervention: str, budget_label: str, pieces_ne: int, pieces_hi: int):
        rows_main = []
        rows_bound = []
        rows_word = []
        rows_cross = []
        if not state.original_order_intact():
            raise RuntimeError("Original merges lost their order")
        aug_bytes = set(state.vocab.values())
        model_name = f"{rep}__{intervention}__budget_{budget_label}.json"
        if intervention != "baseline":
            save_model(MODEL_OUT / model_name, state)
        for lang in LANGS:
            ids = encode_bytes(test_bytes[lang], state.merges)
            if decode_bytes(ids, state.vocab) != test_bytes[lang]:
                raise RuntimeError(f"Roundtrip failed {rep} {intervention} {budget_label} {lang}")
            spans = token_spans(ids, state.vocab)
            ded = dedicated_cache[(rep, lang)]
            dev = count_script(spans, labels[lang], "devanagari")
            penalty = (len(ids) / ded["tokens"] - 1) * 100
            dev_penalty = (dev / ded["devanagari"] - 1) * 100 if ded["devanagari"] else None
            published_shared = published["replicates"][rep]["models"]["shared_multilingual"]["test"][lang]["metrics"]["tokens"]
            if intervention == "baseline" and len(ids) != published_shared:
                raise RuntimeError(
                    f"Shared baseline mismatch {rep} {lang}: {len(ids)} != {published_shared}"
                )
            if intervention == "baseline" and lang != "english":
                prior_row = [
                    r
                    for r in prior
                    if r["experiment"] == SOURCE_EXPERIMENT
                    and r["replicate"] == rep
                    and r["language"] == lang
                    and r["script_at_token_start"] == "devanagari"
                ]
                if prior_row and int(prior_row[0]["shared_tokens"]) != dev:
                    raise RuntimeError(f"Devanagari shared count drifted for {rep} {lang}")
            account = boundary_account(ded["spans"], ded["tokenizer"].vocab, spans, aug_bytes)
            d_counts = overlap_counts(words[lang], ded["spans"])
            a_counts = overlap_counts(words[lang], spans)
            increases = []
            more = less = equal = 0
            extra_sum = 0
            for d_n, a_n in zip(d_counts, a_counts):
                extra_sum += a_n - d_n
                if d_n <= 0:
                    continue
                increases.append((a_n - d_n) / d_n * 100)
                if a_n > d_n:
                    more += 1
                elif a_n < d_n:
                    less += 1
                else:
                    equal += 1
            n_words = len(d_counts)
            base_tokens = None
            rows_main.append(
                {
                    "replicate": rep,
                    "intervention": intervention,
                    "budget": budget_label,
                    "pieces_added_nepali": pieces_ne,
                    "pieces_added_hindi": pieces_hi,
                    "pieces_added_total": pieces_ne + pieces_hi,
                    "language": lang,
                    "dedicated_tokens": ded["tokens"],
                    "model_tokens": len(ids),
                    "penalty_percent": fnum(penalty),
                    "devanagari_dedicated_tokens": ded["devanagari"],
                    "devanagari_model_tokens": dev,
                    "devanagari_penalty_percent": fnum(dev_penalty),
                    "roundtrip_exact_bytes": 1,
                    "original_merge_order_intact": 1,
                }
            )
            rows_bound.append(
                {
                    "replicate": rep,
                    "intervention": intervention,
                    "budget": budget_label,
                    "language": lang,
                    **account,
                }
            )
            rows_word.append(
                {
                    "replicate": rep,
                    "intervention": intervention,
                    "budget": budget_label,
                    "language": lang,
                    "word_occurrences": n_words,
                    "fraction_more_tokens": fnum(more / n_words if n_words else None),
                    "fraction_fewer_tokens": fnum(less / n_words if n_words else None),
                    "fraction_equal_tokens": fnum(equal / n_words if n_words else None),
                    "mean_additional_tokens_per_word": fnum(extra_sum / n_words if n_words else None),
                    "median_percent_increase": fnum(quantile(increases, 0.5)),
                    "p90_percent_increase": fnum(quantile(increases, 0.9)),
                    "mean_percent_increase": fnum(sum(increases) / len(increases) if increases else None),
                }
            )
            rows_cross.append(
                {
                    "replicate": rep,
                    "intervention": intervention,
                    "budget": budget_label,
                    "evaluated_language": lang,
                    "model_tokens": len(ids),
                    "penalty_percent": fnum(penalty),
                    "devanagari_model_tokens": dev,
                    "devanagari_penalty_percent": fnum(dev_penalty),
                }
            )
            _ = base_tokens
        return rows_main, rows_bound, rows_word, rows_cross

    main_rows = []
    bound_rows = []
    word_rows = []
    cross_rows = []

    for rep in REPS:
        print(f"Evaluating {rep}", flush=True)
        shared = specs[rep]["shared"]
        ne = specs[rep]["languages"]["nepali"]["freq_accepted"]
        hi = specs[rep]["languages"]["hindi"]["freq_accepted"]
        ne_r = specs[rep]["languages"]["nepali"]["rand_accepted"]
        hi_r = specs[rep]["languages"]["hindi"]["rand_accepted"]

        configs = [("baseline", "0", shared.snapshot(), 0, 0)]

        def add_prefixes(intervention, sequence, source_lang):
            for label, size in budget_sizes(sequence):
                state = shared.snapshot()
                for cand in sequence[:size]:
                    ok, reason = state.try_add(cand)
                    if not ok:
                        raise RuntimeError(f"Prefix piece failed for {intervention} {label}: {reason}")
                ne_n = size if source_lang == "nepali" else 0
                hi_n = size if source_lang == "hindi" else 0
                configs.append((intervention, label, state, ne_n, hi_n))

        add_prefixes("nepali_frequency", ne, "nepali")
        add_prefixes("hindi_frequency", hi, "hindi")
        add_prefixes("nepali_random", ne_r, "nepali")
        add_prefixes("hindi_random", hi_r, "hindi")

        def add_combined(intervention, ne_seq, hi_seq):
            for budget in BUDGETS:
                if budget == "all":
                    target = max(len(ne_seq), len(hi_seq))
                    label = "all"
                else:
                    target = int(budget)
                    label = str(budget)
                state, got_ne, got_hi, _dropped = interleave(shared, ne_seq, hi_seq, target)
                configs.append((intervention, label, state, len(got_ne), len(got_hi)))

        add_combined("combined_frequency", ne, hi)
        add_combined("combined_random", ne_r, hi_r)

        # The combined builder above can append duplicate budgets. Keep the last exact one.
        # Baseline and single-language configs are unique. Combined may have been appended
        # inside add_combined only through configs.append at the end. Good.
        seen = set()
        for intervention, label, state, n_ne, n_hi in configs:
            key = (intervention, label)
            if key in seen and intervention.startswith("combined"):
                continue
            seen.add(key)
            print(f"  {intervention} budget {label} (+{n_ne} ne, +{n_hi} hi)", flush=True)
            a, b, c, d = evaluate_state(rep, state, intervention, label, n_ne, n_hi)
            main_rows.extend(a)
            bound_rows.extend(b)
            word_rows.extend(c)
            cross_rows.extend(d)

    # Deltas versus baseline for the cross-language file.
    baseline_tokens = {
        (r["replicate"], r["evaluated_language"]): r
        for r in cross_rows
        if r["intervention"] == "baseline"
    }
    for row in cross_rows:
        base = baseline_tokens[(row["replicate"], row["evaluated_language"])]
        row["token_delta_vs_baseline"] = int(row["model_tokens"]) - int(base["model_tokens"])
        row["devanagari_token_delta_vs_baseline"] = int(row["devanagari_model_tokens"]) - int(
            base["devanagari_model_tokens"]
        )

    write_csv(OUT / "augmentation_results.csv", [{k: fnum(v) if isinstance(v, float) else v for k, v in r.items()} for r in main_rows])
    write_csv(OUT / "candidate_pieces.csv", candidate_rows)
    write_csv(OUT / "boundary_decomposition.csv", bound_rows)
    write_csv(OUT / "word_fragmentation_results.csv", word_rows)
    write_csv(OUT / "cross_language_effects.csv", cross_rows)

    # Figures: mean Devanagari penalty and overall penalty versus pieces added.
    def mean_metric(intervention, language, field, xfield):
        groups = defaultdict(list)
        for row in main_rows:
            if row["intervention"] == intervention and row["language"] == language:
                groups[int(row[xfield])].append(float(row[field]) if row[field] != "" else None)
        points = []
        for x in sorted(groups):
            vals = [v for v in groups[x] if v is not None]
            if vals:
                points.append((x, sum(vals) / len(vals)))
        return points

    write_lines(
        FIG / "devanagari_penalty_vs_budget.svg",
        "Devanagari-starting token penalty versus added pieces",
        "Mean of three replicates. X is pieces added from that language, or per language for combined.",
        [
            ("Nepali pieces on Nepali", [(0, mean_metric("baseline", "nepali", "devanagari_penalty_percent", "pieces_added_total")[0][1])] + [
                (x, y) for x, y in mean_metric("nepali_frequency", "nepali", "devanagari_penalty_percent", "pieces_added_nepali")
            ]),
            ("Hindi pieces on Hindi", [(0, mean_metric("baseline", "hindi", "devanagari_penalty_percent", "pieces_added_total")[0][1])] + [
                (x, y) for x, y in mean_metric("hindi_frequency", "hindi", "devanagari_penalty_percent", "pieces_added_hindi")
            ]),
            ("Hindi pieces on Nepali", [(0, mean_metric("baseline", "nepali", "devanagari_penalty_percent", "pieces_added_total")[0][1])] + [
                (x, y) for x, y in mean_metric("hindi_frequency", "nepali", "devanagari_penalty_percent", "pieces_added_hindi")
            ]),
            ("Nepali pieces on Hindi", [(0, mean_metric("baseline", "hindi", "devanagari_penalty_percent", "pieces_added_total")[0][1])] + [
                (x, y) for x, y in mean_metric("nepali_frequency", "hindi", "devanagari_penalty_percent", "pieces_added_nepali")
            ]),
        ],
        "Added pieces",
        "Devanagari penalty (%)",
        "Source: repeated_vocabulary_sharing_v1 copies. Added merges rank below every original shared merge.",
    )
    write_lines(
        FIG / "overall_penalty_vs_budget.svg",
        "Overall tokenization penalty versus added pieces",
        "Mean of three replicates. Penalty is (model tokens / dedicated tokens − 1) × 100.",
        [
            ("Nepali frequency → Nepali", [(0, mean_metric("baseline", "nepali", "penalty_percent", "pieces_added_total")[0][1])] + [
                (x, y) for x, y in mean_metric("nepali_frequency", "nepali", "penalty_percent", "pieces_added_nepali")
            ]),
            ("Hindi frequency → Hindi", [(0, mean_metric("baseline", "hindi", "penalty_percent", "pieces_added_total")[0][1])] + [
                (x, y) for x, y in mean_metric("hindi_frequency", "hindi", "penalty_percent", "pieces_added_hindi")
            ]),
            ("Nepali random → Nepali", [(0, mean_metric("baseline", "nepali", "penalty_percent", "pieces_added_total")[0][1])] + [
                (x, y) for x, y in mean_metric("nepali_random", "nepali", "penalty_percent", "pieces_added_nepali")
            ]),
            ("Combined frequency → Nepali", [(0, mean_metric("baseline", "nepali", "penalty_percent", "pieces_added_total")[0][1])] + [
                (x, y) for x, y in mean_metric("combined_frequency", "nepali", "penalty_percent", "pieces_added_nepali")
            ]),
        ],
        "Added pieces from the named language",
        "Overall penalty (%)",
        "Source: fixed held-out tests. Combined x-axis is the Nepali piece count in the equal per-language budget.",
    )
    def delta_points(intervention, language, piece_field):
        base = {
            row["replicate"]: int(row["model_tokens"])
            for row in main_rows
            if row["intervention"] == "baseline" and row["language"] == language
        }
        groups = defaultdict(list)
        for row in main_rows:
            if row["intervention"] != intervention or row["language"] != language:
                continue
            x = int(row[piece_field])
            groups[x].append(int(row["model_tokens"]) - base[row["replicate"]])
        return [(x, sum(vals) / len(vals)) for x, vals in sorted(groups.items())]

    write_lines(
        FIG / "cross_language_token_delta.svg",
        "Token-count change versus the original shared model",
        "Mean of three replicates. Negative means fewer tokens than the unmodified shared vocabulary.",
        [
            ("Nepali pieces, Nepali test", delta_points("nepali_frequency", "nepali", "pieces_added_nepali")),
            ("Nepali pieces, Hindi test", delta_points("nepali_frequency", "hindi", "pieces_added_nepali")),
            ("Nepali pieces, English test", delta_points("nepali_frequency", "english", "pieces_added_nepali")),
            ("Hindi pieces, Hindi test", delta_points("hindi_frequency", "hindi", "pieces_added_hindi")),
            ("Hindi pieces, Nepali test", delta_points("hindi_frequency", "nepali", "pieces_added_hindi")),
            ("Hindi pieces, English test", delta_points("hindi_frequency", "english", "pieces_added_hindi")),
        ],
        "Added pieces",
        "Token delta versus baseline shared model",
        "Source: repeated_vocabulary_sharing_v1 held-out tests. Frequency augmentation.",
    )
    print(f"Wrote results under {OUT}", flush=True)


if __name__ == "__main__":
    main()
