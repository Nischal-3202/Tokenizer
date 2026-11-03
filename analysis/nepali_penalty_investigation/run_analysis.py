"""
Why Nepali's shared-vocabulary penalty is higher.

Reads saved byte-level BPE models, training corpora, and held-out tests.
Does not train, and does not write outside analysis/nepali_penalty_investigation/.

Primary regime: repeated_vocabulary_sharing_v1
  dedicated training ~300 KB per language, shared training ~900 KB, vocab 1024.
Confirming regime: total_budget_matched_v1
  dedicated and shared use the same ~100 KB per-language samples, vocab 1024.

Identity of a learned piece is its byte string, not its integer id and not its
Unicode appearance. Partial UTF-8 sequences stay in the analysis.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import regex
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bpe_fast import encode_bytes, token_spans
from tokenizer.tokenizer import Tokenizer

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
LANGUAGES = ("english", "hindi", "nepali")
REPLICATES = ("replicate_1", "replicate_2", "replicate_3")
COND = {
    "english": "english_separate",
    "hindi": "hindi_separate",
    "nepali": "nepali_separate",
    "shared": "shared_multilingual",
}
PREDOMINANT = 0.80
MIN_WORD_OCCURRENCES = 5
SUFFIX_MIN_N = 40
NGRAM_NS = (2, 3, 6, 9)
NGRAM_MIN_COUNT = 10

EXPERIMENTS = {
    "repeated_vocabulary_sharing_v1": {
        "results": ROOT / "results" / "repeated_vocabulary_sharing_v1.json",
        "label": "fixed_per_language_about_900KB_shared",
    },
    "total_budget_matched_v1": {
        "results": ROOT / "results" / "total_budget_matched_v1.json",
        "label": "total_budget_matched_about_300KB_shared",
    },
}


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


def utf8_text(blob: bytes) -> tuple[str, bool]:
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return "", False
    return text.replace("\r", "\\r").replace("\n", "\\n"), True


def gloss_bytes(blob: bytes) -> str:
    """Decode complete UTF-8 characters and leave incomplete bytes visible."""
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
    return "".join(out).replace("\r", "\\r").replace("\n", "\\n")


def piece_label(blob: bytes) -> str:
    return gloss_bytes(blob)


def shannon(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for value in counts.values():
        p = value / total
        h -= p * math.log2(p)
    return h


def script_of_char(ch: str) -> str:
    if "\u0900" <= ch <= "\u097f":
        return "devanagari"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isspace():
        return "whitespace"
    if ch.isascii():
        return "ascii_other"
    return "other"


def annotate(text: str):
    """Per-byte script labels, codepoint starts, grapheme ends, char metadata."""
    script_at = []
    cp_starts = [0]
    categories = Counter()
    scripts = Counter()
    dependents = Counter()
    dev_letters = 0
    grapheme_cp = Counter()
    byte_pos = 0
    for ch in text:
        blob = ch.encode("utf-8")
        lab = script_of_char(ch)
        script_at.extend([lab] * len(blob))
        scripts[lab] += 1
        categories[unicodedata.category(ch)] += 1
        cp = ord(ch)
        byte_pos += len(blob)
        cp_starts.append(byte_pos)
        if "\u0900" <= ch <= "\u097f":
            name = unicodedata.name(ch, "")
            if "VOWEL SIGN" in name:
                dependents["vowel_sign"] += 1
            elif ch == "\u094d":
                dependents["virama"] += 1
            elif ch == "\u093c":
                dependents["nukta"] += 1
            elif ch == "\u0902":
                dependents["anusvara"] += 1
            elif ch == "\u0901":
                dependents["candrabindu"] += 1
            elif ch == "\u0903":
                dependents["visarga"] += 1
            elif unicodedata.category(ch) == "Lo":
                dev_letters += 1
    graphemes = list(regex.finditer(r"\X", text))
    g_ends = set()
    gpos = 0
    multi_cp_graphemes = 0
    for match in graphemes:
        g = match.group()
        grapheme_cp[len(g)] += 1
        if len(g) > 1:
            multi_cp_graphemes += 1
        gpos += len(g.encode("utf-8"))
        g_ends.add(gpos)
    cp_set = set(cp_starts)
    return {
        "script_at": script_at,
        "cp_starts": cp_set,
        "g_ends": g_ends,
        "categories": categories,
        "scripts": scripts,
        "dependents": dependents,
        "dev_letters": dev_letters,
        "grapheme_cp": grapheme_cp,
        "n_graphemes": len(graphemes),
        "multi_cp_graphemes": multi_cp_graphemes,
        "n_bytes": len(script_at),
        "n_chars": len(text),
    }


def word_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    byte_i = 0
    buf: list[str] = []
    start_b = 0
    for ch in text:
        bl = len(ch.encode("utf-8"))
        if ch.isspace():
            if buf:
                spans.append((start_b, byte_i, "".join(buf)))
                buf = []
        else:
            if not buf:
                start_b = byte_i
            buf.append(ch)
        byte_i += bl
    if buf:
        spans.append((start_b, byte_i, "".join(buf)))
    return spans


def devanagari_runs(text: str) -> list[bytes]:
    runs = []
    buf = bytearray()
    for ch in text:
        if "\u0900" <= ch <= "\u097f":
            buf.extend(ch.encode("utf-8"))
        elif buf:
            runs.append(bytes(buf))
            buf = bytearray()
    if buf:
        runs.append(bytes(buf))
    return runs


def has_letter(word: str) -> bool:
    return any(unicodedata.category(ch).startswith("L") for ch in word)


def is_dependent(ch: str) -> bool:
    if not ("\u0900" <= ch <= "\u097f"):
        return False
    if unicodedata.category(ch) in {"Mn", "Mc"}:
        return True
    return ch in {"\u0901", "\u0902", "\u0903", "\u093c", "\u094d"}


def char_byte_index(text: str) -> list[tuple[int, str]]:
    out = []
    pos = 0
    for ch in text:
        out.append((pos, ch))
        pos += len(ch.encode("utf-8"))
    return out


def allocation_class(counts: dict[str, int], threshold: float) -> str:
    total = sum(counts.values())
    if total <= 0:
        return "unused"
    positive = [lang for lang, n in counts.items() if n > 0]
    if len(positive) == 1:
        return f"exclusive_{positive[0]}"
    top = max(LANGUAGES, key=lambda lang: counts[lang])
    if counts[top] / total >= threshold:
        return f"predominant_{top}"
    return "cross_lingual"


def raw_class(counts: dict[str, int]) -> str:
    present = [lang for lang in LANGUAGES if counts[lang] > 0]
    if not present:
        return "none"
    if len(present) == 1:
        return f"only_{present[0]}"
    return "+".join(present)


def structural_class(blob: bytes) -> str:
    text, ok = utf8_text(blob)
    if not ok:
        return "partial_utf8"
    has_dev = any("\u0900" <= ch <= "\u097f" for ch in text)
    has_lat = any(ch.isascii() and ch.isalpha() for ch in text)
    if has_dev and has_lat:
        return "mixed_script"
    if has_dev:
        return "devanagari_utf8"
    if all(b < 128 for b in blob):
        return "ascii"
    if has_lat:
        return "latin_utf8"
    return "other_utf8"


def merge_records(tok: Tokenizer):
    rules = set()
    produced = {}
    order = []
    for (left_id, right_id), idx in tok.merges.items():
        left = tok.vocab[left_id]
        right = tok.vocab[right_id]
        whole = left + right
        rules.add((left, right))
        produced[whole] = idx
        order.append((idx, whole, left, right))
    return rules, produced, order


def ngram_table(data: bytes, n: int) -> Counter:
    counts: Counter = Counter()
    if len(data) < n:
        return counts
    for i in range(len(data) - n + 1):
        counts[data[i : i + n]] += 1
    return counts


def ngram_table_runs(runs: list[bytes], n: int) -> Counter:
    counts: Counter = Counter()
    for blob in runs:
        if len(blob) < n:
            continue
        for i in range(len(blob) - n + 1):
            counts[blob[i : i + n]] += 1
    return counts


def coverage_stats(counts: Counter, vocab_bytes: set[bytes], min_count: int, top_k: int):
    filtered = [(g, c) for g, c in counts.items() if c >= min_count]
    mass = sum(c for _, c in filtered)
    mass_hit = sum(c for g, c in filtered if g in vocab_bytes)
    types = len(filtered)
    types_hit = sum(1 for g, _ in filtered if g in vocab_bytes)
    top = counts.most_common(top_k)
    top_mass = sum(c for _, c in top)
    top_hit = sum(c for g, c in top if g in vocab_bytes)
    return {
        "types": types,
        "types_in_vocab": types_hit,
        "type_coverage": types_hit / types if types else None,
        "mass": mass,
        "mass_in_vocab": mass_hit,
        "mass_coverage": mass_hit / mass if mass else None,
        "top_k": len(top),
        "top_mass_coverage": top_hit / top_mass if top_mass else None,
        "entropy_bits": shannon(counts),
    }


def overlaps(words, spans):
    """Pieces of a global tokenization that overlap each word span."""
    assigned = []
    j = 0
    n = len(spans)
    for start, end, word in words:
        while j < n and spans[j][1] <= start:
            j += 1
        k = j
        pieces = []
        while k < n and spans[k][0] < end:
            pieces.append(spans[k])
            k += 1
        assigned.append(pieces)
    return assigned


def render_pieces(pieces, vocab, word_start, word_end) -> str:
    """Show only the bytes of each overlapping token that fall inside the word."""
    labels = []
    for start, end, tid in pieces:
        blob = vocab[tid]
        lo = max(start, word_start)
        hi = min(end, word_end)
        clipped = blob[lo - start : hi - start]
        text = gloss_bytes(clipped)
        if start < word_start:
            text = "…" + text
        if end > word_end:
            text = text + "…"
        labels.append(text)
    rendered = " | ".join(labels)
    if len(rendered) > 240:
        return rendered[:237] + "..."
    return rendered


def boundary_sets(spans):
    bounds = {0}
    for _, end, _ in spans:
        bounds.add(end)
    return bounds


def pct(part, whole):
    if not whole:
        return None
    return 100.0 * part / whole


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fnum(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return value


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def write_grouped_svg(path, title, subtitle, categories, series, ylabel, caption):
    width, height = 760, 460
    left, right, top, bottom = 64, 20, 78, 108
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [v for _, _, data in series for v in data]
    ymax = max(values) if values else 1
    if ymax <= 0:
        ymax = 1
    ymax *= 1.18
    n = len(categories)
    k = max(1, len(series))
    group_w = plot_w / max(1, n)
    bar_w = min(18, (group_w * 0.8) / k)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-family="Georgia, serif" font-size="16">{svg_escape(title)}</text>',
        f'<text x="{left}" y="48" font-family="Helvetica, sans-serif" font-size="11" fill="#444">{svg_escape(subtitle)}</text>',
    ]
    for frac in (0, 0.5, 1):
        y = top + plot_h * (1 - frac)
        val = ymax * frac
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(
            f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Helvetica, sans-serif" font-size="10" fill="#333">{val:.0f}</text>'
        )
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>')
    parts.append(
        f'<text x="16" y="{top+plot_h/2}" transform="rotate(-90 16 {top+plot_h/2})" font-family="Helvetica, sans-serif" font-size="11">{svg_escape(ylabel)}</text>'
    )
    for i, cat in enumerate(categories):
        gx = left + i * group_w + group_w * 0.1
        for s_i, (_, color, data) in enumerate(series):
            val = data[i]
            bh = 0 if ymax == 0 else plot_h * (val / ymax)
            x = gx + s_i * bar_w
            y = top + plot_h - bh
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-1:.1f}" height="{bh:.1f}" fill="{color}"/>')
        parts.append(
            f'<text x="{left + i*group_w + group_w/2:.1f}" y="{top+plot_h+16}" text-anchor="middle" font-family="Helvetica, sans-serif" font-size="11">{svg_escape(cat)}</text>'
        )
    lx = left
    ly = height - 62
    for name, color, _ in series:
        parts.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{color}"/>')
        parts.append(
            f'<text x="{lx+14}" y="{ly+10}" font-family="Helvetica, sans-serif" font-size="11">{svg_escape(name)}</text>'
        )
        lx += 18 + 7 * len(name)
    parts.append(
        f'<text x="{left}" y="{height-18}" font-family="Helvetica, sans-serif" font-size="10" fill="#444">{svg_escape(caption)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_line_svg(path, title, subtitle, xlabels, series, ylabel, caption):
    width, height = 760, 440
    left, right, top, bottom = 64, 20, 78, 100
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [v for _, _, data in series for v in data]
    ymax = max(values) if values else 1
    ymax = max(ymax * 1.15, 0.01)
    n = len(xlabels)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-family="Georgia, serif" font-size="16">{svg_escape(title)}</text>',
        f'<text x="{left}" y="48" font-family="Helvetica, sans-serif" font-size="11" fill="#444">{svg_escape(subtitle)}</text>',
    ]
    for frac in (0, 0.5, 1):
        y = top + plot_h * (1 - frac)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(
            f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Helvetica, sans-serif" font-size="10">{ymax*frac:.0f}</text>'
        )
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>')
    parts.append(
        f'<text x="16" y="{top+plot_h/2}" transform="rotate(-90 16 {top+plot_h/2})" font-family="Helvetica, sans-serif" font-size="11">{svg_escape(ylabel)}</text>'
    )

    def xy(i, val):
        x = left + (0 if n == 1 else plot_w * i / (n - 1))
        y = top + plot_h * (1 - val / ymax)
        return x, y

    for name, color, data in series:
        coords = [xy(i, v) for i, v in enumerate(data)]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        for x, y in coords:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
    for i, lab in enumerate(xlabels):
        x, _ = xy(i, 0)
        parts.append(
            f'<text x="{x:.1f}" y="{top+plot_h+16}" text-anchor="middle" font-family="Helvetica, sans-serif" font-size="10">{svg_escape(lab)}</text>'
        )
    lx = left
    ly = height - 36
    for name, color, _ in series:
        parts.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{color}"/>')
        parts.append(
            f'<text x="{lx+14}" y="{ly+10}" font-family="Helvetica, sans-serif" font-size="11">{svg_escape(name)}</text>'
        )
        lx += 18 + 7 * len(name)
    parts.append(
        f'<text x="{left}" y="{height-12}" font-family="Helvetica, sans-serif" font-size="10" fill="#444">{svg_escape(caption)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def analyze_pair(text, ded_ids, ded_vocab, shared_ids, shared_vocab, shared_bytes):
    import bisect

    ded_sp = token_spans(ded_ids, ded_vocab)
    sh_sp = token_spans(shared_ids, shared_vocab)
    n_bytes = len(text.encode("utf-8"))
    if (ded_sp[-1][1] if ded_sp else 0) != n_bytes or (sh_sp[-1][1] if sh_sp else 0) != n_bytes:
        raise RuntimeError("Token spans do not cover the UTF-8 text")
    d_bounds = boundary_sets(ded_sp)
    s_bounds = boundary_sets(sh_sp)
    extra = s_bounds - d_bounds
    saved = d_bounds - s_bounds
    net = len(sh_sp) - len(ded_sp)
    if net != len(extra) - len(saved):
        raise RuntimeError("Boundary accounting identity failed")

    shared_starts = [start for start, _, _ in sh_sp]
    extra_sorted = sorted(extra)
    extra_missing = 0
    extra_present = 0
    aligned_intact = 0
    aligned_frag_missing = 0
    aligned_frag_present = 0
    unaligned = 0
    split_examples = defaultdict(Counter)

    for start, end, tid in ded_sp:
        blob = ded_vocab[tid]
        in_shared = blob in shared_bytes
        start_ok = start in s_bounds
        end_ok = end in s_bounds
        if start_ok and end_ok:
            i0 = bisect.bisect_left(shared_starts, start)
            i1 = bisect.bisect_left(shared_starts, end)
            pieces = sh_sp[i0:i1]
            internal = max(0, len(pieces) - 1)
            if internal == 0 and len(pieces) == 1:
                aligned_intact += 1
            else:
                if in_shared:
                    aligned_frag_present += 1
                else:
                    aligned_frag_missing += 1
                if internal > 0:
                    rendered = " | ".join(piece_label(shared_vocab[p[2]]) for p in pieces)
                    split_examples[blob][rendered] += 1
            if in_shared:
                extra_present += internal
            else:
                extra_missing += internal
        else:
            unaligned += 1
            lo = bisect.bisect_right(extra_sorted, start)
            hi = bisect.bisect_left(extra_sorted, end)
            inside = hi - lo
            if in_shared:
                extra_present += inside
            else:
                extra_missing += inside
    if extra_missing + extra_present != len(extra):
        raise RuntimeError(
            f"Extra-cut attribution failed: {extra_missing}+{extra_present}!={len(extra)}"
        )
    return {
        "dedicated_tokens": len(ded_sp),
        "shared_tokens": len(sh_sp),
        "net_extra_tokens": net,
        "extra_boundaries": len(extra),
        "crossed_dedicated_boundaries": len(saved),
        "extra_boundaries_in_missing_dedicated_tokens": extra_missing,
        "extra_boundaries_in_present_dedicated_tokens": extra_present,
        "aligned_intact_dedicated_tokens": aligned_intact,
        "aligned_fragmented_missing": aligned_frag_missing,
        "aligned_fragmented_present": aligned_frag_present,
        "unaligned_dedicated_tokens": unaligned,
        "ded_spans": ded_sp,
        "shared_spans": sh_sp,
        "extra_positions": extra,
        "split_examples": split_examples,
    }


def script_deltas(script_at, ded_spans, shared_spans):
    def starts(spans):
        counts = Counter()
        for start, _, _ in spans:
            if start >= len(script_at):
                lab = "end"
            else:
                lab = script_at[start]
            counts[lab] += 1
        return counts

    d = starts(ded_spans)
    s = starts(shared_spans)
    keys = sorted(set(d) | set(s))
    return {key: {"dedicated": d[key], "shared": s[key], "delta": s[key] - d[key]} for key in keys}


def boundary_quality(text_meta, spans, text: str):
    if len(spans) <= 1:
        return {}
    internal = [end for _, end, _ in spans[:-1]]
    n = len(internal)
    mid_utf8 = 0
    split_grapheme = 0
    before_dependent = 0
    chars = char_byte_index(text)
    # map byte -> char index
    # chars[i] = (byte_start, ch); next starts at chars[i+1]
    start_to_i = {pos: i for i, (pos, _) in enumerate(chars)}
    for pos in internal:
        if pos not in text_meta["cp_starts"]:
            mid_utf8 += 1
        elif pos not in text_meta["g_ends"]:
            split_grapheme += 1
        idx = start_to_i.get(pos)
        if idx is not None and is_dependent(chars[idx][1]):
            before_dependent += 1
    return {
        "internal_boundaries": n,
        "boundaries_inside_utf8_codepoint": mid_utf8,
        "boundaries_inside_grapheme_but_on_codepoint": split_grapheme,
        "boundaries_immediately_before_devanagari_dependent": before_dependent,
        "fraction_inside_utf8": mid_utf8 / n,
        "fraction_inside_grapheme": split_grapheme / n,
        "fraction_before_dependent": before_dependent / n,
    }


def select_examples(per_rep_words, language):
    """per_rep_words[rep] = list of dicts for each word occurrence, same order."""
    reps = list(per_rep_words)
    base = per_rep_words[reps[0]]
    # group occurrence indices by word type
    groups = defaultdict(list)
    for i, row in enumerate(base):
        groups[row["word"]].append(i)
    records = []
    for word, idxs in groups.items():
        if not has_letter(word):
            continue
        if len(idxs) < MIN_WORD_OCCURRENCES:
            continue
        ded = []
        shared = []
        for rep in reps:
            d = sum(per_rep_words[rep][i]["ded_n"] for i in idxs)
            s = sum(per_rep_words[rep][i]["shared_n"] for i in idxs)
            ded.append(d)
            shared.append(s)
        mean_d = sum(ded) / len(ded)
        mean_s = sum(shared) / len(shared)
        records.append(
            {
                "word": word,
                "occurrences": len(idxs),
                "mean_dedicated_tokens": mean_d,
                "mean_shared_tokens": mean_s,
                "mean_extra_tokens": mean_s - mean_d,
                "pct_increase": (mean_s - mean_d) / mean_d * 100 if mean_d else None,
                "idxs": idxs,
                "char_len": len(word),
            }
        )
    selected = []
    used = set()

    def add(rows, method, limit):
        added = 0
        for row in rows:
            if row["word"] in used:
                continue
            used.add(row["word"])
            selected.append((method, row))
            added += 1
            if added >= limit:
                break

    by_impact = sorted(records, key=lambda r: (-r["mean_extra_tokens"], -r["occurrences"], r["word"]))
    add(by_impact, "highest_total_extra_tokens", 8)
    short = [r for r in records if (r["mean_dedicated_tokens"] / r["occurrences"]) <= 4]
    by_rate = sorted(
        short,
        key=lambda r: (-(r["mean_extra_tokens"] / r["occurrences"]), -r["occurrences"], r["word"]),
    )
    add(by_rate, "highest_extra_tokens_per_occurrence", 6)
    positive = [r for r in records if r["pct_increase"] is not None and r["pct_increase"] > 0]
    if positive:
        med = statistics.median(r["pct_increase"] for r in positive)
        by_med = sorted(positive, key=lambda r: (abs(r["pct_increase"] - med), -r["occurrences"], r["word"]))
        add(by_med, "nearest_median_percent_increase", 4)
    examples = []
    rep0 = reps[0]
    for method, row in selected:
        idxs = row["idxs"]
        extras = [
            per_rep_words[rep0][i]["shared_n"] - per_rep_words[rep0][i]["ded_n"] for i in idxs
        ]
        med_extra = statistics.median(extras)
        chosen = min(idxs, key=lambda i: (abs(extras[idxs.index(i)] - med_extra), i))
        # idxs.index in the key is quadratic but lists are short; fix properly:
        chosen = min(idxs, key=lambda i: (abs((per_rep_words[rep0][i]["shared_n"] - per_rep_words[rep0][i]["ded_n"]) - med_extra), i))
        occ = per_rep_words[rep0][chosen]
        rep_increase = 0
        for rep in reps:
            d = sum(per_rep_words[rep][i]["ded_n"] for i in idxs)
            s = sum(per_rep_words[rep][i]["shared_n"] for i in idxs)
            if s > d:
                rep_increase += 1
        examples.append(
            {
                "language": language,
                "selection_method": method,
                "word": row["word"],
                "char_length": row["char_len"],
                "heldout_occurrences": row["occurrences"],
                "mean_dedicated_token_count": row["mean_dedicated_tokens"],
                "mean_shared_token_count": row["mean_shared_tokens"],
                "mean_dedicated_tokens_per_occurrence": row["mean_dedicated_tokens"] / row["occurrences"],
                "mean_shared_tokens_per_occurrence": row["mean_shared_tokens"] / row["occurrences"],
                "mean_extra_tokens": row["mean_extra_tokens"],
                "percent_increase": row["pct_increase"],
                "replicates_with_increase": rep_increase,
                "replicate_count": len(reps),
                "example_replicate": rep0,
                "example_dedicated_tokens": occ["ded_n"],
                "example_shared_tokens": occ["shared_n"],
                "example_percent_increase": (
                    (occ["shared_n"] - occ["ded_n"]) / occ["ded_n"] * 100 if occ["ded_n"] else None
                ),
                "dedicated_pieces": occ["ded_pieces"],
                "shared_pieces": occ["shared_pieces"],
                "median_percent_increase_among_increasing_types": (
                    statistics.median(r["pct_increase"] for r in positive) if positive else None
                ),
            }
        )
    summary = {
        "eligible_word_types": len(records),
        "word_occurrences_eligible": sum(r["occurrences"] for r in records),
    }
    return examples, records, summary


def word_occurrence_rows(text, ded_spans, ded_vocab, shared_spans, shared_vocab):
    words = word_spans(text)
    ded_ov = overlaps(words, ded_spans)
    sh_ov = overlaps(words, shared_spans)
    rows = []
    for (start, end, word), d_pieces, s_pieces in zip(words, ded_ov, sh_ov):
        rows.append(
            {
                "word": word,
                "start": start,
                "end": end,
                "ded_n": len(d_pieces),
                "shared_n": len(s_pieces),
                "ded_pieces": render_pieces(d_pieces, ded_vocab, start, end),
                "shared_pieces": render_pieces(s_pieces, shared_vocab, start, end),
            }
        )
    return rows


def fragment_summary(records_by_rep_rows):
    """records_by_rep_rows is a list of occurrence rows for one tokenization."""
    rows = records_by_rep_rows
    if not rows:
        return {}
    ded = [r["ded_n"] for r in rows]
    shared = [r["shared_n"] for r in rows]
    increases = []
    more = less = equal = 0
    for d, s in zip(ded, shared):
        if d <= 0:
            continue
        inc = (s - d) / d * 100
        increases.append(inc)
        if s > d:
            more += 1
        elif s < d:
            less += 1
        else:
            equal += 1
    increases_sorted = sorted(increases)
    def q(p):
        if not increases_sorted:
            return None
        k = (len(increases_sorted) - 1) * p
        lo = math.floor(k)
        hi = math.ceil(k)
        if lo == hi:
            return increases_sorted[lo]
        return increases_sorted[lo] * (hi - k) + increases_sorted[hi] * (k - lo)
    return {
        "word_occurrences": len(rows),
        "mean_dedicated_tokens_per_word": sum(ded) / len(ded),
        "mean_shared_tokens_per_word": sum(shared) / len(shared),
        "fraction_words_more_shared_tokens": more / len(rows),
        "fraction_words_fewer_shared_tokens": less / len(rows),
        "fraction_words_equal_tokens": equal / len(rows),
        "median_percent_increase": q(0.5),
        "p90_percent_increase": q(0.9),
        "mean_percent_increase": sum(increases) / len(increases) if increases else None,
    }


def position_hist(words_rows, extra_positions, bins=10):
    hist = [0] * bins
    inside = 0
    outside = 0
    # words sorted
    j = 0
    positions = sorted(extra_positions)
    for pos in positions:
        while j < len(words_rows) and words_rows[j]["end"] <= pos:
            j += 1
        if j < len(words_rows) and words_rows[j]["start"] < pos < words_rows[j]["end"]:
            span = words_rows[j]["end"] - words_rows[j]["start"]
            rel = (pos - words_rows[j]["start"]) / span
            b = min(bins - 1, int(rel * bins))
            hist[b] += 1
            inside += 1
        else:
            outside += 1
    return hist, inside, outside


def suffix_stats(words_rows, ded_starts, shared_starts):
    """Data-driven word-final 2- and 3-codepoint sequences."""
    acc = {}
    ded_set = set(ded_starts)
    sh_set = set(shared_starts)
    for row in words_rows:
        word = row["word"]
        if len(word) < 4 or not has_letter(word):
            continue
        offs = []
        pos = row["start"]
        for ch in word:
            offs.append(pos)
            pos += len(ch.encode("utf-8"))
        internal = offs[1:]
        if not internal:
            continue
        d_internal = sum(1 for off in internal if off in ded_set)
        s_internal = sum(1 for off in internal if off in sh_set)
        for length in (2, 3):
            if len(word) <= length:
                continue
            suf = word[-length:]
            off = offs[-length]
            key = (length, suf)
            slot = acc.setdefault(
                key,
                {
                    "n": 0,
                    "ded_hit": 0,
                    "shared_hit": 0,
                    "ded_internal_hits": 0,
                    "shared_internal_hits": 0,
                    "internal_sites": 0,
                },
            )
            slot["n"] += 1
            slot["ded_hit"] += int(off in ded_set)
            slot["shared_hit"] += int(off in sh_set)
            slot["ded_internal_hits"] += d_internal
            slot["shared_internal_hits"] += s_internal
            slot["internal_sites"] += len(internal)
    rows = []
    for (length, suf), slot in acc.items():
        if slot["n"] < SUFFIX_MIN_N:
            continue
        d_rate = slot["ded_hit"] / slot["n"]
        s_rate = slot["shared_hit"] / slot["n"]
        d_base = slot["ded_internal_hits"] / slot["internal_sites"]
        s_base = slot["shared_internal_hits"] / slot["internal_sites"]
        rows.append(
            {
                "suffix": suf,
                "suffix_codepoints": length,
                "word_occurrences": slot["n"],
                "dedicated_boundary_rate": d_rate,
                "shared_boundary_rate": s_rate,
                "dedicated_internal_boundary_rate": d_base,
                "shared_internal_boundary_rate": s_base,
                "suffix_rate_increase": s_rate - d_rate,
                "internal_rate_increase": s_base - d_base,
                "excess_over_internal_increase": (s_rate - d_rate) - (s_base - d_base),
            }
        )
    return rows


def lexicon_rows(words_rows, other_lexicon):
    bins = ((1, 2), (3, 4), (5, 8), (9, 10**9))
    acc = {}
    for row in words_rows:
        word = row["word"]
        if not has_letter(word):
            continue
        nch = len(word)
        for lo, hi in bins:
            if lo <= nch <= hi:
                label = f"{lo}-{hi if hi < 10**6 else 'plus'}"
                if hi > 10**6:
                    label = f"{lo}+"
                break
        else:
            continue
        kind = "also_in_other_language_train" if word in other_lexicon else "absent_from_other_language_train"
        key = (kind, label)
        slot = acc.setdefault(key, {"n": 0, "ded": 0, "shared": 0})
        slot["n"] += 1
        slot["ded"] += row["ded_n"]
        slot["shared"] += row["shared_n"]
    out = []
    for (kind, label), slot in sorted(acc.items()):
        out.append(
            {
                "lexicon": kind,
                "length_bin": label,
                "word_occurrences": slot["n"],
                "mean_dedicated_tokens": slot["ded"] / slot["n"],
                "mean_shared_tokens": slot["shared"] / slot["n"],
                "percent_increase": (slot["shared"] - slot["ded"]) / slot["ded"] * 100 if slot["ded"] else None,
            }
        )
    return out


def codepoint_counter(text: str) -> Counter:
    return Counter(text)


def total_variation(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    ta = sum(a.values()) or 1
    tb = sum(b.values()) or 1
    return 0.5 * sum(abs(a[k] / ta - b[k] / tb) for k in keys)


def run():
    FIG.mkdir(parents=True, exist_ok=True)
    test_text = {lang: read_text(ROOT / "data" / "processed" / lang / "test.txt") for lang in LANGUAGES}
    test_meta = {lang: annotate(test_text[lang]) for lang in LANGUAGES}
    published = {}
    token_rows = []
    alloc_rows = []
    coverage_rows = []
    pattern_rows = []
    decomp_rows = []
    frag_rows = []
    frag_summary_rows = []
    example_rows = []
    unicode_rows = []
    script_rows = []
    suffix_rows = []
    lex_rows = []
    ngram_rows = []
    validation = []
    position_store = []
    summary = {"experiments": {}, "unicode_heldout": {}, "validation": validation}

    # Held-out Unicode profile is independent of tokenizers.
    cp_counts = {lang: codepoint_counter(test_text[lang]) for lang in LANGUAGES}
    for lang in LANGUAGES:
        meta = test_meta[lang]
        base = {
            "experiment": "held_out_test",
            "replicate": "fixed",
            "language": lang,
            "condition": "corpus",
        }
        metrics = {
            "utf8_bytes": meta["n_bytes"],
            "codepoints": meta["n_chars"],
            "bytes_per_codepoint": meta["n_bytes"] / meta["n_chars"],
            "unique_codepoints": len(cp_counts[lang]),
            "graphemes": meta["n_graphemes"],
            "codepoints_per_grapheme": meta["n_chars"] / meta["n_graphemes"],
            "fraction_graphemes_with_multiple_codepoints": meta["multi_cp_graphemes"] / meta["n_graphemes"],
            "devanagari_letters": meta["dev_letters"],
        }
        for lab, n in meta["scripts"].items():
            metrics[f"codepoint_fraction_{lab}"] = n / meta["n_chars"]
        for lab, n in meta["categories"].items():
            metrics[f"category_fraction_{lab}"] = n / meta["n_chars"]
        for lab, n in meta["dependents"].items():
            metrics[f"dependents_per_1000_devanagari_letters_{lab}"] = (
                1000 * n / meta["dev_letters"] if meta["dev_letters"] else None
            )
        for length, n in sorted(meta["grapheme_cp"].items()):
            if length <= 4:
                metrics[f"fraction_graphemes_with_{length}_codepoints"] = n / meta["n_graphemes"]
        runs = devanagari_runs(test_text[lang])
        dev_bytes = b"".join(runs)
        tri = ngram_table_runs(runs, 3)
        bi = ngram_table_runs(runs, 6)
        metrics["devanagari_run_bytes"] = len(dev_bytes)
        metrics["devanagari_byte_trigram_entropy_bits"] = shannon(tri)
        metrics["devanagari_byte_6gram_entropy_bits"] = shannon(bi)
        metrics["devanagari_byte_trigram_types"] = len(tri)
        metrics["devanagari_byte_trigram_types_per_1000_bytes"] = (
            1000 * len(tri) / len(dev_bytes) if dev_bytes else None
        )
        for key, value in metrics.items():
            unicode_rows.append({**base, "metric": key, "value": fnum(value), "comparison_language": "", "codepoint": "", "detail": ""})
        summary["unicode_heldout"][lang] = metrics

    for a, b in (("hindi", "nepali"), ("english", "hindi"), ("english", "nepali")):
        tv = total_variation(cp_counts[a], cp_counts[b])
        unicode_rows.append(
            {
                "experiment": "held_out_test",
                "replicate": "fixed",
                "language": a,
                "condition": "corpus",
                "metric": "codepoint_total_variation_distance",
                "value": fnum(tv),
                "comparison_language": b,
                "codepoint": "",
                "detail": "",
            }
        )
    # Largest Hindi/Nepali codepoint rate gaps.
    ha = cp_counts["hindi"]
    nb = cp_counts["nepali"]
    ta = sum(ha.values())
    tb = sum(nb.values())
    gaps = []
    for ch in set(ha) | set(nb):
        if not ("\u0900" <= ch <= "\u097f"):
            continue
        gaps.append((abs(ha[ch] / ta - nb[ch] / tb), ch, ha[ch] / ta, nb[ch] / tb))
    gaps.sort(reverse=True)
    for abs_gap, ch, pa, pb in gaps[:25]:
        unicode_rows.append(
            {
                "experiment": "held_out_test",
                "replicate": "fixed",
                "language": "hindi_vs_nepali",
                "condition": "corpus",
                "metric": "absolute_codepoint_rate_gap",
                "value": fnum(abs_gap),
                "comparison_language": "nepali_minus_context",
                "codepoint": f"U+{ord(ch):04X}",
                "detail": f"{ch} hindi_rate={pa:.6f} nepali_rate={pb:.6f} name={unicodedata.name(ch, '')}",
            }
        )
    summary["hindi_nepali_top_codepoint_gaps"] = [
        {
            "character": ch,
            "codepoint": f"U+{ord(ch):04X}",
            "name": unicodedata.name(ch, ""),
            "abs_rate_gap": abs_gap,
            "hindi_rate": pa,
            "nepali_rate": pb,
        }
        for abs_gap, ch, pa, pb in gaps[:15]
    ]

    for exp_name, exp_cfg in EXPERIMENTS.items():
        print(f"\n=== {exp_name} ===", flush=True)
        if not exp_cfg["results"].exists():
            raise FileNotFoundError(f"Missing results artifact: {exp_cfg['results']}")
        saved = json.loads(exp_cfg["results"].read_text(encoding="utf-8"))
        published[exp_name] = saved
        exp_summary = {"replicates": {}}
        # examples only for the primary regime, after all replicates are encoded
        primary_word_rows = {lang: {} for lang in LANGUAGES} if exp_name == "repeated_vocabulary_sharing_v1" else None
        suffix_acc = []
        lex_acc = []

        for rep in REPLICATES:
            print(f"  {rep}", flush=True)
            block = saved["replicates"][rep]
            models = {}
            for key, cond in COND.items():
                rel = block["models"][cond]["model_file"]
                models[key] = load_tokenizer(ROOT / rel)
            trains = {}
            for lang in LANGUAGES:
                # Training path is recorded by hash in the results; use the manifest paths
                # already embedded only as hashes. The experiment scripts use fixed paths.
                if exp_name == "repeated_vocabulary_sharing_v1":
                    train_path = ROOT / "data" / "controlled_multilingual_replicates" / rep / f"{lang}_train.txt"
                else:
                    train_path = ROOT / "data" / "total_budget_matched_replicates" / rep / f"{lang}_train.txt"
                trains[lang] = read_text(train_path)
                recorded = block["training_data"][lang]["sha256"]
                import hashlib
                digest = hashlib.sha256(trains[lang].encode("utf-8")).hexdigest()
                if digest != recorded:
                    raise RuntimeError(f"Training file hash mismatch for {exp_name} {rep} {lang}")

            shared = models["shared"]
            shared_bytes = set(shared.vocab.values())
            shared_rules, shared_produced, _ = merge_records(shared)
            ids = {}
            # Encode tests for every saved condition so published counts can be checked,
            # and encode each language's own training text with the shared model and
            # with that language's dedicated model.
            for key in ("english", "hindi", "nepali", "shared"):
                ids[key] = {}
                for lang in LANGUAGES:
                    data = test_text[lang].encode("utf-8")
                    got = encode_bytes(data, models[key].merges)
                    expect = block["models"][COND[key]]["test"][lang]["metrics"]["tokens"]
                    if len(got) != expect:
                        raise RuntimeError(
                            f"Token-count mismatch {exp_name} {rep} {key} on {lang}: {len(got)} != {expect}"
                        )
                    ids[key][lang] = got
                    validation.append(
                        {
                            "experiment": exp_name,
                            "replicate": rep,
                            "model": key,
                            "text": f"test:{lang}",
                            "tokens": len(got),
                            "published_tokens": expect,
                            "match": True,
                        }
                    )
            train_ids_shared = {}
            train_ids_ded = {}
            for lang in LANGUAGES:
                raw = trains[lang].encode("utf-8")
                train_ids_shared[lang] = encode_bytes(raw, shared.merges)
                train_ids_ded[lang] = encode_bytes(raw, models[lang].merges)

            # Shared vocabulary allocation.
            train_emit = {lang: Counter(train_ids_shared[lang]) for lang in LANGUAGES}
            test_emit = {lang: Counter(ids["shared"][lang]) for lang in LANGUAGES}
            class_counts = Counter()
            class_train = Counter()
            class_test = Counter()
            raw_counts = Counter()
            struct_counts = Counter()
            for (pair, idx) in shared.merges.items():
                blob = shared.vocab[idx]
                tr = {lang: train_emit[lang][idx] for lang in LANGUAGES}
                te = {lang: test_emit[lang][idx] for lang in LANGUAGES}
                raw = {lang: trains[lang].encode("utf-8").count(blob) for lang in LANGUAGES}
                cls = allocation_class(tr, PREDOMINANT)
                rcls = raw_class(raw)
                scls = structural_class(blob)
                text, ok = utf8_text(blob)
                class_counts[cls] += 1
                class_train[cls] += sum(tr.values())
                class_test[cls] += sum(te.values())
                raw_counts[rcls] += 1
                struct_counts[scls] += 1
                token_rows.append(
                    {
                        "experiment": exp_name,
                        "replicate": rep,
                        "token_id": idx,
                        "merge_rank": idx - 256,
                        "byte_length": len(blob),
                        "hex": blob.hex(),
                        "valid_utf8": int(ok),
                        "text": text,
                        "structural_class": scls,
                        "train_emit_english": tr["english"],
                        "train_emit_hindi": tr["hindi"],
                        "train_emit_nepali": tr["nepali"],
                        "test_emit_english": te["english"],
                        "test_emit_hindi": te["hindi"],
                        "test_emit_nepali": te["nepali"],
                        "raw_train_english": raw["english"],
                        "raw_train_hindi": raw["hindi"],
                        "raw_train_nepali": raw["nepali"],
                        "train_emission_class": cls,
                        "raw_presence_class": rcls,
                    }
                )
            learned_n = len(shared.merges)
            for cls, n in sorted(class_counts.items()):
                alloc_rows.append(
                    {
                        "experiment": exp_name,
                        "replicate": rep,
                        "basis": f"train_emission_share_{PREDOMINANT:.2f}",
                        "category": cls,
                        "token_count": n,
                        "fraction_of_learned_tokens": n / learned_n,
                        "train_emissions": class_train[cls],
                        "test_emissions": class_test[cls],
                    }
                )
            for cls, n in sorted(raw_counts.items()):
                alloc_rows.append(
                    {
                        "experiment": exp_name,
                        "replicate": rep,
                        "basis": "raw_training_byte_presence",
                        "category": cls,
                        "token_count": n,
                        "fraction_of_learned_tokens": n / learned_n,
                        "train_emissions": "",
                        "test_emissions": "",
                    }
                )
            for cls, n in sorted(struct_counts.items()):
                alloc_rows.append(
                    {
                        "experiment": exp_name,
                        "replicate": rep,
                        "basis": "byte_structure",
                        "category": cls,
                        "token_count": n,
                        "fraction_of_learned_tokens": n / learned_n,
                        "train_emissions": "",
                        "test_emissions": "",
                    }
                )

            rep_cov = {}
            rep_decomp = {}
            for lang in LANGUAGES:
                ded = models[lang]
                ded_rules, ded_produced, ded_order = merge_records(ded)
                rules_hit = sum(1 for rule in ded_rules if rule in shared_rules)
                types_hit = sum(1 for blob in ded_produced if blob in shared_bytes)
                test_counts = Counter(ids[lang][lang])
                learned_emissions = []
                for (pair, idx) in ded.merges.items():
                    blob = ded.vocab[idx]
                    learned_emissions.append((test_counts[idx], blob, idx))
                learned_emissions.sort(reverse=True)
                emis_total = sum(c for c, _, _ in learned_emissions)
                emis_hit = sum(c for c, blob, _ in learned_emissions if blob in shared_bytes)
                def top_cov(k):
                    top = learned_emissions[:k]
                    if not top:
                        return None, None
                    return (
                        sum(1 for _, blob, _ in top if blob in shared_bytes) / len(top),
                        sum(c for c, blob, _ in top if blob in shared_bytes) / sum(c for c, _, _ in top),
                    )
                t50, m50 = top_cov(50)
                t100, m100 = top_cov(100)
                t200, m200 = top_cov(200)
                present_ranks = [shared_produced[blob] - 256 for _, blob, _ in learned_emissions if blob in shared_produced]
                top100_present_ranks = [
                    shared_produced[blob] - 256
                    for _, blob, _ in learned_emissions[:100]
                    if blob in shared_produced
                ]
                pair = analyze_pair(
                    test_text[lang],
                    ids[lang][lang],
                    ded.vocab,
                    ids["shared"][lang],
                    shared.vocab,
                    shared_bytes,
                )
                published_penalty = (
                    block["models"]["shared_multilingual"]["test"][lang]["metrics"]["tokens"]
                    / block["models"][COND[lang]]["test"][lang]["metrics"]["tokens"]
                    - 1
                ) * 100
                coverage_rows.append(
                    {
                        "experiment": exp_name,
                        "replicate": rep,
                        "language": lang,
                        "dedicated_learned_types": len(ded_produced),
                        "dedicated_types_in_shared_vocab": types_hit,
                        "type_coverage": types_hit / len(ded_produced),
                        "dedicated_rules": len(ded_rules),
                        "rules_with_same_child_bytes_in_shared": rules_hit,
                        "rule_coverage": rules_hit / len(ded_rules),
                        "test_learned_emissions": emis_total,
                        "test_learned_emissions_in_shared": emis_hit,
                        "emission_coverage": emis_hit / emis_total if emis_total else None,
                        "top50_type_coverage": t50,
                        "top100_type_coverage": t100,
                        "top200_type_coverage": t200,
                        "top50_emission_coverage": m50,
                        "top100_emission_coverage": m100,
                        "top200_emission_coverage": m200,
                        "median_shared_rank_of_overlapping_types": (
                            statistics.median(present_ranks) if present_ranks else None
                        ),
                        "median_shared_rank_of_present_top100": (
                            statistics.median(top100_present_ranks) if top100_present_ranks else None
                        ),
                        "recomputed_penalty_percent": (
                            pair["shared_tokens"] / pair["dedicated_tokens"] - 1
                        ) * 100,
                        "published_penalty_percent": published_penalty,
                    }
                )
                rep_cov[lang] = coverage_rows[-1]
                decomp = {
                    "experiment": exp_name,
                    "replicate": rep,
                    "language": lang,
                    "dedicated_tokens": pair["dedicated_tokens"],
                    "shared_tokens": pair["shared_tokens"],
                    "net_extra_tokens": pair["net_extra_tokens"],
                    "extra_boundaries": pair["extra_boundaries"],
                    "boundaries_where_shared_crossed_dedicated_boundary": pair["crossed_dedicated_boundaries"],
                    "extra_boundaries_inside_dedicated_tokens_absent_from_shared": pair[
                        "extra_boundaries_in_missing_dedicated_tokens"
                    ],
                    "extra_boundaries_inside_dedicated_tokens_present_in_shared": pair[
                        "extra_boundaries_in_present_dedicated_tokens"
                    ],
                    "aligned_intact_dedicated_tokens": pair["aligned_intact_dedicated_tokens"],
                    "aligned_fragmented_and_absent": pair["aligned_fragmented_missing"],
                    "aligned_fragmented_but_present": pair["aligned_fragmented_present"],
                    "unaligned_dedicated_tokens": pair["unaligned_dedicated_tokens"],
                }
                decomp_rows.append(decomp)
                rep_decomp[lang] = decomp

                # High-frequency missing patterns and high-frequency present-but-split patterns.
                train_counts = Counter(train_ids_ded[lang])
                raw_test = test_text[lang].encode("utf-8")
                missing_ranked = [(c, blob, idx) for c, blob, idx in learned_emissions if blob not in shared_bytes]
                for rank, (count, blob, idx) in enumerate(missing_ranked[:15], start=1):
                    modal, modal_n = ("", 0)
                    if pair["split_examples"][blob]:
                        modal, modal_n = pair["split_examples"][blob].most_common(1)[0]
                    isolated = encode_bytes(blob, shared.merges)
                    iso = " | ".join(piece_label(shared.vocab[i]) for i in isolated)
                    text, ok = utf8_text(blob)
                    pattern_rows.append(
                        {
                            "experiment": exp_name,
                            "replicate": rep,
                            "language": lang,
                            "status": "absent_from_shared_vocab",
                            "rank_by_dedicated_test_emissions": rank,
                            "dedicated_test_emissions": count,
                            "dedicated_train_emissions": train_counts[idx],
                            "raw_nonoverlapping_test_count": raw_test.count(blob),
                            "byte_length": len(blob),
                            "valid_utf8": int(ok),
                            "text": gloss_bytes(blob),
                            "modal_in_context_shared_split": modal,
                            "modal_split_count": modal_n,
                            "isolated_shared_encoding": iso,
                        }
                    )
                # present tokens that still contribute the most in-context extra pieces
                present_frag = []
                for count, blob, idx in learned_emissions:
                    if blob not in shared_bytes:
                        continue
                    splits = pair["split_examples"].get(blob)
                    if not splits:
                        continue
                    frag_occs = sum(splits.values())
                    present_frag.append((frag_occs, count, blob, idx, splits))
                present_frag.sort(reverse=True)
                for rank, (frag_occs, count, blob, idx, splits) in enumerate(present_frag[:8], start=1):
                    modal, modal_n = splits.most_common(1)[0]
                    text, ok = utf8_text(blob)
                    pattern_rows.append(
                        {
                            "experiment": exp_name,
                            "replicate": rep,
                            "language": lang,
                            "status": "present_in_shared_vocab_but_often_split",
                            "rank_by_dedicated_test_emissions": rank,
                            "dedicated_test_emissions": count,
                            "dedicated_train_emissions": train_counts[idx],
                            "raw_nonoverlapping_test_count": raw_test.count(blob),
                            "byte_length": len(blob),
                            "valid_utf8": int(ok),
                            "text": gloss_bytes(blob),
                            "modal_in_context_shared_split": modal,
                            "modal_split_count": modal_n,
                            "isolated_shared_encoding": "",
                        }
                    )

                # Script contribution and boundary quality.
                deltas = script_deltas(test_meta[lang]["script_at"], pair["ded_spans"], pair["shared_spans"])
                for lab, vals in deltas.items():
                    script_rows.append(
                        {
                            "experiment": exp_name,
                            "replicate": rep,
                            "language": lang,
                            "script_at_token_start": lab,
                            "dedicated_tokens": vals["dedicated"],
                            "shared_tokens": vals["shared"],
                            "delta_tokens": vals["delta"],
                        }
                    )
                for cond_name, spans in (("dedicated", pair["ded_spans"]), ("shared", pair["shared_spans"])):
                    qual = boundary_quality(test_meta[lang], spans, test_text[lang])
                    for metric, value in qual.items():
                        unicode_rows.append(
                            {
                                "experiment": exp_name,
                                "replicate": rep,
                                "language": lang,
                                "condition": cond_name,
                                "metric": metric,
                                "value": fnum(value),
                                "comparison_language": "",
                                "codepoint": "",
                                "detail": "",
                            }
                        )

                # Words, suffixes, lexicon.
                wrows = word_occurrence_rows(
                    test_text[lang], pair["ded_spans"], ded.vocab, pair["shared_spans"], shared.vocab
                )
                sm = fragment_summary(wrows)
                sm.update({"experiment": exp_name, "replicate": rep, "language": lang})
                frag_summary_rows.append(sm)
                hist, inside, outside = position_hist(wrows, pair["extra_positions"])
                position_store.append(
                    {
                        "experiment": exp_name,
                        "replicate": rep,
                        "language": lang,
                        "hist": hist,
                        "inside": inside,
                        "outside": outside,
                    }
                )
                for b, n in enumerate(hist, start=1):
                    frag_summary_rows[-1][f"extra_boundary_bin_{b}"] = n
                frag_summary_rows[-1]["extra_boundaries_inside_words"] = inside
                frag_summary_rows[-1]["extra_boundaries_outside_words"] = outside
                suf = suffix_stats(
                    wrows,
                    [s for s, _, _ in pair["ded_spans"]],
                    [s for s, _, _ in pair["shared_spans"]],
                )
                for row in suf:
                    suffix_rows.append({"experiment": exp_name, "replicate": rep, "language": lang, **row})
                other = "hindi" if lang == "nepali" else "nepali" if lang == "hindi" else None
                if other:
                    lexicon = set()
                    for token in trains[other].split():
                        if has_letter(token):
                            lexicon.add(token)
                    for row in lexicon_rows(wrows, lexicon):
                        lex_rows.append({"experiment": exp_name, "replicate": rep, "language": lang, **row})

                # Frequent byte-pattern coverage on this replicate's training text.
                train_bytes = trains[lang].encode("utf-8")
                runs = devanagari_runs(trains[lang])
                for n in NGRAM_NS:
                    full_counts = ngram_table(train_bytes, n)
                    for vocab_name, vocab_bytes in (
                        ("dedicated", set(ded.vocab.values())),
                        ("shared", shared_bytes),
                    ):
                        stats = coverage_stats(full_counts, vocab_bytes, NGRAM_MIN_COUNT, 200)
                        ngram_rows.append(
                            {
                                "experiment": exp_name,
                                "replicate": rep,
                                "language": lang,
                                "domain": "all_training_bytes",
                                "n": n,
                                "vocabulary": vocab_name,
                                **{k: fnum(v) if isinstance(v, float) else v for k, v in stats.items()},
                            }
                        )
                    if lang != "english":
                        run_counts = ngram_table_runs(runs, n)
                        for vocab_name, vocab_bytes in (
                            ("dedicated", set(ded.vocab.values())),
                            ("shared", shared_bytes),
                        ):
                            stats = coverage_stats(run_counts, vocab_bytes, NGRAM_MIN_COUNT, 200)
                            ngram_rows.append(
                                {
                                    "experiment": exp_name,
                                    "replicate": rep,
                                    "language": lang,
                                    "domain": "devanagari_runs_only",
                                    "n": n,
                                    "vocabulary": vocab_name,
                                    **{k: fnum(v) if isinstance(v, float) else v for k, v in stats.items()},
                                }
                            )
                if primary_word_rows is not None:
                    primary_word_rows[lang][rep] = wrows

            exp_summary["replicates"][rep] = {"coverage": rep_cov, "decomposition": rep_decomp}
            # end replicate

        if primary_word_rows is not None:
            for lang in LANGUAGES:
                examples, records, info = select_examples(primary_word_rows[lang], lang)
                for ex in examples:
                    example_rows.append({"experiment": exp_name, **ex})
                exp_summary["example_selection"] = exp_summary.get("example_selection", {})
                exp_summary["example_selection"][lang] = {
                    "eligible_word_types": info["eligible_word_types"],
                    "eligible_occurrences": info["word_occurrences_eligible"],
                    "median_percent_increase": (
                        statistics.median(
                            [r["pct_increase"] for r in records if r["pct_increase"] and r["pct_increase"] > 0]
                        )
                        if any(r["pct_increase"] and r["pct_increase"] > 0 for r in records)
                        else None
                    ),
                }
        summary["experiments"][exp_name] = {
            "label": exp_cfg["label"],
            "example_selection": exp_summary.get("example_selection", {}),
        }

    # Figures from replicate means of the primary and matched regimes.
    def mean_alloc(exp, category):
        vals = [
            row["token_count"]
            for row in alloc_rows
            if row["experiment"] == exp
            and row["basis"].startswith("train_emission")
            and row["category"] == category
        ]
        return sum(vals) / len(vals) if vals else 0

    classes = [
        "exclusive_english",
        "predominant_english",
        "exclusive_hindi",
        "predominant_hindi",
        "exclusive_nepali",
        "predominant_nepali",
        "cross_lingual",
        "unused",
    ]
    # Collapse exclusive+predominant for the figure.
    def collapsed(exp, lang):
        return mean_alloc(exp, f"exclusive_{lang}") + mean_alloc(exp, f"predominant_{lang}")

    write_grouped_svg(
        FIG / "shared_vocab_allocation.svg",
        "Shared learned tokens by training-emission language",
        "Mean of three replicates. A token is assigned to a language at ≥80% of its training emissions.",
        ["English", "Hindi", "Nepali", "Cross-lingual", "Unused"],
        [
            (
                "About 900 KB shared",
                "#0072B2",
                [
                    collapsed("repeated_vocabulary_sharing_v1", "english"),
                    collapsed("repeated_vocabulary_sharing_v1", "hindi"),
                    collapsed("repeated_vocabulary_sharing_v1", "nepali"),
                    mean_alloc("repeated_vocabulary_sharing_v1", "cross_lingual"),
                    mean_alloc("repeated_vocabulary_sharing_v1", "unused"),
                ],
            ),
            (
                "About 300 KB shared",
                "#E69F00",
                [
                    collapsed("total_budget_matched_v1", "english"),
                    collapsed("total_budget_matched_v1", "hindi"),
                    collapsed("total_budget_matched_v1", "nepali"),
                    mean_alloc("total_budget_matched_v1", "cross_lingual"),
                    mean_alloc("total_budget_matched_v1", "unused"),
                ],
            ),
        ],
        "Learned tokens (count)",
        "Source: saved vocab-1024 models. Emission counts are from each replicate's own training files.",
    )

    def mean_cov(exp, lang, field):
        vals = [
            row[field]
            for row in coverage_rows
            if row["experiment"] == exp and row["language"] == lang and row[field] is not None
        ]
        return 100 * sum(vals) / len(vals) if vals else 0

    write_grouped_svg(
        FIG / "dedicated_merge_coverage.svg",
        "Dedicated learned tokens retained by the shared vocabulary",
        "Mean of three replicates. Retention is exact byte-string membership, weighted by held-out emissions.",
        ["English", "Hindi", "Nepali"],
        [
            ("900 KB type coverage", "#0072B2", [mean_cov("repeated_vocabulary_sharing_v1", lang, "type_coverage") for lang in LANGUAGES]),
            ("900 KB emission coverage", "#56B4E9", [mean_cov("repeated_vocabulary_sharing_v1", lang, "emission_coverage") for lang in LANGUAGES]),
            ("300 KB type coverage", "#E69F00", [mean_cov("total_budget_matched_v1", lang, "type_coverage") for lang in LANGUAGES]),
            ("300 KB emission coverage", "#D55E00", [mean_cov("total_budget_matched_v1", lang, "emission_coverage") for lang in LANGUAGES]),
        ],
        "Coverage of dedicated learned tokens (%)",
        "Source: dedicated vs shared vocabularies at size 1024. Base bytes 0–255 are excluded.",
    )

    def mean_decomp(exp, lang, field):
        vals = [row[field] for row in decomp_rows if row["experiment"] == exp and row["language"] == lang]
        return sum(vals) / len(vals) if vals else 0

    write_grouped_svg(
        FIG / "penalty_boundary_accounting.svg",
        "Where the extra shared-tokenizer boundaries fall",
        "Mean of three replicates on the fixed held-out tests. Net extra tokens equal extra cuts minus crossed dedicated boundaries.",
        ["English", "Hindi", "Nepali"],
        [
            (
                "Cuts in tokens absent from shared vocab",
                "#D55E00",
                [mean_decomp("repeated_vocabulary_sharing_v1", lang, "extra_boundaries_inside_dedicated_tokens_absent_from_shared") for lang in LANGUAGES],
            ),
            (
                "Cuts in tokens present in shared vocab",
                "#0072B2",
                [mean_decomp("repeated_vocabulary_sharing_v1", lang, "extra_boundaries_inside_dedicated_tokens_present_in_shared") for lang in LANGUAGES],
            ),
            (
                "Net extra tokens",
                "#009E73",
                [mean_decomp("repeated_vocabulary_sharing_v1", lang, "net_extra_tokens") for lang in LANGUAGES],
            ),
        ],
        "Token boundaries (count)",
        "Source: repeated_vocabulary_sharing_v1, vocabulary 1024, fixed held-out tests.",
    )

    def mean_delta(exp, lang, script):
        vals = [
            row["delta_tokens"]
            for row in script_rows
            if row["experiment"] == exp and row["language"] == lang and row["script_at_token_start"] == script
        ]
        return sum(vals) / len(vals) if vals else 0

    write_grouped_svg(
        FIG / "script_contribution_to_extra_tokens.svg",
        "Extra shared tokens by the script at the token start",
        "Mean token-count difference (shared minus dedicated) across three replicates. Parts sum to the net penalty.",
        ["English", "Hindi", "Nepali"],
        [
            ("Devanagari", "#009E73", [mean_delta("repeated_vocabulary_sharing_v1", lang, "devanagari") for lang in LANGUAGES]),
            ("Latin letters", "#0072B2", [mean_delta("repeated_vocabulary_sharing_v1", lang, "latin") for lang in LANGUAGES]),
            ("Whitespace", "#999999", [mean_delta("repeated_vocabulary_sharing_v1", lang, "whitespace") for lang in LANGUAGES]),
            ("Other ASCII", "#E69F00", [mean_delta("repeated_vocabulary_sharing_v1", lang, "ascii_other") for lang in LANGUAGES]),
        ],
        "Extra tokens (shared − dedicated)",
        "Source: repeated_vocabulary_sharing_v1 on fixed held-out tests. Script is the codepoint containing the token's first byte.",
    )

    # Position histograms, mean fraction inside words.
    xlabels = [f"{i*10}-{(i+1)*10}%" for i in range(10)]
    series = []
    colors = {"english": "#0072B2", "hindi": "#E69F00", "nepali": "#009E73"}
    for lang in LANGUAGES:
        hist = [0] * 10
        n = 0
        for row in position_store:
            if row["experiment"] == "repeated_vocabulary_sharing_v1" and row["language"] == lang:
                for i, v in enumerate(row["hist"]):
                    hist[i] += v
                n += 1
        total = sum(hist) or 1
        series.append((lang.capitalize(), colors[lang], [100 * v / total for v in hist]))
    write_line_svg(
        FIG / "extra_boundary_position.svg",
        "Position of extra shared boundaries inside held-out words",
        "Mean distribution across three replicates. 0% is the start of a whitespace-delimited word.",
        xlabels,
        series,
        "Share of in-word extra boundaries (%)",
        "Source: repeated_vocabulary_sharing_v1. Extra boundaries are shared cuts that the dedicated tokenizer does not make.",
    )

    write_grouped_svg(
        FIG / "unicode_profile_heldout.svg",
        "Held-out orthography is similar for Hindi and Nepali",
        "Fixed test sets. Devanagari rates are per 1,000 Devanagari letters. Bytes per codepoint use the full test text.",
        ["Bytes / codepoint", "Devanagari % of codepoints", "Vowel signs / 1,000 letters", "Virama / 1,000 letters"],
        [
            (
                "English",
                "#0072B2",
                [
                    summary["unicode_heldout"]["english"]["bytes_per_codepoint"],
                    100 * summary["unicode_heldout"]["english"].get("codepoint_fraction_devanagari", 0),
                    summary["unicode_heldout"]["english"].get("dependents_per_1000_devanagari_letters_vowel_sign") or 0,
                    summary["unicode_heldout"]["english"].get("dependents_per_1000_devanagari_letters_virama") or 0,
                ],
            ),
            (
                "Hindi",
                "#E69F00",
                [
                    summary["unicode_heldout"]["hindi"]["bytes_per_codepoint"],
                    100 * summary["unicode_heldout"]["hindi"].get("codepoint_fraction_devanagari", 0),
                    summary["unicode_heldout"]["hindi"].get("dependents_per_1000_devanagari_letters_vowel_sign") or 0,
                    summary["unicode_heldout"]["hindi"].get("dependents_per_1000_devanagari_letters_virama") or 0,
                ],
            ),
            (
                "Nepali",
                "#009E73",
                [
                    summary["unicode_heldout"]["nepali"]["bytes_per_codepoint"],
                    100 * summary["unicode_heldout"]["nepali"].get("codepoint_fraction_devanagari", 0),
                    summary["unicode_heldout"]["nepali"].get("dependents_per_1000_devanagari_letters_vowel_sign") or 0,
                    summary["unicode_heldout"]["nepali"].get("dependents_per_1000_devanagari_letters_virama") or 0,
                ],
            ),
        ],
        "Mixed units (see category labels)",
        "Source: data/processed/{language}/test.txt. This figure mixes units and is only a profile, not a single scale.",
    )

    # The mixed-unit figure is a bad scientific figure. Replace it with two clean figures.
    write_grouped_svg(
        FIG / "heldout_script_composition.svg",
        "Script composition of the fixed held-out tests",
        "Share of Unicode codepoints. English, Hindi, and Nepali tests are the sets used for the published penalties.",
        ["English", "Hindi", "Nepali"],
        [
            ("Devanagari", "#009E73", [100 * summary["unicode_heldout"][lang].get("codepoint_fraction_devanagari", 0) for lang in LANGUAGES]),
            ("Latin letters", "#0072B2", [100 * summary["unicode_heldout"][lang].get("codepoint_fraction_latin", 0) for lang in LANGUAGES]),
            ("Whitespace", "#999999", [100 * summary["unicode_heldout"][lang].get("codepoint_fraction_whitespace", 0) for lang in LANGUAGES]),
            ("Other ASCII", "#E69F00", [100 * summary["unicode_heldout"][lang].get("codepoint_fraction_ascii_other", 0) for lang in LANGUAGES]),
            ("Other", "#CC79A7", [100 * summary["unicode_heldout"][lang].get("codepoint_fraction_other", 0) for lang in LANGUAGES]),
        ],
        "Codepoints (%)",
        "Source: data/processed/{english,hindi,nepali}/test.txt.",
    )
    write_grouped_svg(
        FIG / "devanagari_dependents_heldout.svg",
        "Devanagari dependent signs per 1,000 letters",
        "Held-out tests. Letters are Devanagari category Lo. English has essentially no Devanagari letters.",
        ["Vowel sign", "Virama", "Anusvara", "Candrabindu", "Nukta", "Visarga"],
        [
            (
                "Hindi",
                "#E69F00",
                [
                    summary["unicode_heldout"]["hindi"].get(f"dependents_per_1000_devanagari_letters_{name}") or 0
                    for name in ("vowel_sign", "virama", "anusvara", "candrabindu", "nukta", "visarga")
                ],
            ),
            (
                "Nepali",
                "#009E73",
                [
                    summary["unicode_heldout"]["nepali"].get(f"dependents_per_1000_devanagari_letters_{name}") or 0
                    for name in ("vowel_sign", "virama", "anusvara", "candrabindu", "nukta", "visarga")
                ],
            ),
        ],
        "Signs per 1,000 Devanagari letters",
        "Source: fixed held-out tests. A higher rate is an orthographic difference, not by itself a tokenization penalty.",
    )
    # Remove the mixed-unit figure so it is not cited by mistake.
    mixed = FIG / "unicode_profile_heldout.svg"
    if mixed.exists():
        mixed.unlink()

    # Write tables.
    write_csv(
        OUT / "vocabulary_allocation.csv",
        [
            {
                **row,
                "fraction_of_learned_tokens": fnum(row["fraction_of_learned_tokens"]),
            }
            for row in alloc_rows
        ],
        ["experiment", "replicate", "basis", "category", "token_count", "fraction_of_learned_tokens", "train_emissions", "test_emissions"],
    )
    write_csv(
        OUT / "vocabulary_allocation_tokens.csv",
        token_rows,
        list(token_rows[0].keys()) if token_rows else ["experiment"],
    )
    cov_fields = list(coverage_rows[0].keys())
    write_csv(
        OUT / "merge_coverage.csv",
        [{k: fnum(v) if isinstance(v, float) else v for k, v in row.items()} for row in coverage_rows],
        cov_fields,
    )
    write_csv(
        OUT / "merge_coverage_patterns.csv",
        pattern_rows,
        list(pattern_rows[0].keys()) if pattern_rows else ["experiment"],
    )
    write_csv(
        OUT / "fragmentation_examples.csv",
        [{k: fnum(v) if isinstance(v, float) else v for k, v in row.items()} for row in example_rows],
        list(example_rows[0].keys()) if example_rows else ["language"],
    )
    # fragmentation summary has dynamic bin keys
    frag_fields = []
    for row in frag_summary_rows:
        for key in row:
            if key not in frag_fields:
                frag_fields.append(key)
    write_csv(
        OUT / "fragmentation_summary.csv",
        [{k: fnum(v) if isinstance(v, float) else v for k, v in row.items()} for row in frag_summary_rows],
        frag_fields,
    )
    write_csv(
        OUT / "unicode_analysis.csv",
        unicode_rows,
        ["experiment", "replicate", "language", "condition", "metric", "value", "comparison_language", "codepoint", "detail"],
    )
    write_csv(
        OUT / "penalty_decomposition.csv",
        decomp_rows,
        list(decomp_rows[0].keys()),
    )
    write_csv(
        OUT / "script_penalty.csv",
        script_rows,
        list(script_rows[0].keys()),
    )
    write_csv(
        OUT / "suffix_boundaries.csv",
        [{k: fnum(v) if isinstance(v, float) else v for k, v in row.items()} for row in suffix_rows],
        list(suffix_rows[0].keys()) if suffix_rows else ["experiment"],
    )
    write_csv(
        OUT / "lexicon_fragmentation.csv",
        [{k: fnum(v) if isinstance(v, float) else v for k, v in row.items()} for row in lex_rows],
        list(lex_rows[0].keys()) if lex_rows else ["experiment"],
    )
    write_csv(
        OUT / "ngram_coverage.csv",
        ngram_rows,
        list(ngram_rows[0].keys()) if ngram_rows else ["experiment"],
    )
    summary["validation_pairs"] = len(validation)
    summary["method"] = {
        "predominant_threshold": PREDOMINANT,
        "token_identity": "exact UTF-8 byte string",
        "emission_class": "share of shared-tokenizer emissions on the three training files of that replicate",
        "rule_match": "same ordered pair of child byte strings",
        "type_match": "same resulting byte string in the vocabulary",
        "boundary_identity": "net extra tokens = shared boundaries absent from dedicated segmentation minus dedicated boundaries absent from shared segmentation",
        "word_examples": "repeated_vocabulary_sharing_v1 only; whitespace-delimited types with >=5 held-out occurrences and a letter; ranked by mean extra overlapping tokens across three replicates; median band is the types nearest the median percent increase among types that increase",
        "example_segmentation": "replicate_1 occurrence whose extra token count is closest to the median extra for that type",
        "suffixes": "word-final 2- and 3-codepoint strings on words of at least 4 codepoints with a letter; minimum 40 occurrences; not a morphological parse",
        "lexicon": "exact whitespace word match against the other Indic language's training file in the same replicate",
        "ngram_min_count": NGRAM_MIN_COUNT,
    }
    (OUT / "measured_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validation pairs matched: {len(validation)}", flush=True)
    print(f"Wrote tables and figures under {OUT}", flush=True)


if __name__ == "__main__":
    run()
