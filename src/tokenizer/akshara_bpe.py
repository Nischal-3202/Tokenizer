
from collections import Counter

from .aksharas import split_aksharas

BYTE_FALLBACK_BASE = 1_000_000


def get_stats(ids):
    counts = Counter(zip(ids, ids[1:]))
    return dict(counts)


def merge(ids, pair, new_id):
    result = []
    i = 0

    while i < len(ids):
        if (
            i + 1 < len(ids)
            and ids[i] == pair[0]
            and ids[i + 1] == pair[1]
        ):
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1

    return result


class AksharaBPE:
    """BPE over heuristic Devanagari akshara units."""

    def __init__(self):
        self.base_tokens = []
        self.base_to_id = {}
        self.merges = {}
        self.vocab = {}
        self.vocab_size = 0

    def train(self, text, vocab_size):
        units = split_aksharas(text)

        # Deterministic base vocabulary.
        self.base_tokens = sorted(set(units))
        self.base_to_id = {
            token: idx for idx, token in enumerate(self.base_tokens)
        }

        ids = [self.base_to_id[token] for token in units]
        self.merges = {}

        target_merges = max(0, vocab_size - len(self.base_tokens))

        for _ in range(target_merges):
            stats = get_stats(ids)
            if not stats:
                break

            pair = max(stats, key=lambda item: (stats[item], item))
            new_id = len(self.base_tokens) + len(self.merges)

            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id

        self.vocab = {
            idx: token for idx, token in enumerate(self.base_tokens)
        }

        for pair, new_id in self.merges.items():
            self.vocab[new_id] = (
                self.vocab[pair[0]] + self.vocab[pair[1]]
            )

        self.vocab_size = len(self.vocab)

    def encode(self, text):
        units = split_aksharas(text)
        ids = []

        # Use UTF-8 byte fallback for unseen units.
        for unit in units:
            if unit in self.base_to_id:
                ids.append(self.base_to_id[unit])
            else:
                ids.extend(
                    BYTE_FALLBACK_BASE + byte
                    for byte in unit.encode("utf-8")
                )

        # Never merge fallback bytes with ordinary tokens or each other.
        while len(ids) >= 2:
            stats = get_stats(ids)
            available = [
                pair for pair in stats
                if pair in self.merges
                and pair[0] < BYTE_FALLBACK_BASE
                and pair[1] < BYTE_FALLBACK_BASE
            ]

            if not available:
                break

            pair = min(available, key=lambda p: self.merges[p])
            ids = merge(ids, pair, self.merges[pair])

        return ids

    def decode(self, ids):
        result = bytearray()

        for idx in ids:
            if idx >= BYTE_FALLBACK_BASE:
                result.append(idx - BYTE_FALLBACK_BASE)
            else:
                result.extend(self.vocab[idx].encode("utf-8"))

        return result.decode("utf-8")
