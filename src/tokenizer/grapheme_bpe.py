from IPython.testing import skipdoctest
from IPython.testing import skipdoctest
from collections import Counter

from .graphemes import split_graphemes

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


class GraphemeBPE:
    def __init__(self):
        self.base_tokens = []
        self.base_to_id = {}
        self.merges = {}
        self.vocab = {}
        self.vocab_size = 0

    def train(self, text, vocab_size):
        clusters = split_graphemes(text)

        # Deterministic base vocabulary.
        self.base_tokens = sorted(set(clusters))
        self.base_to_id = {
            token: idx
            for idx, token in enumerate(self.base_tokens)
        }

        ids = [self.base_to_id[token] for token in clusters]
        self.merges = {}

        target_merges = max(0, vocab_size - len(self.base_tokens))

        for _ in range(target_merges):
            stats = get_stats(ids)

            if not stats:
                break

            pair = max(
                stats,
                key=lambda item: (stats[item], item),
            )

            new_id = len(self.base_tokens) + len(self.merges)
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id

        self.vocab = {
            idx: token
            for idx, token in enumerate(self.base_tokens)
        }

        for pair, new_id in self.merges.items():
            self.vocab[new_id] = (
                self.vocab[pair[0]] + self.vocab[pair[1]]
            )

        self.vocab_size = len(self.vocab)

    def encode(self, text):
        clusters = split_graphemes(text)
        ids = []

        # Convert graphemes to IDs, using byte fallback for unseen ones.
        for token in clusters:
            if token in self.base_to_id:
                ids.append(self.base_to_id[token])
            else:
                ids.extend(
                    BYTE_FALLBACK_BASE + byte
                    for byte in token.encode("utf-8")
                )

        # Apply learned merges only to ordinary vocabulary IDs.
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
                # Recover an individual UTF-8 byte.
                result.append(idx - BYTE_FALLBACK_BASE)
            else:
                # Convert a normal grapheme or merged token to UTF-8 bytes.
                result.extend(self.vocab[idx].encode("utf-8"))

        return result.decode("utf-8")
