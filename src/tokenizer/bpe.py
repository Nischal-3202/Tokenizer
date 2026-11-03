def get_stats(ids):
    """
        Given a list of token ids, return a dictionary of pairs and their counts
    """
    counts={}
    for pair in zip(ids, ids[1:]):
        counts[pair]=counts.get(pair,0)+1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1

    return newids

def train(ids, vocab_size):
    num_merges = max(0, vocab_size - 256)

    merges = {}

    for i in range(num_merges):
        stats = get_stats(ids)

        if not stats:
            break

        pair = max(stats, key=stats.get)

        idx = 256 + i

        ids = merge(ids, pair, idx)

        merges[pair] = idx

    return ids, merges 

def build_vocab(merges):
    vocab = {idx: bytes([idx]) for idx in range(256)}

    for pair, idx in merges.items():
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

    return vocab


def encode(ids, merges):
    while len(ids) >= 2:
        stats = get_stats(ids)

        pair = min(
            stats,
            key=lambda p: merges.get(p, float("inf"))
        )

        if pair not in merges:
            break

        idx = merges[pair]
        ids = merge(ids, pair, idx)

    return ids


def decode(ids, vocab):
    tokens = b"".join(vocab[idx] for idx in ids)

    return tokens.decode("utf-8", errors="replace")