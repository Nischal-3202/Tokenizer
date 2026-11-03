"""Byte-level BPE encoder equivalent to tokenizer.bpe.encode.

Lowest merge id wins. A newly created id cannot recreate the pair just merged,
and an earlier merge cannot mention a later id. New augmentation merges use
ids above the original shared ids, so every original merge keeps its priority.
"""

from __future__ import annotations

import heapq
from array import array


def encode_bytes(data: bytes, merges: dict[tuple[int, int], int]) -> list[int]:
    n = len(data)
    if n == 0:
        return []
    if n == 1:
        return [data[0]]

    tok = array("I", list(data))
    nxt = array("I", range(1, n + 1))
    prv = array("I", [n, *range(n - 1)])
    dead = bytearray(n)

    heap: list[tuple[int, int]] = []
    for i in range(n - 1):
        nid = merges.get((data[i], data[i + 1]))
        if nid is not None:
            heap.append((nid, i))
    heapq.heapify(heap)

    while heap:
        nid, i = heapq.heappop(heap)
        if dead[i]:
            continue
        j = nxt[i]
        if j >= n or dead[j]:
            continue
        if merges.get((int(tok[i]), int(tok[j]))) != nid:
            continue
        tok[i] = nid
        dead[j] = 1
        nj = nxt[j]
        nxt[i] = nj
        if nj < n:
            prv[nj] = i
        p = prv[i]
        if p < n and not dead[p]:
            new = merges.get((int(tok[p]), int(tok[i])))
            if new is not None:
                heapq.heappush(heap, (new, p))
        if nj < n:
            new = merges.get((int(tok[i]), int(tok[nj])))
            if new is not None:
                heapq.heappush(heap, (new, i))

    out: list[int] = []
    i = 0
    while i < n:
        out.append(int(tok[i]))
        i = int(nxt[i])
    return out


def token_spans(ids: list[int], vocab: dict[int, bytes]) -> list[tuple[int, int, int]]:
    spans = []
    pos = 0
    for tid in ids:
        blob = vocab[tid]
        end = pos + len(blob)
        spans.append((pos, end, tid))
        pos = end
    return spans


def decode_bytes(ids: list[int], vocab: dict[int, bytes]) -> bytes:
    return b"".join(vocab[tid] for tid in ids)
