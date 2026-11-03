import json

from .bpe import (
    train,
    build_vocab,
    encode as bpe_encode,
    decode as bpe_decode,
)


class Tokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.merges = {}
        self.vocab = {}

    def train(self, text, vocab_size):
        ids = list(text.encode("utf-8"))

        _, self.merges = train(ids, vocab_size)
        self.vocab = build_vocab(self.merges)
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        return bpe_encode(ids, self.merges)

    def decode(self, ids):
        return bpe_decode(ids, self.vocab)

    def save(self, path):
        data = {
            "vocab_size": self.vocab_size,
            "merges": [
                [list(pair), idx]
                for pair, idx in self.merges.items()
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]

        self.merges = {
            tuple(pair): idx
            for pair, idx in data["merges"]
        }

        self.vocab = build_vocab(self.merges)