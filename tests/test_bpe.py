from tokenizer.bpe import get_stats, merge, train, build_vocab, encode, decode


def test_get_stats():
    ids = [10, 20, 30, 20, 30]

    stats = get_stats(ids)

    assert stats[(10, 20)] == 1
    assert stats[(20, 30)] == 2
    assert stats[(30, 20)] == 1


def test_merge():
    ids = [10, 20, 30, 20, 30]

    result = merge(ids, (20, 30), 256)

    assert result == [10, 256, 256]

def test_train():
    ids = [1, 2, 1, 2, 1, 2]

    new_ids, merges = train(ids, 257)

    assert len(merges) == 1
    assert (1, 2) in merges
    assert new_ids == [256, 256, 256]

def test_build_vocab():
    merges = {
        (104, 105): 256,
        (256, 33): 257,
    }

    vocab = build_vocab(merges)

    assert vocab[104] == b"h"
    assert vocab[105] == b"i"
    assert vocab[256] == b"hi"
    assert vocab[257] == b"hi!"

def test_encode_decode():
    ids = [104, 105, 33]

    merges = {
        (104, 105): 256,
        (256, 33): 257,
    }

    vocab = build_vocab(merges)

    encoded = encode(ids, merges)

    assert encoded == [257]

    decoded = decode(encoded, vocab)

    assert decoded == "hi!"