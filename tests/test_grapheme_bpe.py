
from tokenizer.grapheme_bpe import GraphemeBPE


def test_roundtrip_nepali():
    text = "नेपालको सुन्दर देश"

    tokenizer = GraphemeBPE()
    tokenizer.train(text, vocab_size=100)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_conjuncts():
    text = "क्षेत्र विद्यालय नमस्ते"

    tokenizer = GraphemeBPE()
    tokenizer.train(text, vocab_size=100)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_graphemes_are_not_split():
    text = "क्षेत्र"

    tokenizer = GraphemeBPE()
    tokenizer.train(text, vocab_size=100)

    encoded = tokenizer.encode(text)
    decoded_tokens = [tokenizer.vocab[idx] for idx in encoded]

    # Every token must consist of whole grapheme clusters.
    from tokenizer.graphemes import split_graphemes

    for token in decoded_tokens:
        assert "".join(split_graphemes(token)) == token


def test_empty_string():
    tokenizer = GraphemeBPE()
    tokenizer.train("नेपाल", vocab_size=20)

    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""

def test_unseen_grapheme_fallback():
    tokenizer = GraphemeBPE()
    tokenizer.train("नेपाल नमस्ते विद्यालय", vocab_size=1024)

    text = "न्फो केँ क्न्या"

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text