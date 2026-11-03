from tokenizer.tokenizer import Tokenizer


def test_tokenizer_roundtrip():
    tokenizer = Tokenizer()

    tokenizer.train("hello hello hello", vocab_size=260)

    text = "hello hello"

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text

def test_save_and_load(tmp_path):
    tokenizer = Tokenizer()

    tokenizer.train(
        "hello hello hello hello",
        vocab_size=270,
    )

    text = "hello hello"

    original_encoded = tokenizer.encode(text)

    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)

    loaded_tokenizer = Tokenizer()
    loaded_tokenizer.load(path)

    loaded_encoded = loaded_tokenizer.encode(text)

    assert loaded_encoded == original_encoded
    assert loaded_tokenizer.decode(loaded_encoded) == text