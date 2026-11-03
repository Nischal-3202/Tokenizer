from tokenizer.unicode_utils import text_to_bytes, bytes_to_text


def test_english_roundtrip():
    text = "hello"

    ids = text_to_bytes(text)

    assert bytes_to_text(ids) == text


def test_nepali_roundtrip():
    text = "नेपाल"

    ids = text_to_bytes(text)

    assert bytes_to_text(ids) == text


def test_hindi_roundtrip():
    text = "भारत"

    ids = text_to_bytes(text)

    assert bytes_to_text(ids) == text