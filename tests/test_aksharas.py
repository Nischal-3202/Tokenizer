from tokenizer.aksharas import split_aksharas


def test_nepali_words():
    assert split_aksharas("नेपाल") == ["ने", "पा", "ल"]
    assert split_aksharas("नमस्ते") == ["न", "म", "स्ते"]


def test_conjuncts():
    assert split_aksharas("क्षेत्र") == ["क्षे", "त्र"]
    assert split_aksharas("विद्यालय") == ["वि", "द्या", "ल", "य"]
    assert split_aksharas("प्रकाश") == ["प्र", "का", "श"]


def test_combining_marks():
    assert split_aksharas("संस्कृति") == ["सं", "स्कृ", "ति"]
    assert split_aksharas("दृष्टि") == ["दृ", "ष्टि"]


def test_whitespace_and_punctuation():
    text = "नेपाल, नमस्ते!"
    assert "".join(split_aksharas(text)) == text


def test_empty_string():
    assert split_aksharas("") == []