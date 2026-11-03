def evaluate_tokenizer(tokenizer, text):
    """Evaluate tokenization efficiency and exact reconstruction."""

    token_ids = tokenizer.encode(text)
    original_bytes = len(text.encode("utf-8"))
    character_count = len(text)

    return {
        "characters": character_count,
        "utf8_bytes": original_bytes,
        "tokens": len(token_ids),
        "bytes_per_token": (
            original_bytes / len(token_ids)
            if token_ids else 0.0
        ),
        "tokens_per_1000_chars": (
            len(token_ids) / character_count * 1000
            if character_count else 0.0
        ),
        "roundtrip_exact": tokenizer.decode(token_ids) == text,
    }