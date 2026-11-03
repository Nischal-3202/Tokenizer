def text_to_bytes(text):
    return list(text.encode("utf-8"))


def bytes_to_text(ids):
    return bytes(ids).decode("utf-8")