import unicodedata


VIRAMA = "\u094d"


def is_devanagari_consonant(char):
    """Check whether a character is a Devanagari consonant."""
    return (
        "\u0915" <= char <= "\u0939"
        or "\u0958" <= char <= "\u095f"
    )


def is_mark(char):
    """Check whether a character is a Unicode combining mark."""
    return unicodedata.category(char).startswith("M")


def split_aksharas(text):
    """
    First-pass Devanagari akshara segmentation.

    Keeps consonant-virama-consonant sequences and following
    combining marks together. Preserves spaces and punctuation.
    """
    result = []
    i = 0

    while i < len(text):
        start = i
        char = text[i]

        # Preserve characters outside the Devanagari block.
        if not ("\u0900" <= char <= "\u097f"):
            cluster = char
            i += 1

            while i < len(text) and is_mark(text[i]):
                cluster += text[i]
                i += 1

            result.append(cluster)
            continue

        i += 1

        while i < len(text):
            # Check virama BEFORE generic combining marks.
            if (
                text[i] == VIRAMA
                and i + 1 < len(text)
                and is_devanagari_consonant(text[i + 1])
            ):
                i += 2
                continue

            # Consume vowel signs, anusvara, chandrabindu, etc.
            if is_mark(text[i]):
                i += 1
                continue

            break

        result.append(text[start:i])

    return result