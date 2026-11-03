
import regex


def split_graphemes(text):
    """Split text into Unicode grapheme clusters."""
    return regex.findall(r"\X", text)


def join_graphemes(graphemes):
    """Reconstruct text from grapheme clusters."""
    return "".join(graphemes)
