"""Count words in a piece of text."""


def count_words(text):
    """Return a dict mapping each lowercased word to how often it appears.

    Words are split on whitespace and stripped of surrounding punctuation. The
    return value is a dict so callers can look words up directly.
    """
    counts = {}
    for raw in text.split():
        word = raw.strip(".,!?;:").lower()
        if not word:
            continue
        counts[word] = counts.get(word, 0) + 1
    return counts


def most_common(text):
    """The most frequent word, ties broken alphabetically, or None for no words."""
    counts = count_words(text)
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
