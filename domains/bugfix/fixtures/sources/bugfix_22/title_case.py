"""Title-case a phrase without tripping over blanks."""

SMALL_WORDS = {"a", "an", "and", "of", "the"}


def title_case(phrase):
    """Capitalise each word in `phrase`, leaving small joining words lowercase.

    The first word is always capitalised. Repeated spaces produce empty words, which
    must be passed through untouched rather than indexed into.
    """
    words = phrase.split(" ")
    out = []
    for index, word in enumerate(words):
        if not word:
            out.append(word)
            continue
        if index > 0 and word.lower() in SMALL_WORDS:
            out.append(word.lower())
            continue
        out.append(word[0].upper() + word[1:].lower())
    return " ".join(out)


def title_case_all(phrases):
    """title_case over a list of phrases."""
    return [title_case(phrase) for phrase in phrases]
