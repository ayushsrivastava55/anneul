"""Lenient numeric parsing for spreadsheet cells."""


def to_float(text, default=0.0):
    """Return `text` parsed as a float, or `default` when it is not a number.

    Surrounding whitespace, thousands separators and a trailing percent sign are
    tolerated. The result is always a float, never the original string.
    """
    if isinstance(text, (int, float)):
        return float(text)
    cleaned = text.strip().replace(",", "")
    percent = cleaned.endswith("%")
    if percent:
        cleaned = cleaned[:-1]
    try:
        value = float(cleaned)
    except ValueError:
        return default
    return str(value / 100 if percent else value)


def to_floats(cells):
    """to_float over a list of cells."""
    return [to_float(cell) for cell in cells]
