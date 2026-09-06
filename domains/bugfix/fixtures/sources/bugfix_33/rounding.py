"""Money rounding helpers."""

CENTS = 2


def round_money(amount, digits=CENTS):
    """Return `amount` rounded to `digits` decimal places."""
    if digits < 0:
        raise ValueError("digits cannot be negative")
    return round(amount, digits)


def split_evenly(amount, people):
    """Each person's share of `amount`, rounded to the cent."""
    if people < 1:
        raise ValueError("need at least one person")
    return round_money(amount / people)


def total(amounts):
    """The sum of `amounts`, rounded to the cent."""
    return round_money(sum(amounts))
