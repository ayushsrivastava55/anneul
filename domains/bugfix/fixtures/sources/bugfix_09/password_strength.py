"""A very small password policy check."""

MIN_LENGTH = 10


def is_strong(password):
    """True only when the password satisfies every rule.

    The rules are: at least MIN_LENGTH characters, at least one uppercase letter,
    at least one digit. All three are required together.
    """
    long_enough = len(password) >= MIN_LENGTH
    has_upper = any(char.isupper() for char in password)
    has_digit = any(char.isdigit() for char in password)
    return long_enough and has_upper and has_digit


def weak_reasons(password):
    """The rules `password` breaks, as a sorted list of names."""
    reasons = []
    if len(password) < MIN_LENGTH:
        reasons.append("length")
    if not any(char.isupper() for char in password):
        reasons.append("uppercase")
    if not any(char.isdigit() for char in password):
        reasons.append("digit")
    return sorted(reasons)
