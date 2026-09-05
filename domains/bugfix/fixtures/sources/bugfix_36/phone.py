"""Validate US phone numbers in the (555) 123-4567 format."""

import re

PHONE_RE = re.compile(r"^\(\d{3}\) \d{3}-\d{4}$")


def is_valid_phone(text):
    """True only when the whole string is a phone number in the canonical format.

    Anything before or after the number makes it invalid.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return PHONE_RE.match(text) is not None


def valid_phones(numbers):
    """The valid numbers from `numbers`, in order."""
    return [number for number in numbers if is_valid_phone(number)]


def digits_only(text):
    """The digits of `text`, as a string."""
    return "".join(char for char in text if char.isdigit())
