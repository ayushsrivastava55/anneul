"""Classic fizzbuzz, returned as a list."""


def fizzbuzz(n):
    """Return the fizzbuzz strings for 1..n inclusive.

    Multiples of 15 become "fizzbuzz", of 3 "fizz", of 5 "buzz", everything else is
    the number as a string. n < 1 yields an empty list.
    """
    out = []
    for value in range(1, n + 1):
        if value % 15 == 0:
            out.append("fizzbuzz")
        elif value % 3 == 0:
            out.append("fizz")
        elif value % 5 == 0:
            out.append("buzz")
        else:
            out.append(str(value))
    return out


def fizz_count(n):
    """How many entries in fizzbuzz(n) are exactly "fizz"."""
    return sum(1 for word in fizzbuzz(n) if word == "fizz")
