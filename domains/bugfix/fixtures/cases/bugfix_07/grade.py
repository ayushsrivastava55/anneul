"""Map a numeric score to a letter grade."""

CUTOFFS = ((90, "A"), (80, "B"), (70, "C"), (60, "D"))


def letter_grade(score):
    """Return the letter grade for `score` out of 100.

    The cutoffs are inclusive: exactly 90 is an A, exactly 80 is a B, and so on.
    Anything under 60 is an F.
    """
    if not 0 <= score <= 100:
        raise ValueError("score must be between 0 and 100")
    for cutoff, letter in CUTOFFS:
        if score > cutoff:
            return letter
    return "F"


def class_grades(scores):
    """Letter grades for a list of scores, in the same order."""
    return [letter_grade(score) for score in scores]
