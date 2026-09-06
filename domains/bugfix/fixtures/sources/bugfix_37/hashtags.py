"""Extract hashtags from a social post."""

import re

HASHTAG_RE = re.compile(r"#(\w+)")


def hashtags(post):
    """Return the hashtag words in `post`, without the leading #, in order.

    A bare "#" with no word after it is not a hashtag and must be skipped.
    """
    return HASHTAG_RE.findall(post)


def unique_hashtags(post):
    """The hashtags of `post`, lowercased and de-duplicated, in first-seen order."""
    seen = []
    for tag in hashtags(post):
        lowered = tag.lower()
        if lowered not in seen:
            seen.append(lowered)
    return seen
