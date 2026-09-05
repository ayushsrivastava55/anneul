from word_count import count_words, most_common


def test_counts_are_a_dict():
    counts = count_words("a b a")
    assert isinstance(counts, dict)
    assert counts["a"] == 2


def test_punctuation_and_case_are_ignored():
    assert count_words("Hi, hi!") == {"hi": 2}


def test_empty_text():
    assert count_words("") == {}


def test_most_common():
    assert most_common("b a a") == "a"
