from fizz_range import fizz_count, fizzbuzz


def test_includes_the_final_number():
    assert fizzbuzz(5) == ["1", "2", "fizz", "4", "buzz"]


def test_length_matches_n():
    assert len(fizzbuzz(20)) == 20


def test_fifteen_is_fizzbuzz():
    assert fizzbuzz(15)[-1] == "fizzbuzz"


def test_empty_for_zero():
    assert fizzbuzz(0) == []


def test_fizz_count():
    assert fizz_count(10) == 3
