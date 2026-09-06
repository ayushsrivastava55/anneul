from grade import class_grades, letter_grade


def test_exactly_on_a_cutoff_gets_the_higher_grade():
    assert letter_grade(90) == "A"
    assert letter_grade(80) == "B"
    assert letter_grade(60) == "D"


def test_between_cutoffs():
    assert letter_grade(85) == "B"


def test_failing_score():
    assert letter_grade(12) == "F"


def test_class_grades():
    assert class_grades([90, 70, 0]) == ["A", "C", "F"]
