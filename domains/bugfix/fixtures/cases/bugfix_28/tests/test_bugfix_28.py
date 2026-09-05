from vat import gross, vat_amount, vat_breakdown


def test_vat_keeps_its_decimals():
    assert vat_amount(12.34) == 2.47


def test_vat_is_a_float():
    assert isinstance(vat_amount(10), float)


def test_gross():
    assert gross(100) == 120.0


def test_breakdown():
    row = vat_breakdown([12.34])[0]
    assert row == {"net": 12.34, "vat": 2.47, "gross": 14.81}
