from ip_address import find_ips, first_ip, is_private


def test_finds_an_address():
    assert find_ips("from 192.168.0.1 ok") == ["192.168.0.1"]


def test_dots_are_literal():
    assert find_ips("id 111x222x333x444 here") == []


def test_two_addresses():
    assert find_ips("10.0.0.1 -> 10.0.0.2") == ["10.0.0.1", "10.0.0.2"]


def test_first_ip_and_private():
    assert first_ip("no addresses") is None
    assert is_private("10.0.0.1") is True
