"""Clamp sensor readings into a valid band before reporting them."""


def clamp(value, low, high):
    """Return `value` limited to the inclusive band [low, high]."""
    if low > high:
        raise ValueError("low must not exceed high")
    if value < low:
        return low
    if value > high:
        return high
    return value


def clamp_reading(reading, band):
    """Clamp one reading into `band`, a (low, high) pair."""
    low, high = band
    return clamp(reading, low, high)


def clamp_all(readings, band):
    """Clamp every reading into `band`."""
    return [clamp_reading(reading, band) for reading in readings]
