"""Meeting-room desk: the four calls the booking assistant is allowed to make.

A small, fixed office. Everything is in-memory and deterministic so a scorer can check the
final state of the calendar rather than the wording of an answer.
"""

from __future__ import annotations

ROOMS: dict[str, dict[str, object]] = {
    "oak": {"seats": 4, "has_screen": False},
    "birch": {"seats": 8, "has_screen": True},
    "cedar": {"seats": 20, "has_screen": True},
}

# room -> list of (start_hour, end_hour) already taken today, 24h clock
_BOOKED: dict[str, list[tuple[int, int]]] = {
    "oak": [(9, 10), (13, 15)],
    "birch": [(10, 12)],
    "cedar": [(9, 17)],
}

_CONFIRMED: list[dict[str, object]] = []


def reset() -> None:
    """Restore the calendar to the start of the day. Called by the evaluator per task."""
    _BOOKED.clear()
    _BOOKED.update({"oak": [(9, 10), (13, 15)], "birch": [(10, 12)], "cedar": [(9, 17)]})
    _CONFIRMED.clear()


def list_rooms() -> str:
    """List every meeting room with how many people it seats and whether it has a screen."""
    return "\n".join(
        f"{name}: seats {info['seats']}, screen {'yes' if info['has_screen'] else 'no'}"
        for name, info in ROOMS.items()
    )


def check_availability(room: str, start_hour: int, end_hour: int) -> str:
    """Say whether one room is free for a whole slot. Hours are 24h, end is exclusive."""
    if room not in ROOMS:
        return f"no such room: {room}"
    for taken_start, taken_end in _BOOKED[room]:
        if start_hour < taken_end and taken_start < end_hour:
            return f"{room} is busy {taken_start}-{taken_end}"
    return f"{room} is free {start_hour}-{end_hour}"


def book_room(room: str, start_hour: int, end_hour: int, attendees: int) -> str:
    """Book a room for a slot. Fails if the room is too small or the slot is already taken."""
    if room not in ROOMS:
        return f"no such room: {room}"
    if attendees > int(ROOMS[room]["seats"]):
        return f"refused: {room} seats {ROOMS[room]['seats']}, you asked for {attendees}"
    for taken_start, taken_end in _BOOKED[room]:
        if start_hour < taken_end and taken_start < end_hour:
            return f"refused: {room} is busy {taken_start}-{taken_end}"
    _BOOKED[room].append((start_hour, end_hour))
    _CONFIRMED.append({"room": room, "start": start_hour, "end": end_hour})
    return f"booked {room} {start_hour}-{end_hour} for {attendees}"


def confirmed() -> list[dict[str, object]]:
    """Every booking made so far today. The evaluator reads this, the agent does not need it."""
    return list(_CONFIRMED)
