"""Module-level airline database plus the `python:` tool impls referenced by tools.yaml.

State lives in `_DB`. `eval.setup(task)` calls `reset()` before every run so each task starts
from the pristine tau-bench snapshot. Every wrapper returns a string, tau-bench style
(JSON on success, "Error: ..." on failure), so the runtime never sees an exception.
"""

from __future__ import annotations

from typing import Any

from .tau_airline.env import data_hash, invoke, load_data

_DB: dict[str, Any] = load_data()


def reset() -> None:
    """Reload the pristine snapshot from data/. Called by eval.setup()."""
    _DB.clear()
    _DB.update(load_data())


def db_hash() -> str:
    """tau-bench consistent_hash of the live database."""
    return data_hash(_DB)


def get_user_details(user_id: str) -> str:
    return invoke(_DB, "get_user_details", {"user_id": user_id})


def get_reservation_details(reservation_id: str) -> str:
    return invoke(_DB, "get_reservation_details", {"reservation_id": reservation_id})


def search_direct_flight(origin: str, destination: str, date: str) -> str:
    return invoke(
        _DB, "search_direct_flight", {"origin": origin, "destination": destination, "date": date}
    )


def search_onestop_flight(origin: str, destination: str, date: str) -> str:
    return invoke(
        _DB, "search_onestop_flight", {"origin": origin, "destination": destination, "date": date}
    )


def list_all_airports() -> str:
    return invoke(_DB, "list_all_airports", {})


def book_reservation(
    user_id: str,
    origin: str,
    destination: str,
    flight_type: str,
    cabin: str,
    flights: list[dict[str, Any]],
    passengers: list[dict[str, Any]],
    payment_methods: list[dict[str, Any]],
    total_baggages: int,
    nonfree_baggages: int,
    insurance: str,
) -> str:
    return invoke(
        _DB,
        "book_reservation",
        {
            "user_id": user_id,
            "origin": origin,
            "destination": destination,
            "flight_type": flight_type,
            "cabin": cabin,
            "flights": flights,
            "passengers": passengers,
            "payment_methods": payment_methods,
            "total_baggages": total_baggages,
            "nonfree_baggages": nonfree_baggages,
            "insurance": insurance,
        },
    )


def update_reservation_flights(
    reservation_id: str, cabin: str, flights: list[dict[str, Any]], payment_id: str
) -> str:
    return invoke(
        _DB,
        "update_reservation_flights",
        {
            "reservation_id": reservation_id,
            "cabin": cabin,
            "flights": flights,
            "payment_id": payment_id,
        },
    )


def update_reservation_baggages(
    reservation_id: str, total_baggages: int, nonfree_baggages: int, payment_id: str
) -> str:
    return invoke(
        _DB,
        "update_reservation_baggages",
        {
            "reservation_id": reservation_id,
            "total_baggages": total_baggages,
            "nonfree_baggages": nonfree_baggages,
            "payment_id": payment_id,
        },
    )


def update_reservation_passengers(reservation_id: str, passengers: list[dict[str, Any]]) -> str:
    return invoke(
        _DB,
        "update_reservation_passengers",
        {"reservation_id": reservation_id, "passengers": passengers},
    )


def cancel_reservation(reservation_id: str) -> str:
    return invoke(_DB, "cancel_reservation", {"reservation_id": reservation_id})


def send_certificate(user_id: str, amount: int) -> str:
    return invoke(_DB, "send_certificate", {"user_id": user_id, "amount": amount})


def calculate(expression: str) -> str:
    return invoke(_DB, "calculate", {"expression": expression})


def transfer_to_human_agents(summary: str) -> str:
    return invoke(_DB, "transfer_to_human_agents", {"summary": summary})
