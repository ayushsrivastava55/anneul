"""Helpdesk routing: the two lookups a triage assistant is allowed to make."""

from __future__ import annotations

_TEAMS = {
    "billing": ("invoice", "refund", "charge", "payment", "receipt"),
    "access": ("password", "login", "locked", "sso", "2fa"),
    "hardware": ("laptop", "monitor", "keyboard", "battery", "docking"),
    "network": ("vpn", "wifi", "slow", "dns", "proxy"),
}


def list_teams() -> str:
    """List every team you can route a ticket to, and what each one handles."""
    return "\n".join(f"{team}: handles {', '.join(words)}" for team, words in _TEAMS.items())


def lookup_team(keyword: str) -> str:
    """Find which team handles a single keyword. Returns the team name, or 'unknown'."""
    word = keyword.strip().lower()
    for team, words in _TEAMS.items():
        if word in words:
            return team
    return "unknown"
