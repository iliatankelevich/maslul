"""Small translation helpers shared by the provider drivers. Imports only core types, never
an SDK, so it doesn't affect the optional-extras isolation."""

from __future__ import annotations

from maslul.types import Message


def last_user_index(messages: list[Message]) -> int:
    """Index of the last ``user``-role message (where ``req.media`` attaches), or -1 if none."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            return i
    return -1
