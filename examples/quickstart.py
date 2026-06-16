"""Minimal maslul example. Set ANTHROPIC_API_KEY, then run:

uv run --extra anthropic python examples/quickstart.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from maslul import Level, Message, Request, Router


async def main() -> None:
    router = Router.from_toml(Path(__file__).with_name("maslul.toml"))

    # Let the router decide the tier (ROUTE_DEFAULT → default_level here).
    reply = await router.complete(Request(messages=[Message(role="user", content="Hello!")]))
    print(f"[{reply.level_used}] {reply.text}")

    # Or pin a tier explicitly.
    hard = await router.complete(
        Request(messages=[Message(role="user", content="Explain the CAP theorem in one line.")]),
        level=Level.HARD,
    )
    print(f"[{hard.level_used}] {hard.text}")


if __name__ == "__main__":
    asyncio.run(main())
