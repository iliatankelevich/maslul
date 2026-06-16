"""The provider contract. A backend implements :class:`Provider` and nothing else."""

from __future__ import annotations

from typing import Protocol

from maslul.types import ModelSpec, Request, Response


class Provider(Protocol):
    """A single LLM backend — the only thing a new provider must implement."""

    name: str

    async def complete(self, spec: ModelSpec, req: Request) -> Response:
        """Run one completion against ``spec``'s model and return a normalized response."""
        ...

    async def healthcheck(self, spec: ModelSpec) -> None:
        """Cheap live call to verify the backend is reachable and authorized (for a doctor)."""
        ...
