"""Test-only fakes. Kept out of the shipped package."""

from __future__ import annotations

from maslul.types import ModelSpec, Request, Response, Usage


class FakeProvider:
    """A canned :class:`~maslul.Provider`: records its calls, returns a fixed reply.

    Structurally satisfies the ``Provider`` protocol (``name`` + ``complete`` +
    ``healthcheck``) without importing it, so M0 can exercise the router with no SDKs.
    """

    def __init__(self, name: str, *, text: str = "fake-reply") -> None:
        self.name = name
        self._text = text
        self.calls: list[tuple[ModelSpec, Request]] = []

    async def complete(self, spec: ModelSpec, req: Request) -> Response:
        self.calls.append((spec, req))
        return Response(
            text=self._text,
            level_used=None,
            provider=spec.provider,
            model=spec.model,
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    async def healthcheck(self, spec: ModelSpec) -> None:
        return None
