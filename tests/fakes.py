"""Test-only fakes. Kept out of the shipped package."""

from __future__ import annotations

from maslul.errors import MaslulError, RateLimited
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


class ScriptedProvider:
    """Returns a pre-scripted sequence of Responses, recording each Request it received.

    Used to drive the router's tool-use loop deterministically: e.g. a tool-call response
    followed by a final text response.
    """

    def __init__(self, name: str, responses: list[Response]) -> None:
        self.name = name
        self._responses = list(responses)
        self.requests: list[Request] = []

    async def complete(self, spec: ModelSpec, req: Request) -> Response:
        self.requests.append(req)
        return self._responses.pop(0)

    async def healthcheck(self, spec: ModelSpec) -> None:
        return None


class FlakyProvider:
    """Raises ``error`` for the first ``fails`` calls, then returns a normal reply. With
    ``fails`` large it never succeeds — used to drive retry and fallback paths."""

    def __init__(self, name: str, *, fails: int, error: MaslulError | None = None) -> None:
        self.name = name
        self._fails = fails
        self._error = error or RateLimited("simulated rate limit")
        self.attempts = 0

    async def complete(self, spec: ModelSpec, req: Request) -> Response:
        self.attempts += 1
        if self.attempts <= self._fails:
            raise self._error
        return Response(
            text="recovered",
            level_used=None,
            provider=spec.provider,
            model=spec.model,
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    async def healthcheck(self, spec: ModelSpec) -> None:
        return None
