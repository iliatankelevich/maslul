"""Anthropic provider — wraps the official ``anthropic`` SDK.

M1 covers plain-text completion with normalized usage, finish reason, and error mapping.
The tool-use loop (M2), structured output, and vision (M3) build on this. Anything the core
doesn't model is passed through verbatim via ``req.provider_options`` (prompt caching,
``thinking``, ``output_config`` effort, …), so nothing the Anthropic API offers is lost.
"""

from __future__ import annotations

from typing import Any

from maslul.errors import AuthError, ProviderError, RateLimited, Timeout
from maslul.types import ModelSpec, Request, Response, Usage

_DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider:
    """Async Anthropic backend. Satisfies the :class:`~maslul.Provider` protocol."""

    name = "anthropic"

    def __init__(self, *, api_key: str | None = None, client: Any | None = None) -> None:
        """``client`` is for tests/advanced wiring; otherwise an ``AsyncAnthropic`` is built
        (resolving ``api_key`` or the ``ANTHROPIC_API_KEY`` environment variable)."""
        if client is not None:
            self._client: Any = client
            return
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()

    async def complete(self, spec: ModelSpec, req: Request) -> Response:
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "max_tokens": req.max_tokens or spec.max_tokens or _DEFAULT_MAX_TOKENS,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        }
        if req.system:
            kwargs["system"] = "\n\n".join(req.system)
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.stop:
            kwargs["stop_sequences"] = req.stop
        kwargs.update(spec.options)
        kwargs.update(req.provider_options)
        try:
            resp = await self._client.messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - normalized to a MaslulError below
            raise _map_error(e) from e
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return Response(
            text=text,
            level_used=None,
            provider=self.name,
            model=spec.model,
            usage=_usage(resp.usage),
            finish_reason=getattr(resp, "stop_reason", None),
            raw=resp,
        )

    async def healthcheck(self, spec: ModelSpec) -> None:
        await self._client.messages.create(
            model=spec.model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )


def _usage(u: Any) -> Usage:
    return Usage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


def _map_error(e: Exception) -> Exception:
    """Normalize an SDK exception to a :class:`~maslul.MaslulError`; pass others through."""
    import anthropic

    if isinstance(e, anthropic.RateLimitError):
        return RateLimited(str(e))
    if isinstance(e, anthropic.APITimeoutError):
        return Timeout(str(e))
    if isinstance(e, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return AuthError(str(e))
    if isinstance(e, anthropic.APIError):
        return ProviderError(str(e))
    return e
