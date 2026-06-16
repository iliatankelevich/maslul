"""Anthropic provider — wraps the official ``anthropic`` SDK.

Covers plain-text completion (M1) and the tool-use translation (M2): tool defs, the
``tool_use``/``tool_result`` round-trip, normalized usage, finish reason, and error mapping.
The loop itself is owned by the router; this provider only translates one turn each way.
Anything the core doesn't model is passed through via ``req.provider_options`` (prompt caching,
``thinking``, ``output_config`` effort, …).
"""

from __future__ import annotations

from typing import Any

from maslul.errors import AuthError, ProviderError, RateLimited, Timeout
from maslul.types import Message, ModelSpec, Request, Response, ToolCall, Usage

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
            "messages": _to_messages(req.messages),
        }
        if req.system:
            kwargs["system"] = "\n\n".join(req.system)
        if req.tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in req.tools
            ]
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
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=dict(b.input))
            for b in resp.content
            if getattr(b, "type", None) == "tool_use"
        ]
        return Response(
            text=text,
            level_used=None,
            provider=self.name,
            model=spec.model,
            usage=_usage(resp.usage),
            tool_calls=tool_calls,
            finish_reason=getattr(resp, "stop_reason", None),
            raw=resp,
        )

    async def healthcheck(self, spec: ModelSpec) -> None:
        await self._client.messages.create(
            model=spec.model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )


def _to_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Normalized messages → Anthropic's shape. Consecutive ``tool`` results collapse into a
    single ``user`` message of ``tool_result`` blocks (the API expects them grouped)."""
    out: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            out.append({"role": "user", "content": list(pending)})
            pending.clear()

    for m in messages:
        if m.role == "tool":
            pending.append(
                {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            )
            continue
        flush()
        if m.role == "assistant" and m.tool_calls:
            content: list[dict[str, Any]] = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            content += [
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                for tc in m.tool_calls
            ]
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": m.role, "content": m.content})
    flush()
    return out


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
