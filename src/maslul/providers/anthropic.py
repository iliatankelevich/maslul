"""Anthropic provider — wraps the official ``anthropic`` SDK.

Covers plain-text completion (M1) and the tool-use translation (M2): tool defs, the
``tool_use``/``tool_result`` round-trip, normalized usage, finish reason, and error mapping.
The loop itself is owned by the router; this provider only translates one turn each way.
Anything the core doesn't model is passed through via ``req.provider_options`` (prompt caching,
``thinking``, ``output_config`` effort, …).

Importing this module requires the ``anthropic`` extra (``pip install maslul[anthropic]``).
"""

from __future__ import annotations

import base64
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from maslul.errors import AuthError, ProviderError, RateLimited, Timeout
from maslul.providers._common import last_user_index
from maslul.types import MediaPart, Message, ModelSpec, Request, Response, ToolCall, Usage

_DEFAULT_MAX_TOKENS = 1024
# Guard against a runaway server-side-tool (web search) resume loop.
_MAX_SERVER_TOOL_TURNS = 10
# Anthropic's server-side web search tool type (versioned).
_WEB_SEARCH_TOOL = "web_search_20250305"


def _has_web_search(server_tools: list[dict[str, Any]] | None) -> bool:
    """True if a raw web_search server tool was already supplied (avoid double-adding)."""
    return any(str(t.get("type", "")).startswith("web_search") for t in (server_tools or []))


class AnthropicProvider:
    """Async Anthropic backend. Satisfies the :class:`~maslul.Provider` protocol."""

    name = "anthropic"

    def __init__(self, *, api_key: str | None = None, client: Any | None = None) -> None:
        """``client`` is for tests/advanced wiring; otherwise an ``AsyncAnthropic`` is built
        (resolving ``api_key`` or the ``ANTHROPIC_API_KEY`` environment variable)."""
        self._client: Any = client or (
            AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()
        )

    async def complete(self, spec: ModelSpec, req: Request) -> Response:
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "max_tokens": req.max_tokens or spec.max_tokens or _DEFAULT_MAX_TOKENS,
        }
        # Client tools (router-executed) + raw server-side tools (web search, run by Anthropic).
        tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in (req.tools or [])
        ]
        tools += list(req.server_tools or [])
        # Normalized web search → Anthropic's server-side web_search tool (unless the caller already
        # passed one raw via server_tools, for back-compat).
        if req.web_search and not _has_web_search(req.server_tools):
            web: dict[str, Any] = {"type": _WEB_SEARCH_TOOL, "name": "web_search"}
            if req.web_search_max_uses is not None:
                web["max_uses"] = req.web_search_max_uses
            tools.append(web)
        if tools:
            kwargs["tools"] = tools
        if req.system:
            kwargs["system"] = "\n\n".join(req.system)
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.stop:
            kwargs["stop_sequences"] = req.stop
        kwargs.update(spec.options)
        kwargs.update(req.provider_options)
        if req.response_format is not None:
            # merge into output_config so it coexists with effort/thinking from provider_options
            output_config = dict(kwargs.get("output_config") or {})
            output_config["format"] = {"type": "json_schema", "schema": req.response_format}
            kwargs["output_config"] = output_config

        # Server-side tools (web search) pause the turn; resume by echoing the raw assistant
        # content until a terminal stop reason. Usage accumulates across the resumed calls.
        messages = _to_messages(req.messages, req.media)
        usage = Usage()
        turns = 0
        while True:
            try:
                resp = await self._client.messages.create(messages=messages, **kwargs)
            except Exception as e:  # noqa: BLE001 - normalized to a MaslulError below
                raise _map_error(e) from e
            _add_usage(usage, resp.usage)
            turns += 1
            paused = getattr(resp, "stop_reason", None) == "pause_turn"
            if not paused or turns >= _MAX_SERVER_TOOL_TURNS:
                break
            messages = [*messages, {"role": "assistant", "content": resp.content}]
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
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=getattr(resp, "stop_reason", None),
            sources=_sources(resp.content),
            raw=resp,
        )

    async def healthcheck(self, spec: ModelSpec) -> None:
        await self._client.messages.create(
            model=spec.model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )


def _to_messages(messages: list[Message], media: list[MediaPart] | None) -> list[dict[str, Any]]:
    """Normalized messages → Anthropic's shape. Consecutive ``tool`` results collapse into a
    single ``user`` message of ``tool_result`` blocks (the API expects them grouped). Any
    ``media`` is attached to the last user message as image/document blocks."""
    media_at = last_user_index(messages) if media else -1
    out: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            out.append({"role": "user", "content": list(pending)})
            pending.clear()

    for i, m in enumerate(messages):
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
        elif i == media_at and media:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks += [_media_block(p) for p in media]
            out.append({"role": "user", "content": blocks})
        else:
            out.append({"role": m.role, "content": m.content})
    flush()
    return out


def _media_block(part: MediaPart) -> dict[str, Any]:
    """A base64 image/document content block. PDFs use a ``document`` block; images, ``image``."""
    b64 = base64.standard_b64encode(part.data).decode()
    kind = "document" if part.mime_type == "application/pdf" else "image"
    return {"type": kind, "source": {"type": "base64", "media_type": part.mime_type, "data": b64}}


def _usage(u: Any) -> Usage:
    return Usage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


def _add_usage(total: Usage, u: Any) -> None:
    one = _usage(u)
    total.input_tokens += one.input_tokens
    total.output_tokens += one.output_tokens
    total.cache_read_input_tokens += one.cache_read_input_tokens
    total.cache_creation_input_tokens += one.cache_creation_input_tokens


def _sources(content: Any) -> list[str]:
    """Unique citation URLs from text blocks (server-side web search results)."""
    urls: list[str] = []
    for block in content or []:
        if getattr(block, "type", None) != "text":
            continue
        for citation in getattr(block, "citations", None) or []:
            url = getattr(citation, "url", None)
            if url and url not in urls:
                urls.append(url)
    return urls


def _map_error(e: Exception) -> Exception:
    """Normalize an SDK exception to a :class:`~maslul.MaslulError`; pass others through."""
    if isinstance(e, anthropic.RateLimitError):
        return RateLimited(str(e))
    if isinstance(e, anthropic.APITimeoutError):
        return Timeout(str(e))
    if isinstance(e, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return AuthError(str(e))
    if isinstance(e, anthropic.APIError):
        return ProviderError(str(e))
    return e
