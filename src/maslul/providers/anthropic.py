"""Anthropic provider — wraps the official ``anthropic`` SDK.

Covers plain-text completion (M1) and the tool-use translation (M2): tool defs, the
``tool_use``/``tool_result`` round-trip, normalized usage, finish reason, and error mapping.
The loop itself is owned by the router; this provider only translates one turn each way.
Anything the core doesn't model is passed through via ``req.provider_options`` (``thinking``,
``output_config`` effort, a hand-built content-block ``system``, …).

**Prompt caching.** Anthropic is the only provider that requires the caller to *mark* what to
cache; the other three cache a matching prefix automatically. So this is the one driver that
translates :class:`~maslul.ContextCache` into a mechanism — ``cache_control`` breakpoints, placed
by ``_place_breakpoints`` within the API's 4-breakpoint budget. ``ContextCache.key`` is ignored
(Anthropic has no affinity knob; the prefix itself is the key). Below the model's minimum
cacheable prefix (2,048–4,096 tokens, model-dependent) the API silently ignores a marker rather
than erroring — maslul does not try to predict that; it shows up as
``cache_creation_input_tokens == 0``.

Importing this module requires the ``anthropic`` extra (``pip install maslul[anthropic]``).
"""

from __future__ import annotations

import base64
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from maslul.errors import AuthError, ProviderError, RateLimited, Timeout
from maslul.providers._common import media_first, media_index
from maslul.types import (
    ContextCache,
    MediaPart,
    Message,
    ModelSpec,
    Request,
    Response,
    ToolCall,
    Usage,
)

_DEFAULT_MAX_TOKENS = 1024
# Guard against a runaway server-side-tool (web search) resume loop.
_MAX_SERVER_TOOL_TURNS = 10
# Anthropic's server-side web search tool type (versioned).
_WEB_SEARCH_TOOL = "web_search_20250305"
# Hard API limit: a request carrying more than four cache_control breakpoints is a 400.
_MAX_BREAKPOINTS = 4
# ``ttl_seconds`` at or above this opts into the 1-hour cache (2x write premium vs 1.25x).
_LONG_TTL_SECONDS = 3600


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
        cache = req.context_cache
        # Prompt caching. Anthropic is the only provider that requires explicit markers; the render
        # order is tools → system → messages, and a breakpoint caches *everything before it*.
        # ``system_block`` is the last system block: one marker there caches tools + system both.
        system_blocks: list[dict[str, Any]] | None = None
        system_block: dict[str, Any] | None = None
        if req.system:
            if cache is not None and cache.system:
                system_blocks = [{"type": "text", "text": s} for s in req.system]
                system_block = system_blocks[-1]
                kwargs["system"] = system_blocks
            else:
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
        # A caller-pinned system (provider_options — Kippy's hand-built content blocks) *overrides*
        # ours, so our block is now an orphan: don't mark or count it.
        if kwargs.get("system") is not system_blocks:
            system_block = None

        # Server-side tools (web search) pause the turn; resume by echoing the raw assistant
        # content until a terminal stop reason. Usage accumulates across the resumed calls.
        messages, media_block, history_block = _to_messages(req.messages, req.media, cache)
        if cache is not None:
            _place_breakpoints(kwargs, messages, cache, media_block, system_block, history_block)
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


def _to_messages(
    messages: list[Message],
    media: list[MediaPart] | None,
    cache: ContextCache | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    """Normalized messages → Anthropic's shape. Consecutive ``tool`` results collapse into a
    single ``user`` message of ``tool_result`` blocks (the API expects them grouped).

    ``media`` attaches to the last user message by default, or the **first** when
    ``ContextCache(media=True)`` moves it into the stable prefix (see ``_common.media_index``).

    Also returns the two blocks a caller may want to mark: the **last media block**, and the
    **last content block of the last message** (the conversation prefix a follow-up turn reads
    back). Both are ``None`` when there is nothing to mark.
    """
    media_at = media_index(messages, media, cache)
    out: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    media_block: dict[str, Any] | None = None

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
            text = [{"type": "text", "text": m.content}] if m.content else []
            parts = [_media_block(p) for p in media]
            media_block = parts[-1]
            # Media BEFORE the text when caching it: the question is volatile and must not sit
            # inside the prefix the document's breakpoint caches (see _common.media_first).
            blocks = [*parts, *text] if media_first(cache) else [*text, *parts]
            out.append({"role": "user", "content": blocks})
        else:
            out.append({"role": m.role, "content": m.content})
    flush()
    # Only when history caching is actually requested: _last_block *rewrites* a plain-string
    # content into a block list so a marker can attach, and doing that unconditionally would
    # change the wire format of every request that never asked for caching.
    history_block = _last_block(out) if cache is not None and cache.history else None
    return out, media_block, history_block


def _last_block(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The last content block of the last message, converting a plain-string content to a text
    block so a marker can be attached. ``None`` when there is nothing markable (empty content)."""
    if not messages:
        return None
    last = messages[-1]
    content = last["content"]
    if isinstance(content, str):
        if not content:
            return None  # an empty turn carries no block to mark
        content = [{"type": "text", "text": content}]
        last["content"] = content
    return content[-1] if content else None


def _media_block(part: MediaPart) -> dict[str, Any]:
    """A base64 image/document content block. PDFs use a ``document`` block; images, ``image``."""
    b64 = base64.standard_b64encode(part.data).decode()
    kind = "document" if part.mime_type == "application/pdf" else "image"
    return {"type": kind, "source": {"type": "base64", "media_type": part.mime_type, "data": b64}}


def _cache_control(cache: ContextCache) -> dict[str, Any]:
    """``ttl_seconds >= 3600`` → the 1-hour cache; otherwise the 5-minute default."""
    if cache.ttl_seconds is not None and cache.ttl_seconds >= _LONG_TTL_SECONDS:
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


def _count_breakpoints(obj: Any) -> int:
    """``cache_control`` markers anywhere in a rendered system / tools / messages value — including
    ones the caller hand-placed through ``provider_options``, which count against the same limit."""
    if isinstance(obj, dict):
        return ("cache_control" in obj) + sum(
            _count_breakpoints(v) for v in obj.values() if isinstance(v, dict | list)
        )
    if isinstance(obj, list):
        return sum(_count_breakpoints(v) for v in obj)
    return 0


def _place_breakpoints(
    kwargs: dict[str, Any],
    messages: list[dict[str, Any]],
    cache: ContextCache,
    media_block: dict[str, Any] | None,
    system_block: dict[str, Any] | None,
    history_block: dict[str, Any] | None,
) -> None:
    """Mark the requested blocks, respecting Anthropic's 4-breakpoint limit.

    Marked in **descending value** order — media > system > history — so that when the budget is
    tight the marker that never lands is the cheapest one. (Media is the 100k-token PDF; system is a
    few thousand tokens and cheap to re-write; history is the least valuable per token.)

    Two aliasing cases the ``cache_control in block`` guard covers: on a single-turn request the
    media *is* the last content block, so media and history resolve to the same dict and it must be
    marked (and counted) once; and a caller's own ``provider_options`` markers already count against
    the same API limit, so we budget around them rather than pushing the request into a 400.
    """
    budget = _MAX_BREAKPOINTS - (
        _count_breakpoints(kwargs.get("system"))
        + _count_breakpoints(kwargs.get("tools"))
        + _count_breakpoints(messages)
    )
    control = _cache_control(cache)
    for enabled, block in (
        (cache.media, media_block),
        (cache.system, system_block),
        (cache.history, history_block),
    ):
        if budget <= 0:
            break
        if not enabled or block is None or "cache_control" in block:
            continue
        block["cache_control"] = dict(control)
        budget -= 1


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
