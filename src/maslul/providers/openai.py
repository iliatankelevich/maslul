"""OpenAI provider — the official ``openai`` SDK (Chat Completions).

Covers text completion, tool use (function calling), structured output (``json_schema``), vision
(``image_url``), and web search (``web_search_options`` — on search-capable models). Citations from
the message's ``url_citation`` annotations normalize into ``Response.sources``.

Importing this module requires the ``openai`` extra (``pip install maslul[openai]``).
"""

from __future__ import annotations

import base64
import json
from typing import Any

import openai
from openai import AsyncOpenAI

from maslul.errors import AuthError, ProviderError, RateLimited, Timeout
from maslul.providers._common import media_first, media_index, split_input
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
# ``ttl_seconds`` at or above this opts into OpenAI's extended (24-hour) cache retention.
_LONG_TTL_SECONDS = 3600


class OpenAIProvider:
    """Async OpenAI backend. Satisfies the :class:`~maslul.Provider` protocol."""

    name = "openai"

    def __init__(self, *, api_key: str | None = None, client: Any | None = None) -> None:
        """``client`` is for tests/advanced wiring; otherwise an ``AsyncOpenAI`` is built
        (resolving ``api_key`` or the ``OPENAI_API_KEY`` environment variable)."""
        self._client: Any = client or (AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI())

    async def complete(self, spec: ModelSpec, req: Request) -> Response:
        messages = _to_messages(req.messages, req.media, req.context_cache)
        if req.system:
            messages = [{"role": "system", "content": "\n\n".join(req.system)}, *messages]
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "messages": messages,
            "max_completion_tokens": req.max_tokens or spec.max_tokens or _DEFAULT_MAX_TOKENS,
        }
        if req.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in req.tools
            ]
        if req.response_format is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": req.response_format, "strict": False},
            }
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.stop:
            kwargs["stop"] = req.stop
        if req.web_search:  # search-capable models only (e.g. gpt-4o-search-preview)
            kwargs["web_search_options"] = {}
        # Prompt caching. OpenAI caches a matching prefix automatically — the stable-first ordering
        # in _to_messages is what earns the hit. The affinity key just routes like prompts to the
        # same cache; ttl opts into extended retention.
        cache = req.context_cache
        if cache is not None:
            if cache.key is not None:
                kwargs["prompt_cache_key"] = cache.key
            if cache.ttl_seconds is not None and cache.ttl_seconds >= _LONG_TTL_SECONDS:
                kwargs["prompt_cache_retention"] = "24h"
        kwargs.update(spec.options)
        kwargs.update(req.provider_options)
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - normalized below
            raise _map_error(e) from e
        choice = resp.choices[0]
        message = choice.message
        return Response(
            text=getattr(message, "content", None) or "",
            level_used=None,
            provider=self.name,
            model=spec.model,
            usage=_usage(getattr(resp, "usage", None)),
            tool_calls=_tool_calls(message),
            finish_reason=getattr(choice, "finish_reason", None),
            sources=_sources(message),
            raw=resp,
        )

    async def healthcheck(self, spec: ModelSpec) -> None:
        await self._client.chat.completions.create(
            model=spec.model,
            max_completion_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )


def _to_messages(
    messages: list[Message],
    media: list[MediaPart] | None,
    cache: ContextCache | None = None,
) -> list[dict[str, Any]]:
    """Normalized messages → OpenAI's shape. ``media`` is attached to the last user message as
    ``image_url`` parts — or the **first** when ``ContextCache(media=True)`` moves it into the
    cacheable prefix; tool results become ``role="tool"`` messages keyed by ``tool_call_id``."""
    media_at = media_index(messages, media, cache)
    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
        elif i == media_at and media:
            text = [{"type": "text", "text": m.content}] if m.content else []
            parts = [_image_part(p) for p in media]
            # Media before the text when caching it, so the volatile question stays out of the
            # cacheable prefix (see _common.media_first).
            content = [*parts, *text] if media_first(cache) else [*text, *parts]
            out.append({"role": m.role, "content": content})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _image_part(part: MediaPart) -> dict[str, Any]:
    b64 = base64.standard_b64encode(part.data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{part.mime_type};base64,{b64}"}}


def _tool_calls(message: Any) -> list[ToolCall]:
    out: list[ToolCall] = []
    for tc in getattr(message, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        raw_args = getattr(fn, "arguments", "") if fn is not None else ""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except (ValueError, TypeError):
            args = {}
        out.append(
            ToolCall(
                id=getattr(tc, "id", "") or "",
                name=getattr(fn, "name", "") if fn is not None else "",
                input=args,
            )
        )
    return out


def _usage(u: Any) -> Usage:
    """``prompt_tokens`` is cached-**inclusive** on OpenAI; maslul's convention is disjoint, so the
    cached count is subtracted back out (see :class:`~maslul.Usage`)."""
    if u is None:
        return Usage()
    details = getattr(u, "prompt_tokens_details", None)
    billable, cached = split_input(
        getattr(u, "prompt_tokens", 0) or 0,
        getattr(details, "cached_tokens", 0) or 0,
    )
    return Usage(
        input_tokens=billable,
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
        cache_read_input_tokens=cached,
    )


def _sources(message: Any) -> list[str]:
    """Unique citation URLs from web-search ``url_citation`` annotations."""
    out: list[str] = []
    for ann in getattr(message, "annotations", None) or []:
        if getattr(ann, "type", None) != "url_citation":
            continue
        url = getattr(getattr(ann, "url_citation", None), "url", None)
        if url and url not in out:
            out.append(url)
    return out


def _map_error(e: Exception) -> Exception:
    """Normalize an SDK exception to a :class:`~maslul.MaslulError`; pass others through."""
    if isinstance(e, openai.RateLimitError):
        return RateLimited(str(e))
    if isinstance(e, openai.APITimeoutError):
        return Timeout(str(e))
    if isinstance(e, openai.AuthenticationError | openai.PermissionDeniedError):
        return AuthError(str(e))
    if isinstance(e, openai.APIError):
        return ProviderError(str(e))
    return e
