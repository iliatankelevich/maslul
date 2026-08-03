"""Gemini provider — Google ``google-genai`` SDK via Vertex AI.

Mirrors Kippy's auth pattern: Vertex AI + Application Default Credentials (no API key) when
``vertex_project`` is set; an API-key path is supported for the Gemini Developer API. Covers
plain-text completion (M1) and tool-use translation (M2): function declarations, the
``function_call``/``function_response`` round-trip, normalized usage, and finish reason.

Importing this module requires the ``gemini`` extra (``pip install maslul[gemini]``).
"""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from maslul.errors import AuthError, ProviderError, RateLimited, Timeout
from maslul.providers._common import media_first, media_index, split_input
from maslul.types import ModelSpec, Request, Response, ToolCall, Usage


class GeminiProvider:
    """Async Gemini backend. Satisfies the :class:`~maslul.Provider` protocol."""

    name = "gemini"

    def __init__(
        self,
        *,
        vertex_project: str | None = None,
        vertex_location: str = "global",
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client: Any = client
        elif vertex_project:
            self._client = genai.Client(
                vertexai=True, project=vertex_project, location=vertex_location
            )
        elif api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = genai.Client()  # resolve from ADC / environment

    async def complete(self, spec: ModelSpec, req: Request) -> Response:
        config: dict[str, Any] = {}
        if req.system:
            config["system_instruction"] = "\n\n".join(req.system)
        max_tokens = req.max_tokens or spec.max_tokens
        if max_tokens:
            config["max_output_tokens"] = max_tokens
        if req.temperature is not None:
            config["temperature"] = req.temperature
        if req.stop:
            config["stop_sequences"] = req.stop
        if req.response_format is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = req.response_format
        tools: list[Any] = []
        if req.tools:
            tools.append(
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name,
                            description=t.description,
                            parameters_json_schema=t.input_schema,
                        )
                        for t in req.tools
                    ]
                )
            )
        # Normalized web search → Gemini's Google Search grounding. NB: some models reject
        # google_search combined with function_declarations in one request — verify per model.
        if req.web_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        if tools:
            config["tools"] = tools
        try:
            resp = await self._client.aio.models.generate_content(
                model=spec.model,
                contents=_contents(req),
                config=types.GenerateContentConfig(**config) if config else None,
            )
        except Exception as e:  # noqa: BLE001 - normalized below
            raise _map_error(e) from e
        return Response(
            text=_text(resp),
            level_used=None,
            provider=self.name,
            model=spec.model,
            usage=_usage(getattr(resp, "usage_metadata", None)),
            tool_calls=_tool_calls(resp),
            finish_reason=_finish_reason(resp),
            sources=_sources(resp),
            raw=resp,
        )

    async def healthcheck(self, spec: ModelSpec) -> None:
        await self._client.aio.models.generate_content(model=spec.model, contents="ping")


def _contents(req: Request) -> list[Any]:
    """Media attaches to the last user message, or the **first** when ``ContextCache(media=True)``
    moves it into the cacheable prefix — all Gemini needs, since it caches a matching prefix
    implicitly (Gemini 2.5+). Explicit ``CachedContent`` objects are deliberately not used: they
    add server-side state maslul has never had, and implicit caching may capture the win already.

    ⚠️ Consecutive ``tool`` results collapse into a **single** ``tool`` content, the way Anthropic's
    ``tool_result`` blocks do. Gemini counts parts per turn, not ids: a model turn holding N
    ``functionCall`` parts must be answered by ONE turn holding N ``functionResponse`` parts, or the
    request is rejected outright ("Please ensure that the number of function response parts is equal
    to the number of function call parts of the function call turn"). One content per result made
    every **parallel** tool call — two tools in one turn — a hard 400.
    """
    media_at = media_index(req.messages, req.media, req.context_cache)
    out: list[Any] = []
    pending: list[Any] = []

    def flush() -> None:
        if pending:
            out.append(types.Content(role="tool", parts=list(pending)))
            pending.clear()

    for i, m in enumerate(req.messages):
        if m.role == "tool":
            pending.append(
                types.Part.from_function_response(name=m.name or "", response={"result": m.content})
            )
            continue
        flush()
        if m.role == "assistant" and m.tool_calls:
            parts: list[Any] = []
            if m.content:
                parts.append(types.Part.from_text(text=m.content))
            # ⚠️ `thought_signature` is NOT decoration — replaying a Gemini 3 function call without
            # the signature it minted is a hard 400 on the NEXT request ("Function call is missing
            # a thought_signature in functionCall parts"), i.e. the tool loop dies on its second
            # iteration. Older models mint none; the field is then None and Gemini ignores it.
            parts += [
                types.Part(
                    function_call=types.FunctionCall(name=tc.name, args=tc.input),
                    thought_signature=tc.signature,
                )
                for tc in m.tool_calls
            ]
            out.append(types.Content(role="model", parts=parts))
        else:
            text = [types.Part.from_text(text=m.content)] if m.content else []
            if i == media_at and req.media:
                blobs = [
                    types.Part.from_bytes(data=p.data, mime_type=p.mime_type) for p in req.media
                ]
                # Media before the text when caching it (see _common.media_first).
                parts = [*blobs, *text] if media_first(req.context_cache) else [*text, *blobs]
            else:
                parts = text
            out.append(
                types.Content(role="model" if m.role == "assistant" else "user", parts=parts)
            )
    flush()
    return out


def _response_parts(resp: Any) -> list[Any]:
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    return list(getattr(content, "parts", None) or [])


def _tool_calls(resp: Any) -> list[ToolCall]:
    """Gemini function calls carry no id — results match back by name.

    ⚠️ Read the **parts**, not the ``resp.function_calls`` convenience accessor. That accessor
    yields bare ``FunctionCall`` objects, and Gemini 3's thought signature is not on the call — it
    is on the ``Part`` that wraps it. Take the shortcut and the signature is silently unreachable,
    so the loop only fails one request later, at replay, with an error naming a field this function
    never saw. ``_contents`` puts it back.
    """
    calls: list[ToolCall] = []
    for part in _response_parts(resp):
        fc = getattr(part, "function_call", None)
        if fc is None:
            continue
        calls.append(
            ToolCall(
                id=getattr(fc, "id", None) or fc.name,
                name=fc.name,
                input=dict(fc.args or {}),
                signature=getattr(part, "thought_signature", None),
            )
        )
    return calls


def _text(resp: Any) -> str:
    try:
        return resp.text or ""
    except Exception:  # noqa: BLE001 - .text can raise when the turn is function-calls-only
        return ""


def _usage(um: Any) -> Usage:
    """``prompt_token_count`` is cached-**inclusive** on Gemini; maslul's convention is disjoint, so
    the cached count is subtracted back out (see :class:`~maslul.Usage`)."""
    if um is None:
        return Usage()
    billable, cached = split_input(
        getattr(um, "prompt_token_count", 0) or 0,
        getattr(um, "cached_content_token_count", 0) or 0,
    )
    return Usage(
        input_tokens=billable,
        output_tokens=getattr(um, "candidates_token_count", 0) or 0,
        cache_read_input_tokens=cached,
    )


def _finish_reason(resp: Any) -> str | None:
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return None
    fr = getattr(candidates[0], "finish_reason", None)
    if fr is None:
        return None
    return getattr(fr, "name", None) or str(fr)


def _sources(resp: Any) -> list[str]:
    """Unique grounding URLs from Google Search grounding metadata (web search citations)."""
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return []
    gm = getattr(candidates[0], "grounding_metadata", None)
    urls: list[str] = []
    for chunk in getattr(gm, "grounding_chunks", None) or []:
        uri = getattr(getattr(chunk, "web", None), "uri", None)
        if uri and uri not in urls:
            urls.append(uri)
    return urls


def _map_error(e: Exception) -> Exception:
    """Map a google-genai ``APIError`` to the :class:`~maslul.MaslulError` hierarchy by HTTP
    status code (``e.code``); anything without a recognized code becomes a ``ProviderError``."""
    code = getattr(e, "code", None)
    if code == 429:
        return RateLimited(str(e))
    if code == 408:
        return Timeout(str(e))
    if code in (401, 403):
        return AuthError(str(e))
    return ProviderError(str(e))
