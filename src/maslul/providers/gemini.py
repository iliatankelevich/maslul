"""Gemini provider — Google ``google-genai`` SDK via Vertex AI.

Auth follows the Vertex pattern: Application Default Credentials (no API key) when
``vertex_project`` is set; an API-key path is supported for the Gemini Developer API. Covers
plain-text completion (M1) and tool-use translation (M2): function declarations, the
``function_call``/``function_response`` round-trip, normalized usage, and finish reason.

Importing this module requires the ``gemini`` extra (``pip install maslul[gemini]``).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import time
import uuid
from collections import OrderedDict
from typing import Any

from google import genai
from google.genai import types

from maslul.errors import AuthError, ProviderError, RateLimited, Timeout
from maslul.providers._common import (
    generation_params,
    last_user_index,
    media_first,
    media_index,
    raw_options,
    split_input,
)
from maslul.types import ContextCache, ModelSpec, Request, Response, ToolCall, Usage

log = logging.getLogger(__name__)

# How many cache handles one provider instance keeps. Each entry is a short string, so the bound is
# about server-side hygiene (we delete on eviction), not memory.
_MAX_HANDLES = 32

# Vertex's documented stand-in for a functionCall that has no signature of its own: history another
# provider wrote (a fallback that resumes a tool loop), or a call the caller built. Gemini 3 refuses
# such a call with a 400 when it opens a step of the current turn. The value skips that check at
# some cost to the answer, so it goes only where the alternative is the 400. The SDK sends bytes as
# base64url, so these are the bytes whose encoding is the documented string.
_SKIP_SIGNATURE = base64.urlsafe_b64decode("skip_thought_signature_validator")
_GEMINI_MAJOR = re.compile(r"(?:^|/)gemini-(\d+)")


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
        # key -> (cache resource name, local expiry). Bounded; see _MAX_HANDLES.
        self._handles: OrderedDict[str, tuple[str, float]] = OrderedDict()

    async def complete(self, spec: ModelSpec, req: Request) -> Response:
        params = generation_params(spec, req)
        config: dict[str, Any] = {}
        if req.system:
            config["system_instruction"] = "\n\n".join(req.system)
        if params.max_tokens:
            config["max_output_tokens"] = params.max_tokens
        for key in ("temperature", "top_p", "top_k", "seed"):
            value = getattr(params, key)
            if value is not None:
                config[key] = value
        if params.stop:
            config["stop_sequences"] = params.stop
        if params.reasoning is not None:
            # Gemini 3's thinking_level. Gemini 2.5 takes a thinking_budget instead — set it raw,
            # in the tier's options (thinking_config = { thinking_budget = ... }).
            config["thinking_config"] = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel[params.reasoning.upper()]
            )
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
        if req.tools and params.tool_choice not in (None, "auto"):  # auto is Gemini's default
            config["tool_config"] = _tool_config(params.tool_choice)
        # Raw options override the generic ones above: the tier's GenerateContentConfig fields,
        # then the request's provider_options — only its "gemini" entry, since callers put
        # Anthropic-shaped values in the unscoped form, which a mixed-provider router can send here.
        config.update(spec.options)
        config.update(raw_options(self.name, req.provider_options, unscoped=False))
        # Phase 4: explicit CachedContent. Opt-in — only when the caller declared a TTL, because
        # this is the one path that creates server-side state. Everything stable (system + ALL
        # tools) moves into the cache: Vertex rejects a request that sets `cached_content` together
        # with `tools` or `system_instruction` ("Tool config, tools and system instruction should
        # not be set"), so it is all-or-nothing, not a split.
        cached_name = await self._explicit_cache(spec, req, config)
        if cached_name is not None:
            config = {k: v for k, v in config.items() if k not in ("system_instruction", "tools")}
            config["cached_content"] = cached_name
        try:
            resp = await self._client.aio.models.generate_content(
                model=spec.model,
                contents=_contents(req, spec.model),
                config=types.GenerateContentConfig(**config) if config else None,
            )
        except Exception as e:  # noqa: BLE001 - normalized below
            # A cache can expire server-side between our TTL bookkeeping and this call. That is a
            # race we caused, so pay for it once: forget the handle and retry uncached rather than
            # surfacing an error the caller cannot act on.
            if cached_name is not None and _is_missing_cache(e):
                self._forget(cached_name)
                log.debug("gemini cache %s vanished; retrying uncached", cached_name)
                retry = {k: v for k, v in config.items() if k != "cached_content"}
                if req.system:
                    retry["system_instruction"] = "\n\n".join(req.system)
                if tools:
                    retry["tools"] = tools
                try:
                    resp = await self._client.aio.models.generate_content(
                        model=spec.model,
                        contents=_contents(req, spec.model),
                        config=types.GenerateContentConfig(**retry) if retry else None,
                    )
                except Exception as e2:  # noqa: BLE001 - normalized below
                    raise _map_error(e2) from e2
            else:
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

    async def _explicit_cache(
        self, spec: ModelSpec, req: Request, config: dict[str, Any]
    ) -> str | None:
        """The resource name of a ``CachedContent`` holding this request's stable prefix, or None.

        None means "send the request the ordinary way" and is the answer for every failure mode:
        not opted in, nothing stable to cache, the prefix is under the model's minimum, quota,
        a transient API error. **A caching problem must never become the caller's problem** — the
        request still succeeds, it just costs full price, which is exactly how the Anthropic
        provider treats a below-minimum breakpoint.
        """
        cc: ContextCache | None = req.context_cache
        # ttl_seconds is the opt-in. `system=True` alone is a layout hint that costs nothing;
        # creating a server-side object is a different kind of promise, so it needs asking for.
        if cc is None or not cc.ttl_seconds or not cc.system:
            return None
        if "system_instruction" not in config and "tools" not in config:
            return None  # nothing stable to put in it
        if "tool_config" in config:
            # Vertex refuses tool_config next to cached_content, and a forced tool choice is
            # per-turn (the router resets it after the first turn), so caching it would mint a
            # new CachedContent per turn. Send this one uncached.
            return None

        key = cc.key or _cache_key(spec.model, config)
        now = time.monotonic()
        hit = self._handles.get(key)
        if hit is not None:
            name, expires = hit
            # Re-check with a margin: a handle that expires mid-flight costs a failed call and a
            # retry, and the margin is far cheaper than the round trip.
            if expires - now > 5.0:
                self._handles.move_to_end(key)
                return name
            self._handles.pop(key, None)

        create: dict[str, Any] = {"ttl": f"{int(cc.ttl_seconds)}s"}
        if "system_instruction" in config:
            create["system_instruction"] = config["system_instruction"]
        if "tools" in config:
            create["tools"] = config["tools"]
        try:
            created = await self._client.aio.caches.create(
                model=spec.model, config=types.CreateCachedContentConfig(**create)
            )
        except Exception as exc:  # noqa: BLE001 - caching is best-effort by design
            # The commonest case by far is a prefix under the model's minimum (4,096 tokens on the
            # 3.x line), which the API refuses with a 400 naming the number. Debug, not warning:
            # it is a normal outcome for a small prompt, not a fault.
            log.debug("gemini explicit cache unavailable (%s); sending uncached", exc)
            return None

        name = getattr(created, "name", None)
        if not name:
            return None
        self._handles[key] = (name, now + float(cc.ttl_seconds))
        self._handles.move_to_end(key)
        while len(self._handles) > _MAX_HANDLES:
            _, (evicted, _) = self._handles.popitem(last=False)
            await self._delete(evicted)
        return name

    def _forget(self, name: str) -> None:
        for key, (handle, _) in list(self._handles.items()):
            if handle == name:
                self._handles.pop(key, None)

    async def _delete(self, name: str) -> None:
        """Best-effort server-side delete. A leaked cache expires on its own TTL, so a failure here
        is a tidiness problem, never a correctness one."""
        try:
            await self._client.aio.caches.delete(name=name)
        except Exception as exc:  # noqa: BLE001 - tidiness only
            log.debug("gemini cache %s could not be deleted (%s); it will expire", name, exc)

    async def healthcheck(self, spec: ModelSpec) -> None:
        await self._client.aio.models.generate_content(model=spec.model, contents="ping")


def _contents(req: Request, model: str) -> list[Any]:
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
    # A turn opens at the latest user message; tool results are role "tool", not "user".
    current_turn = last_user_index(req.messages) if _requires_signatures(model) else None
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
            signatures = [tc.signature for tc in m.tool_calls]
            # Only the first call of a step carries one, even when the model called several.
            if current_turn is not None and i > current_turn and signatures[0] is None:
                signatures[0] = _SKIP_SIGNATURE
            parts += [
                types.Part(
                    function_call=types.FunctionCall(name=tc.name, args=tc.input),
                    thought_signature=signature,
                )
                for tc, signature in zip(m.tool_calls, signatures, strict=True)
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


def _requires_signatures(model: str) -> bool:
    match = _GEMINI_MAJOR.search(model)
    return match is not None and int(match.group(1)) >= 3


def _response_parts(resp: Any) -> list[Any]:
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    return list(getattr(content, "parts", None) or [])


def _tool_calls(resp: Any) -> list[ToolCall]:
    """Gemini function calls usually carry no id — results match back by name. A minted id is
    unique per call anyway: a fallback that resumes the tool loop replays this transcript to a
    provider that matches results by id, and there two calls of one tool must differ.

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
                id=getattr(fc, "id", None) or f"{fc.name}-{uuid.uuid4().hex[:12]}",
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


def _tool_config(choice: str) -> types.ToolConfig:
    """Generic tool choice → Gemini's function-calling mode; a tool name is ANY, limited to it."""
    if choice == "none":
        mode = types.FunctionCallingConfigMode.NONE
        return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode))
    names = None if choice == "required" else [choice]
    return types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY, allowed_function_names=names
        )
    )


def _cache_key(model: str, config: dict[str, Any]) -> str:
    """Identity of a cacheable prefix: the model plus everything that goes inside the cache.

    The model belongs in the key because a ``CachedContent`` is model-scoped — reusing one across
    an escalation would 400. The tools belong in it because ``web_search`` changes the cached tool
    set, and two chats that differ only by whether search is on must not share a handle.
    """
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(str(config.get("system_instruction", "")).encode())
    for tool in config.get("tools") or []:
        h.update(b"\x00")
        h.update(_tool_fingerprint(tool).encode())
    return h.hexdigest()


def _tool_fingerprint(tool: Any) -> str:
    """A stable string for one Gemini ``Tool``.

    Uses the SDK's own serialization when available and falls back to ``repr`` — a fingerprint only
    has to be stable and collision-free, not readable.
    """
    for attr in ("model_dump_json", "to_json_dict"):
        dump = getattr(tool, attr, None)
        if callable(dump):
            try:
                out = dump()
                return out if isinstance(out, str) else json.dumps(out, sort_keys=True, default=str)
            except Exception:  # noqa: BLE001 - fall through to repr
                pass
    return repr(tool)


def _is_missing_cache(exc: Exception) -> bool:
    """Whether an error says the cache we referenced is gone (expired or deleted elsewhere)."""
    text = str(exc).lower()
    return "cachedcontent" in text.replace(" ", "") or ("not found" in text and "cache" in text)
