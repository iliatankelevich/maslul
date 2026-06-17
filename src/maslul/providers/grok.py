"""Grok provider — official xAI ``xai-sdk`` (gRPC; §8 resolved decision).

The router owns the loop and rebuilds the conversation from normalized messages each turn, so
this provider reconstructs the xAI chat statelessly: prior ``assistant`` tool-call turns become
``chat_pb2.Message`` protos with ``tool_calls``, and tool results use
``tool_result(content, tool_call_id=...)`` (id-matched).

Verified live against ``grok-4.3`` — text + usage, and a calculator tool round-trip that
exercises the reconstructed assistant tool-call turn (see
``tests/integration/test_providers_live.py``).

Importing this module requires the ``grok`` extra (``pip install maslul[grok]``).
"""

from __future__ import annotations

import base64
import json
from typing import Any

import grpc
from xai_sdk import AsyncClient
from xai_sdk.chat import assistant, chat_pb2, image, system, tool, tool_result, user
from xai_sdk.tools import web_search

from maslul.errors import AuthError, ProviderError, RateLimited, Timeout
from maslul.providers._common import last_user_index
from maslul.types import MediaPart, Message, ModelSpec, Request, Response, ToolCall, Usage


class GrokProvider:
    """Async Grok backend. Satisfies the :class:`~maslul.Provider` protocol."""

    name = "grok"

    def __init__(self, *, api_key: str | None = None, client: Any | None = None) -> None:
        self._client: Any = client or (AsyncClient(api_key=api_key) if api_key else AsyncClient())

    async def complete(self, spec: ModelSpec, req: Request) -> Response:
        messages = [system(s) for s in (req.system or [])] + _to_messages(req.messages, req.media)
        kwargs: dict[str, Any] = {"model": spec.model, "messages": messages}
        # Client function tools + the server-side web_search tool (xAI Agent Tools API; the older
        # SearchParameters "Live Search" is deprecated/removed). Both are chat_pb2.Tool entries.
        tools = [
            tool(name=t.name, description=t.description, parameters=t.input_schema)
            for t in (req.tools or [])
        ]
        if req.web_search:
            tools.append(web_search())
        if tools:
            kwargs["tools"] = tools
        if req.response_format is not None:
            kwargs["response_format"] = chat_pb2.ResponseFormat(
                format_type=chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA,
                schema=json.dumps(req.response_format),
            )
        max_tokens = req.max_tokens or spec.max_tokens
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        try:
            chat = self._client.chat.create(**kwargs)
            resp = await chat.sample()
        except Exception as e:  # noqa: BLE001 - normalized below
            raise _map_error(e) from e
        return Response(
            text=getattr(resp, "content", "") or "",
            level_used=None,
            provider=self.name,
            model=spec.model,
            usage=_usage(getattr(resp, "usage", None)),
            tool_calls=_tool_calls(resp),
            finish_reason=_finish_reason(resp),
            sources=_sources(resp),
            raw=resp,
        )

    async def healthcheck(self, spec: ModelSpec) -> None:
        chat = self._client.chat.create(model=spec.model, messages=[user("ping")])
        await chat.sample()


def _to_messages(messages: list[Message], media: list[MediaPart] | None) -> list[Any]:
    media_at = last_user_index(messages) if media else -1
    out: list[Any] = []
    for i, m in enumerate(messages):
        if m.role == "tool":
            out.append(tool_result(m.content, tool_call_id=m.tool_call_id))
        elif m.role == "assistant" and m.tool_calls:
            msg = assistant(m.content) if m.content else assistant("")
            for tc in m.tool_calls:
                msg.tool_calls.append(
                    chat_pb2.ToolCall(
                        id=tc.id,
                        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
                        function=chat_pb2.FunctionCall(
                            name=tc.name, arguments=json.dumps(tc.input)
                        ),
                    )
                )
            out.append(msg)
        elif m.role == "assistant":
            out.append(assistant(m.content))
        elif i == media_at and media:
            args: list[Any] = [m.content] if m.content else []
            args += [image(_data_url(p)) for p in media]
            out.append(user(*args))
        else:
            out.append(user(m.content))
    return out


def _data_url(part: MediaPart) -> str:
    return f"data:{part.mime_type};base64,{base64.standard_b64encode(part.data).decode()}"


def _tool_calls(resp: Any) -> list[ToolCall]:
    out: list[ToolCall] = []
    for tc in getattr(resp, "tool_calls", None) or []:
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
    if u is None:
        return Usage()
    return Usage(
        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cached_prompt_text_tokens", 0) or 0,
    )


def _finish_reason(resp: Any) -> str | None:
    fr = getattr(resp, "finish_reason", None)
    if fr is None:
        return None
    return getattr(fr, "name", None) or str(fr)


def _sources(resp: Any) -> list[str]:
    """Unique citation URLs from xAI Live Search (returned when return_citations is set)."""
    out: list[str] = []
    for c in getattr(resp, "citations", None) or []:
        url = c if isinstance(c, str) else (getattr(c, "url", None) or getattr(c, "uri", None))
        if url and url not in out:
            out.append(url)
    return out


def _map_error(e: Exception) -> Exception:
    """Normalize a gRPC error to a :class:`~maslul.MaslulError`; pass others through."""
    if isinstance(e, grpc.aio.AioRpcError):
        code = e.code()
        if code in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED):
            return AuthError(str(e))
        if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            return RateLimited(str(e))
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return Timeout(str(e))
        return ProviderError(str(e))
    return e
