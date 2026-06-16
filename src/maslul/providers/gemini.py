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

from maslul.errors import ProviderError
from maslul.providers._common import last_user_index
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
        if req.tools:
            config["tools"] = [
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
            ]
        try:
            resp = await self._client.aio.models.generate_content(
                model=spec.model,
                contents=_contents(req),
                config=types.GenerateContentConfig(**config) if config else None,
            )
        except Exception as e:  # noqa: BLE001 - normalized below
            raise ProviderError(str(e)) from e
        return Response(
            text=_text(resp),
            level_used=None,
            provider=self.name,
            model=spec.model,
            usage=_usage(getattr(resp, "usage_metadata", None)),
            tool_calls=_tool_calls(resp),
            finish_reason=_finish_reason(resp),
            raw=resp,
        )

    async def healthcheck(self, spec: ModelSpec) -> None:
        await self._client.aio.models.generate_content(model=spec.model, contents="ping")


def _contents(req: Request) -> list[Any]:
    media_at = last_user_index(req.messages) if req.media else -1
    out: list[Any] = []
    for i, m in enumerate(req.messages):
        if m.role == "tool":
            out.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=m.name or "", response={"result": m.content}
                        )
                    ],
                )
            )
        elif m.role == "assistant" and m.tool_calls:
            parts: list[Any] = []
            if m.content:
                parts.append(types.Part.from_text(text=m.content))
            parts += [
                types.Part(function_call=types.FunctionCall(name=tc.name, args=tc.input))
                for tc in m.tool_calls
            ]
            out.append(types.Content(role="model", parts=parts))
        else:
            parts = [types.Part.from_text(text=m.content)] if m.content else []
            if i == media_at and req.media:
                parts += [
                    types.Part.from_bytes(data=p.data, mime_type=p.mime_type) for p in req.media
                ]
            out.append(
                types.Content(role="model" if m.role == "assistant" else "user", parts=parts)
            )
    return out


def _tool_calls(resp: Any) -> list[ToolCall]:
    # Gemini function calls carry no id — match results back by name.
    return [
        ToolCall(id=getattr(fc, "id", None) or fc.name, name=fc.name, input=dict(fc.args or {}))
        for fc in (getattr(resp, "function_calls", None) or [])
    ]


def _text(resp: Any) -> str:
    try:
        return resp.text or ""
    except Exception:  # noqa: BLE001 - .text can raise when the turn is function-calls-only
        return ""


def _usage(um: Any) -> Usage:
    if um is None:
        return Usage()
    return Usage(
        input_tokens=getattr(um, "prompt_token_count", 0) or 0,
        output_tokens=getattr(um, "candidates_token_count", 0) or 0,
        cache_read_input_tokens=getattr(um, "cached_content_token_count", 0) or 0,
    )


def _finish_reason(resp: Any) -> str | None:
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return None
    fr = getattr(candidates[0], "finish_reason", None)
    if fr is None:
        return None
    return getattr(fr, "name", None) or str(fr)
