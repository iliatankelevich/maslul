"""Gemini provider — Google ``google-genai`` SDK via Vertex AI.

Mirrors Kippy's auth pattern: Vertex AI + Application Default Credentials (no API key) when
``vertex_project`` is set; an API key path is supported for OSS users who prefer the Gemini
Developer API. M1 covers plain-text completion with normalized usage and finish reason.
"""

from __future__ import annotations

from typing import Any

from maslul.errors import ProviderError
from maslul.types import ModelSpec, Request, Response, Usage


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
            return
        from google import genai

        if vertex_project:
            self._client = genai.Client(
                vertexai=True, project=vertex_project, location=vertex_location
            )
        elif api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = genai.Client()  # resolve from ADC / environment

    async def complete(self, spec: ModelSpec, req: Request) -> Response:
        from google.genai import types

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
        try:
            resp = await self._client.aio.models.generate_content(
                model=spec.model,
                contents=_contents(req),
                config=types.GenerateContentConfig(**config) if config else None,
            )
        except Exception as e:  # noqa: BLE001 - normalized below
            raise ProviderError(str(e)) from e
        return Response(
            text=resp.text or "",
            level_used=None,
            provider=self.name,
            model=spec.model,
            usage=_usage(getattr(resp, "usage_metadata", None)),
            finish_reason=_finish_reason(resp),
            raw=resp,
        )

    async def healthcheck(self, spec: ModelSpec) -> None:
        await self._client.aio.models.generate_content(model=spec.model, contents="ping")


def _contents(req: Request) -> list[Any]:
    from google.genai import types

    return [
        types.Content(
            role="model" if m.role == "assistant" else "user",
            parts=[types.Part.from_text(text=m.content)],
        )
        for m in req.messages
    ]


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
