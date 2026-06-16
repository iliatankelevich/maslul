"""Grok provider — official xAI ``xai-sdk`` (gRPC; §8 resolved decision).

The SDK exposes a stateful chat: ``client.chat.create(...)`` → ``chat.append(...)`` →
``await chat.sample()``. M1 covers plain-text completion.

⚠️ **Unverified-live:** the SDK docs don't expose the exact ``usage`` / ``finish_reason``
field names, and no ``XAI_API_KEY`` was available to smoke-test. The text path is solid; the
usage/finish-reason mapping is defensive (``getattr`` with sensible fallbacks) and must be
confirmed against a live response — see ``tests/integration/test_providers_live.py``.
"""

from __future__ import annotations

from typing import Any

from maslul.errors import AuthError, ProviderError, RateLimited, Timeout
from maslul.types import ModelSpec, Request, Response, Usage


class GrokProvider:
    """Async Grok backend. Satisfies the :class:`~maslul.Provider` protocol."""

    name = "grok"

    def __init__(self, *, api_key: str | None = None, client: Any | None = None) -> None:
        if client is not None:
            self._client: Any = client
            return
        from xai_sdk import AsyncClient

        self._client = AsyncClient(api_key=api_key) if api_key else AsyncClient()

    async def complete(self, spec: ModelSpec, req: Request) -> Response:
        from xai_sdk.chat import assistant, system, user

        messages = [system(s) for s in (req.system or [])]
        messages += [
            assistant(m.content) if m.role == "assistant" else user(m.content) for m in req.messages
        ]
        kwargs: dict[str, Any] = {"model": spec.model, "messages": messages}
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
            finish_reason=_finish_reason(resp),
            raw=resp,
        )

    async def healthcheck(self, spec: ModelSpec) -> None:
        from xai_sdk.chat import user

        chat = self._client.chat.create(model=spec.model, messages=[user("ping")])
        await chat.sample()


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


def _map_error(e: Exception) -> Exception:
    """Normalize a gRPC error to a :class:`~maslul.MaslulError`; pass others through."""
    try:
        import grpc
    except ImportError:
        return e
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
