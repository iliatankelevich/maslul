"""Live provider smoke tests — make one real, cheap API call per provider.

Each is gated on its credentials and skips cleanly when absent (like Kippy's ``make doctor``),
so the default ``pytest`` run is hermetic on a machine with no keys. Override the model per
provider with the ``MASLUL_*_MODEL`` env vars when the catalog moves.
"""

from __future__ import annotations

import os

import pytest

from maslul.types import Message, ModelSpec, Request

requires_anthropic = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
requires_gemini = pytest.mark.skipif(
    not (os.getenv("MASLUL_VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")),
    reason="no Vertex project configured (set MASLUL_VERTEX_PROJECT)",
)
requires_grok = pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")


def _prompt() -> Request:
    return Request(
        messages=[Message(role="user", content="Reply with exactly the word: pong")],
        max_tokens=16,
    )


@requires_anthropic
async def test_anthropic_live() -> None:
    from maslul.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider()
    spec = ModelSpec(
        provider="anthropic", model=os.getenv("MASLUL_ANTHROPIC_MODEL", "claude-haiku-4-5")
    )
    resp = await provider.complete(spec, _prompt())
    assert resp.text.strip()
    assert resp.usage.output_tokens > 0
    assert resp.provider == "anthropic"
    await provider.healthcheck(spec)


@requires_gemini
async def test_gemini_live() -> None:
    from maslul.providers.gemini import GeminiProvider

    project = os.getenv("MASLUL_VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    provider = GeminiProvider(
        vertex_project=project, vertex_location=os.getenv("MASLUL_VERTEX_LOCATION", "global")
    )
    spec = ModelSpec(provider="gemini", model=os.getenv("MASLUL_GEMINI_MODEL", "gemini-2.5-flash"))
    resp = await provider.complete(spec, _prompt())
    assert resp.text.strip()
    assert resp.usage.output_tokens > 0
    await provider.healthcheck(spec)


@requires_grok
async def test_grok_live() -> None:
    from maslul.providers.grok import GrokProvider

    provider = GrokProvider()
    spec = ModelSpec(provider="grok", model=os.getenv("MASLUL_GROK_MODEL", "grok-4.3"))
    resp = await provider.complete(spec, _prompt())
    assert resp.text.strip()
    assert resp.usage.output_tokens > 0
    await provider.healthcheck(spec)
