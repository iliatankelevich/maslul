"""Live provider smoke tests — make one real, cheap API call per provider.

Each is gated on its credentials and skips cleanly when absent (like Kippy's ``make doctor``),
so the default ``pytest`` run is hermetic on a machine with no keys. Override the model per
provider with the ``MASLUL_*_MODEL`` env vars when the catalog moves.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from maslul import Level, Router
from maslul.types import Message, ModelSpec, Request, ToolDef

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


_ADD_TOOL = ToolDef(
    name="add",
    description="Add two integers a and b.",
    input_schema={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
)


async def _calculator_round_trip(provider_name: str, provider: Any, model: str) -> None:
    """End-to-end M2: the router drives a real calculator tool round-trip through ``provider``."""
    config = {
        "maslul": {
            "default_level": "hard",
            "tiers": {"hard": {"provider": provider_name, "model": model}},
        }
    }
    router = Router(config, providers={provider_name: provider})

    calls: list[str] = []

    async def add(call: Any) -> str:  # noqa: ANN401 - ToolCall, kept loose for the test
        calls.append(call.name)
        return str(call.input["a"] + call.input["b"])

    req = Request(
        messages=[
            Message(
                role="user",
                content="Use the add tool to compute 21 + 21, then state the resulting number.",
            )
        ],
        tools=[_ADD_TOOL],
        tool_executor=add,
        max_tokens=256,
    )
    resp = await router.complete(req, level=Level.HARD)
    assert calls == ["add"]
    assert "42" in resp.text
    assert any(c.name == "add" for c in resp.tool_calls)


@requires_anthropic
async def test_anthropic_tool_loop_live() -> None:
    from maslul.providers.anthropic import AnthropicProvider

    model = os.getenv("MASLUL_ANTHROPIC_MODEL", "claude-haiku-4-5")
    await _calculator_round_trip("anthropic", AnthropicProvider(), model)


@requires_grok
async def test_grok_tool_loop_live() -> None:
    # Exercises the stateless reconstruction of the assistant tool-call turn (the M2 risk).
    from maslul.providers.grok import GrokProvider

    model = os.getenv("MASLUL_GROK_MODEL", "grok-4.3")
    await _calculator_round_trip("grok", GrokProvider(), model)
