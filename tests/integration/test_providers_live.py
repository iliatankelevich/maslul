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
from maslul.types import MediaPart, Message, ModelSpec, Request, ToolDef

requires_anthropic = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
requires_gemini = pytest.mark.skipif(
    not (os.getenv("MASLUL_VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")),
    reason="no Vertex project configured (set MASLUL_VERTEX_PROJECT)",
)
requires_grok = pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")


def _prompt() -> Request:
    # Headroom matters: thinking models (e.g. gemini-2.5-flash) spend tokens reasoning before
    # any visible text, so too small a budget yields an empty MAX_TOKENS response.
    return Request(
        messages=[Message(role="user", content="Reply with exactly the word: pong")],
        max_tokens=512,
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


def _router(provider_name: str, provider: Any, model: str) -> Router:
    config = {
        "maslul": {
            "default_level": "hard",
            "tiers": {"hard": {"provider": provider_name, "model": model}},
        }
    }
    return Router(config, providers={provider_name: provider})


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
    router = _router(provider_name, provider, model)

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


# --- M3: structured output + vision ------------------------------------------------------

_CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
    "additionalProperties": False,
}


def _solid_png(rgb: tuple[int, int, int], size: int = 32) -> bytes:
    """A valid solid-color RGB PNG, built with stdlib only (no Pillow) for the vision test.

    32x32 (1024px) clears xAI's 512-pixel minimum; Anthropic/Gemini accept smaller too."""
    import struct
    import zlib

    def chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit, color type 2 (RGB)
    raw = (b"\x00" + bytes(rgb) * size) * size  # each row: filter byte 0 + pixels
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


async def _structured_round_trip(provider_name: str, provider: Any, model: str) -> None:
    router = _router(provider_name, provider, model)
    req = Request(
        messages=[
            Message(role="user", content="Extract the city and country from: Paris is in France.")
        ],
        response_format=_CITY_SCHEMA,
        max_tokens=512,
    )
    resp = await router.complete(req, level=Level.HARD)
    assert isinstance(resp.structured, dict), f"no structured output: {resp.text!r}"
    blob = str(resp.structured).lower()
    assert "paris" in blob and "france" in blob


async def _vision_round_trip(provider_name: str, provider: Any, model: str) -> None:
    router = _router(provider_name, provider, model)
    req = Request(
        messages=[Message(role="user", content="What color fills this image? One word.")],
        media=[MediaPart(mime_type="image/png", data=_solid_png((0, 0, 255)))],
        max_tokens=512,
    )
    resp = await router.complete(req, level=Level.HARD)
    assert "blue" in resp.text.lower(), f"expected 'blue', got: {resp.text!r}"


@requires_anthropic
async def test_anthropic_structured_live() -> None:
    from maslul.providers.anthropic import AnthropicProvider

    await _structured_round_trip(
        "anthropic", AnthropicProvider(), os.getenv("MASLUL_ANTHROPIC_MODEL", "claude-haiku-4-5")
    )


@requires_anthropic
async def test_anthropic_vision_live() -> None:
    from maslul.providers.anthropic import AnthropicProvider

    await _vision_round_trip(
        "anthropic", AnthropicProvider(), os.getenv("MASLUL_ANTHROPIC_MODEL", "claude-haiku-4-5")
    )


@requires_gemini
async def test_gemini_structured_live() -> None:
    from maslul.providers.gemini import GeminiProvider

    project = os.getenv("MASLUL_VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    provider = GeminiProvider(vertex_project=project)
    await _structured_round_trip(
        "gemini", provider, os.getenv("MASLUL_GEMINI_MODEL", "gemini-2.5-flash")
    )


@requires_gemini
async def test_gemini_vision_live() -> None:
    from maslul.providers.gemini import GeminiProvider

    project = os.getenv("MASLUL_VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    provider = GeminiProvider(vertex_project=project)
    await _vision_round_trip(
        "gemini", provider, os.getenv("MASLUL_GEMINI_MODEL", "gemini-2.5-flash")
    )


@requires_grok
async def test_grok_structured_live() -> None:
    from maslul.providers.grok import GrokProvider

    await _structured_round_trip("grok", GrokProvider(), os.getenv("MASLUL_GROK_MODEL", "grok-4.3"))


@requires_grok
async def test_grok_vision_live() -> None:
    from maslul.providers.grok import GrokProvider

    await _vision_round_trip("grok", GrokProvider(), os.getenv("MASLUL_GROK_MODEL", "grok-4.3"))
