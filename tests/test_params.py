"""Generic generation parameters: resolution (request over tier), config parsing, and how each
provider translates them into its own SDK call. Hermetic — fake SDK clients only."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import types as gtypes
from xai_sdk.chat import chat_pb2

from maslul import ConfigError, Level, Message, ModelSpec, Request, RouterConfig, ToolDef
from maslul.providers._common import generation_params
from maslul.providers.anthropic import AnthropicProvider
from maslul.providers.gemini import GeminiProvider
from maslul.providers.openai import OpenAIProvider
from maslul.providers.xai import XAIProvider
from tests.test_providers_unit import (
    _cache_resp,
    _cached_req,
    _FakeAnthropic,
    _FakeGemini,
    _FakeGeminiWithCaches,
    _FakeOpenAI,
    _FakeXAI,
    _oai_resp,
)

_TOOL = ToolDef(name="lookup", description="look something up", input_schema={"type": "object"})


def _req(**over: Any) -> Request:
    return Request(messages=[Message(role="user", content="hello")], **over)


def _anthropic_resp() -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )


async def _anthropic_call(model: str, req: Request, **spec: Any) -> dict[str, Any]:
    fake = _FakeAnthropic(_anthropic_resp())
    await AnthropicProvider(client=fake).complete(
        ModelSpec(provider="anthropic", model=model, **spec), req
    )
    return fake.calls[0]


async def _openai_call(req: Request, **spec: Any) -> dict[str, Any]:
    fake = _FakeOpenAI(_oai_resp(content="ok"))
    await OpenAIProvider(client=fake).complete(
        ModelSpec(provider="openai", model="gpt-5.5", **spec), req
    )
    return fake.calls[0]


async def _gemini_config(req: Request, **spec: Any) -> Any:
    fake = _FakeGemini(SimpleNamespace(text="ok", usage_metadata=None, candidates=[]))
    await GeminiProvider(client=fake).complete(
        ModelSpec(provider="gemini", model="gemini-3.8-flash", **spec), req
    )
    return fake.calls[0]["config"]


async def _xai_call(req: Request, **spec: Any) -> dict[str, Any]:
    fake = _FakeXAI(SimpleNamespace(content="ok", usage=None, finish_reason=None))
    await XAIProvider(client=fake).complete(
        ModelSpec(provider="xai", model="grok-4.7", **spec), req
    )
    return fake.calls[0]


# --- resolution + config ---------------------------------------------------------------------


def test_request_values_beat_tier_defaults() -> None:
    spec = ModelSpec(provider="x", model="m", temperature=0.9, top_p=0.5, reasoning="high")
    params = generation_params(spec, _req(temperature=0.1))
    assert (params.temperature, params.top_p, params.reasoning) == (0.1, 0.5, "high")


def test_unknown_reasoning_level_and_unknown_tool_choice_raise() -> None:
    spec = ModelSpec(provider="x", model="m")
    with pytest.raises(ConfigError, match="reasoning"):
        generation_params(spec, _req(reasoning="max"))  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="names no tool"):
        generation_params(spec, _req(tools=[_TOOL], tool_choice="missing"))
    assert generation_params(spec, _req(tools=[_TOOL], tool_choice="lookup")).tool_choice


def test_tier_generic_keys_are_parsed_and_validated() -> None:
    tier = {
        "model": "anthropic:claude-sonnet-4-6",
        "temperature": 0,
        "top_p": 0.9,
        "top_k": 40,
        "seed": 7,
        "reasoning": "low",
    }
    spec = RouterConfig.from_dict({"tiers": {"hard": tier}}).tiers[Level.HARD]
    assert (spec.temperature, spec.top_p, spec.top_k, spec.seed) == (0.0, 0.9, 40, 7)
    assert spec.reasoning == "low"
    with pytest.raises(ConfigError, match="reasoning"):
        RouterConfig.from_dict({"tiers": {"hard": {**tier, "reasoning": "off"}}})


# --- Anthropic ---------------------------------------------------------------------------------


async def test_anthropic_sampling_maps_and_seed_is_ignored() -> None:
    call = await _anthropic_call("claude-sonnet-4-6", _req(temperature=0.2, top_k=40, seed=7))
    assert (call["temperature"], call["top_k"]) == (0.2, 40)
    assert "seed" not in call
    assert call["max_tokens"] == 1024


async def test_anthropic_reasoning_is_adaptive_thinking_plus_effort_and_drops_sampling() -> None:
    req = _req(reasoning="medium", temperature=0.2, response_format={"type": "object"})
    call = await _anthropic_call("claude-opus-5-5", req)
    assert call["thinking"] == {"type": "adaptive"}
    # effort and the structured-output format share output_config
    assert call["output_config"]["effort"] == "medium"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in call  # thinking refuses sampling parameters
    assert call["max_tokens"] == 16000  # thinking counts against max_tokens: a roomier default


async def test_anthropic_reasoning_uses_a_budget_before_claude_4_6() -> None:
    call = await _anthropic_call("claude-haiku-4-5", _req(reasoning="high"))
    assert call["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert "output_config" not in call  # effort errors on Haiku 4.5
    assert call["max_tokens"] > 8192  # the budget must stay below max_tokens


async def test_anthropic_tool_choice_and_scoped_provider_options() -> None:
    req = _req(
        tools=[_TOOL],
        tool_choice="lookup",
        provider_options={"anthropic": {"service_tier": "auto"}, "gemini": {"seed": 1}, "x": 1},
    )
    call = await _anthropic_call("claude-sonnet-4-6", req)
    assert call["tool_choice"] == {"type": "tool", "name": "lookup"}
    assert call["service_tier"] == "auto"  # its own scoped entry
    assert call["x"] == 1  # legacy unscoped keys still apply on Anthropic
    assert "gemini" not in call and "seed" not in call  # another provider's entry never leaks


async def test_tier_options_override_generic_values() -> None:
    call = await _anthropic_call("claude-sonnet-4-6", _req(temperature=0.2), options={"top_k": 5})
    assert call["top_k"] == 5
    call = await _anthropic_call(
        "claude-sonnet-4-6", _req(reasoning="low"), options={"thinking": {"type": "disabled"}}
    )
    assert call["thinking"] == {"type": "disabled"}  # raw always wins


# --- OpenAI ------------------------------------------------------------------------------------


async def test_openai_maps_sampling_seed_and_tool_choice() -> None:
    req = _req(
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        seed=7,
        stop=["END"],
        tools=[_TOOL],
        tool_choice="lookup",
    )
    call = await _openai_call(req)
    assert (call["temperature"], call["top_p"], call["seed"], call["stop"]) == (
        0.2,
        0.9,
        7,
        ["END"],
    )
    assert "top_k" not in call  # no OpenAI equivalent
    assert call["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}


async def test_openai_reasoning_effort_drops_sampling_and_raises_the_default_cap() -> None:
    call = await _openai_call(_req(reasoning="high", temperature=0.2, top_p=0.9, stop=["END"]))
    assert call["reasoning_effort"] == "high"
    assert not {"temperature", "top_p", "stop"} & call.keys()  # refused by reasoning models
    assert call["max_completion_tokens"] == 16000


# --- Gemini ------------------------------------------------------------------------------------


async def test_gemini_maps_sampling_and_reasoning_to_thinking_level() -> None:
    config = await _gemini_config(_req(temperature=0.2, top_k=40, seed=7, reasoning="low"))
    assert (config.temperature, config.top_k, config.seed) == (0.2, 40, 7)
    assert config.thinking_config.thinking_level == gtypes.ThinkingLevel.LOW


async def test_gemini_tool_choice_and_only_the_scoped_provider_options() -> None:
    req = _req(
        tools=[_TOOL],
        tool_choice="required",
        provider_options={"gemini": {"candidate_count": 1}, "system": [{"type": "text"}]},
    )
    config = await _gemini_config(req)
    mode = config.tool_config.function_calling_config.mode
    assert mode == gtypes.FunctionCallingConfigMode.ANY
    assert config.candidate_count == 1  # unscoped "system" was not applied (it would not validate)


async def test_gemini_auto_tool_choice_sends_no_tool_config() -> None:
    config = await _gemini_config(_req(tools=[_TOOL], tool_choice="auto"))
    assert config.tool_config is None


async def test_gemini_forced_tool_choice_skips_the_explicit_cache() -> None:
    fake = _FakeGeminiWithCaches(_cache_resp())
    req = _cached_req(tools=[_TOOL], tool_choice="required")
    await GeminiProvider(client=fake).complete(
        ModelSpec(provider="gemini", model="gemini-3.8-flash"), req
    )
    assert fake.caches.created == []  # Vertex refuses tool_config next to cached_content
    assert fake.calls[0]["config"].tool_config is not None


# --- xAI ---------------------------------------------------------------------------------------


async def test_xai_maps_sampling_and_ignores_top_k() -> None:
    call = await _xai_call(_req(temperature=0.2, top_p=0.9, top_k=40, seed=7, stop=["END"]))
    assert (call["temperature"], call["top_p"], call["seed"], call["stop"]) == (
        0.2,
        0.9,
        7,
        ["END"],
    )
    assert "top_k" not in call


async def test_xai_reasoning_effort_drops_stop() -> None:
    call = await _xai_call(_req(reasoning="medium", stop=["END"]))
    assert call["reasoning_effort"] == "medium"
    assert "stop" not in call  # reasoning models refuse it


async def test_xai_tool_choice_and_scoped_provider_options() -> None:
    req = _req(
        tools=[_TOOL],
        tool_choice="lookup",
        provider_options={"xai": {"parallel_tool_calls": False}, "system": "anthropic-shaped"},
    )
    call = await _xai_call(req)
    assert call["tool_choice"] == chat_pb2.ToolChoice(function_name="lookup")
    assert call["parallel_tool_calls"] is False
    assert "system" not in call
