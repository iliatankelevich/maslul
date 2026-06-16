"""Hermetic provider tests — inject a fake SDK client and assert the normalization logic
(text, usage, finish_reason) without any network call or credentials. These exercise the
real message-builder + response-mapping code in each provider; only the transport is faked.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from maslul.errors import ConfigError
from maslul.providers import build_provider
from maslul.providers.anthropic import AnthropicProvider
from maslul.providers.gemini import GeminiProvider
from maslul.providers.grok import GrokProvider
from maslul.types import Message, ModelSpec, Request


def _req() -> Request:
    return Request(messages=[Message(role="user", content="hello")], system=["be terse"])


# --- Anthropic ---------------------------------------------------------------------------


class _FakeAnthropic:
    def __init__(self, resp: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)
        self._resp = resp

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._resp


async def test_anthropic_normalizes_text_usage_and_finish_reason() -> None:
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="…"),
            SimpleNamespace(type="text", text="hi there"),
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=2,
            cache_creation_input_tokens=1,
        ),
        stop_reason="end_turn",
    )
    fake = _FakeAnthropic(resp)
    out = await AnthropicProvider(client=fake).complete(
        ModelSpec(provider="anthropic", model="claude-haiku-4-5"), _req()
    )
    assert out.text == "hi there"  # only text blocks
    assert out.provider == "anthropic"
    assert (out.usage.input_tokens, out.usage.output_tokens) == (10, 5)
    assert out.usage.cache_read_input_tokens == 2
    assert out.finish_reason == "end_turn"
    # system is folded into a single string; messages are mapped to role/content dicts
    assert fake.calls[0]["system"] == "be terse"
    assert fake.calls[0]["messages"] == [{"role": "user", "content": "hello"}]


# --- Gemini ------------------------------------------------------------------------------


class _FakeGemini:
    def __init__(self, resp: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._gen))
        self._resp = resp

    async def _gen(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._resp


async def test_gemini_normalizes_text_usage_and_finish_reason() -> None:
    resp = SimpleNamespace(
        text="hi from gemini",
        usage_metadata=SimpleNamespace(
            prompt_token_count=7,
            candidates_token_count=3,
            cached_content_token_count=1,
        ),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
    )
    out = await GeminiProvider(client=_FakeGemini(resp)).complete(
        ModelSpec(provider="gemini", model="gemini-2.5-flash"), _req()
    )
    assert out.text == "hi from gemini"
    assert out.provider == "gemini"
    assert (out.usage.input_tokens, out.usage.output_tokens) == (7, 3)
    assert out.usage.cache_read_input_tokens == 1
    assert out.finish_reason == "STOP"


# --- Grok --------------------------------------------------------------------------------


class _FakeGrokChat:
    def __init__(self, resp: Any) -> None:
        self._resp = resp

    async def sample(self) -> Any:
        return self._resp


class _FakeGrok:
    def __init__(self, resp: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(create=self._create)
        self._resp = resp

    def _create(self, **kwargs: Any) -> _FakeGrokChat:
        self.calls.append(kwargs)
        return _FakeGrokChat(self._resp)


async def test_grok_normalizes_text_usage_and_finish_reason() -> None:
    resp = SimpleNamespace(
        content="hi from grok",
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, cached_prompt_text_tokens=0),
        finish_reason="stop",
    )
    fake = _FakeGrok(resp)
    out = await GrokProvider(client=fake).complete(
        ModelSpec(provider="grok", model="grok-4.3"), _req()
    )
    assert out.text == "hi from grok"
    assert out.provider == "grok"
    assert (out.usage.input_tokens, out.usage.output_tokens) == (4, 2)
    assert out.finish_reason == "stop"
    assert fake.calls[0]["model"] == "grok-4.3"


# --- build_provider factory --------------------------------------------------------------


async def test_build_provider_dispatches_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # async so a running event loop exists — xai_sdk.AsyncClient grabs it at construction.
    monkeypatch.setenv("MASLUL_DUMMY_KEY", "dummy")
    cfg = {"api_key_env": "MASLUL_DUMMY_KEY"}
    assert isinstance(build_provider("anthropic", cfg), AnthropicProvider)
    assert isinstance(build_provider("gemini", cfg), GeminiProvider)
    assert isinstance(build_provider("grok", cfg), GrokProvider)


def test_build_provider_unknown_raises() -> None:
    with pytest.raises(ConfigError):
        build_provider("openai", {})
