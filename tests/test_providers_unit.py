"""Hermetic provider tests — inject a fake SDK client and assert the normalization logic
(text, usage, finish_reason) without any network call or credentials. These exercise the
real message-builder + response-mapping code in each provider; only the transport is faked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from xai_sdk.chat import chat_pb2

from maslul.errors import ConfigError
from maslul.providers import build_provider
from maslul.providers.anthropic import AnthropicProvider
from maslul.providers.gemini import GeminiProvider
from maslul.providers.grok import GrokProvider
from maslul.types import MediaPart, Message, ModelSpec, Request, ToolCall, ToolDef


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


# --- tool-use translation (M2) -----------------------------------------------------------
# Each asserts both directions: parsing a tool-call response into Response.tool_calls, and
# translating an assistant tool-call turn + tool result back into the SDK's shape.

_TOOL = ToolDef(name="add", description="add", input_schema={"type": "object"})
_HISTORY = [
    Message(role="user", content="2+3?"),
    Message(role="assistant", tool_calls=[ToolCall(id="c1", name="add", input={"a": 2, "b": 3})]),
    Message(role="tool", content="5", tool_call_id="c1", name="add"),
]


async def test_anthropic_tool_use_translation() -> None:
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="c1", name="add", input={"a": 2, "b": 3})],
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=1,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        stop_reason="tool_use",
    )
    fake = _FakeAnthropic(resp)
    provider = AnthropicProvider(client=fake)
    spec = ModelSpec(provider="anthropic", model="claude-haiku-4-5")

    out = await provider.complete(
        spec, Request(messages=[Message(role="user", content="2+3?")], tools=[_TOOL])
    )
    assert [(c.id, c.name, c.input) for c in out.tool_calls] == [("c1", "add", {"a": 2, "b": 3})]
    assert fake.calls[0]["tools"][0]["name"] == "add"

    await provider.complete(spec, Request(messages=_HISTORY))
    sent = fake.calls[1]["messages"]
    assert sent[1]["role"] == "assistant" and sent[1]["content"][0]["type"] == "tool_use"
    assert sent[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "5"}],
    }


async def test_gemini_tool_use_translation() -> None:
    resp = SimpleNamespace(
        text=None,
        function_calls=[SimpleNamespace(name="add", args={"a": 2, "b": 3})],
        usage_metadata=SimpleNamespace(
            prompt_token_count=1, candidates_token_count=1, cached_content_token_count=0
        ),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
    )
    fake = _FakeGemini(resp)
    provider = GeminiProvider(client=fake)
    spec = ModelSpec(provider="gemini", model="gemini-2.5-flash")

    out = await provider.complete(
        spec, Request(messages=[Message(role="user", content="2+3?")], tools=[_TOOL])
    )
    assert [(c.name, c.input) for c in out.tool_calls] == [("add", {"a": 2, "b": 3})]
    assert fake.calls[0]["config"].tools[0].function_declarations[0].name == "add"

    await provider.complete(spec, Request(messages=_HISTORY))
    contents = fake.calls[1]["contents"]
    assert contents[1].role == "model" and contents[1].parts[0].function_call.name == "add"
    assert contents[2].role == "tool" and contents[2].parts[0].function_response.name == "add"


async def test_grok_tool_use_translation() -> None:
    resp = SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(
                id="c1", function=SimpleNamespace(name="add", arguments='{"a": 2, "b": 3}')
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, cached_prompt_text_tokens=0),
        finish_reason="tool_calls",
    )
    fake = _FakeGrok(resp)
    provider = GrokProvider(client=fake)
    spec = ModelSpec(provider="grok", model="grok-4.3")

    out = await provider.complete(
        spec, Request(messages=[Message(role="user", content="2+3?")], tools=[_TOOL])
    )
    assert [(c.id, c.name, c.input) for c in out.tool_calls] == [("c1", "add", {"a": 2, "b": 3})]
    assert fake.calls[0]["tools"][0].function.name == "add"

    await provider.complete(spec, Request(messages=_HISTORY))
    sent = fake.calls[1]["messages"]
    assistant_msgs = [m for m in sent if list(getattr(m, "tool_calls", []))]
    assert assistant_msgs and assistant_msgs[0].tool_calls[0].function.name == "add"
    tool_msgs = [m for m in sent if getattr(m, "tool_call_id", "")]
    assert tool_msgs and tool_msgs[0].tool_call_id == "c1"


class _ScriptedAnthropic:
    """Returns a scripted sequence of responses (for the pause_turn server-tool loop)."""

    def __init__(self, responses: list[Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._responses.pop(0)


async def test_anthropic_resumes_server_tool_pause_and_collects_sources() -> None:
    paused = SimpleNamespace(
        content=[SimpleNamespace(type="server_tool_use", id="s1", name="web_search", input={})],
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=2,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        stop_reason="pause_turn",
    )
    final = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text="~14M.",
                citations=[SimpleNamespace(url="https://example.com/tokyo")],
            )
        ],
        usage=SimpleNamespace(
            input_tokens=6,
            output_tokens=4,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        stop_reason="end_turn",
    )
    fake = _ScriptedAnthropic([paused, final])
    out = await AnthropicProvider(client=fake).complete(
        ModelSpec(provider="anthropic", model="claude-sonnet-4-6"),
        Request(
            messages=[Message(role="user", content="population of Tokyo?")],
            server_tools=[{"type": "web_search_20250305", "name": "web_search"}],
        ),
    )
    assert out.text == "~14M."
    assert out.sources == ["https://example.com/tokyo"]
    assert out.finish_reason == "end_turn"
    assert out.usage.output_tokens == 6  # 2 (paused) + 4 (final), accumulated across resume
    assert len(fake.calls) == 2  # the paused turn was resumed
    assert fake.calls[0]["tools"] == [{"type": "web_search_20250305", "name": "web_search"}]
    assert fake.calls[1]["messages"][-1]["role"] == "assistant"  # raw content echoed back


# --- structured output + vision translation (M3) -----------------------------------------
# Assert each provider forwards response_format (json schema) and attaches media to the SDK
# request. (The provider returns JSON text; Router decodes it into Response.structured.)

_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def _vision_req() -> Request:
    return Request(
        messages=[Message(role="user", content="is it ok?")],
        response_format=_SCHEMA,
        media=[MediaPart(mime_type="image/png", data=b"\x89PNGfake")],
    )


async def test_anthropic_structured_and_media_translation() -> None:
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"ok": true}')],
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=1,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        stop_reason="end_turn",
    )
    fake = _FakeAnthropic(resp)
    await AnthropicProvider(client=fake).complete(
        ModelSpec(provider="anthropic", model="claude-haiku-4-5"), _vision_req()
    )
    sent = fake.calls[0]
    assert sent["output_config"]["format"] == {"type": "json_schema", "schema": _SCHEMA}
    user_content = sent["messages"][0]["content"]
    assert any(b.get("type") == "image" for b in user_content)


async def test_gemini_structured_and_media_translation() -> None:
    resp = SimpleNamespace(
        text='{"ok": true}',
        usage_metadata=SimpleNamespace(
            prompt_token_count=1, candidates_token_count=1, cached_content_token_count=0
        ),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
    )
    fake = _FakeGemini(resp)
    await GeminiProvider(client=fake).complete(
        ModelSpec(provider="gemini", model="gemini-2.5-flash"), _vision_req()
    )
    cfg = fake.calls[0]["config"]
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == _SCHEMA
    parts = fake.calls[0]["contents"][0].parts
    assert any(getattr(p, "inline_data", None) is not None for p in parts)


async def test_grok_structured_and_media_translation() -> None:
    resp = SimpleNamespace(
        content='{"ok": true}',
        tool_calls=[],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, cached_prompt_text_tokens=0),
        finish_reason="stop",
    )
    fake = _FakeGrok(resp)
    await GrokProvider(client=fake).complete(
        ModelSpec(provider="grok", model="grok-4.3"), _vision_req()
    )
    kwargs = fake.calls[0]
    rf = kwargs["response_format"]
    assert rf.format_type == chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA
    assert json.loads(rf.schema) == _SCHEMA
    user_msg = kwargs["messages"][-1]
    assert any(c.HasField("image_url") for c in user_msg.content)
