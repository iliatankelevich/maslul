from __future__ import annotations

from typing import Any

import pytest

from maslul import (
    ConfigError,
    Level,
    Message,
    Request,
    Response,
    Router,
    Strategy,
    ToolCall,
    ToolDef,
    Usage,
)
from tests.fakes import FakeProvider, ScriptedProvider


def _config() -> dict[str, Any]:
    return {
        "maslul": {
            "strategy": "route_default",
            "default_level": "hard",
            "tiers": {
                "simple": {"provider": "fake", "model": "small"},
                "medium": {"provider": "fake", "model": "mid"},
                "hard": {"provider": "fake", "model": "big"},
            },
        }
    }


def _req() -> Request:
    return Request(messages=[Message(role="user", content="hello")])


async def test_level_dispatch_returns_normalized_response() -> None:
    fake = FakeProvider("fake", text="hi")
    router = Router(_config(), providers={"fake": fake})

    resp = await router.complete(_req(), level=Level.SIMPLE)

    assert resp.text == "hi"
    assert resp.level_used is Level.SIMPLE
    assert resp.provider == "fake"
    assert resp.model == "small"  # the SIMPLE tier
    assert len(fake.calls) == 1
    assert fake.calls[0][0].model == "small"


async def test_complete_without_level_is_not_implemented_yet() -> None:
    router = Router(_config(), providers={"fake": FakeProvider("fake")})
    with pytest.raises(NotImplementedError):
        await router.complete(_req())


async def test_missing_provider_raises_config_error() -> None:
    router = Router(_config(), providers={})
    with pytest.raises(ConfigError):
        await router.complete(_req(), level=Level.HARD)


def test_from_dict_parses_tiers_and_routing_knobs() -> None:
    cfg = Router(_config(), providers={"fake": FakeProvider("fake")}).config
    assert cfg.strategy is Strategy.ROUTE_DEFAULT
    assert cfg.default_level is Level.HARD
    assert cfg.tiers[Level.SIMPLE].model == "small"
    assert cfg.tiers[Level.HARD].model == "big"


async def test_router_auto_builds_providers_when_none_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []

    def fake_build(name: str, config: dict[str, Any]) -> FakeProvider:
        built.append(name)
        return FakeProvider(name)

    monkeypatch.setattr("maslul.router.build_provider", fake_build)

    router = Router(_config())  # no providers → auto-build the names the tiers reference
    resp = await router.complete(_req(), level=Level.SIMPLE)

    assert built == ["fake"]  # built once, only the referenced provider
    assert resp.provider == "fake"


async def test_tool_loop_runs_tool_feeds_result_back_and_terminates() -> None:
    provider = ScriptedProvider(
        "fake",
        [
            # turn 1: the model asks for the tool
            Response(
                text="",
                level_used=None,
                provider="fake",
                model="small",
                usage=Usage(input_tokens=3, output_tokens=2),
                tool_calls=[ToolCall(id="c1", name="add", input={"a": 2, "b": 3})],
            ),
            # turn 2: with the result in hand, it answers
            Response(
                text="It's 5.",
                level_used=None,
                provider="fake",
                model="small",
                usage=Usage(input_tokens=4, output_tokens=1),
            ),
        ],
    )
    router = Router(_config(), providers={"fake": provider})

    executed: list[ToolCall] = []

    async def executor(call: ToolCall) -> str:
        executed.append(call)
        return str(call.input["a"] + call.input["b"])

    req = Request(
        messages=[Message(role="user", content="2+3?")],
        tools=[ToolDef(name="add", description="add two numbers", input_schema={"type": "object"})],
        tool_executor=executor,
    )
    resp = await router.complete(req, level=Level.SIMPLE)

    assert resp.text == "It's 5."
    assert [c.name for c in executed] == ["add"]
    assert [c.name for c in resp.tool_calls] == ["add"]  # tool I/O surfaced on the final response
    assert resp.usage.output_tokens == 3  # 2 + 1 accumulated across turns
    # the second model call saw the assistant tool-call turn + the tool result
    second = provider.requests[1].messages
    assert any(m.role == "assistant" and m.tool_calls for m in second)
    assert any(m.role == "tool" and m.content == "5" and m.tool_call_id == "c1" for m in second)


async def test_response_format_parses_text_into_structured() -> None:
    provider = ScriptedProvider(
        "fake",
        [
            Response(
                text='{"city": "Paris", "country": "France"}',
                level_used=None,
                provider="fake",
                model="small",
                usage=Usage(),
            )
        ],
    )
    router = Router(_config(), providers={"fake": provider})
    req = Request(
        messages=[Message(role="user", content="extract")],
        response_format={"type": "object"},
    )
    resp = await router.complete(req, level=Level.SIMPLE)
    assert resp.structured == {"city": "Paris", "country": "France"}


async def test_tool_loop_raises_when_iterations_exhausted() -> None:
    from maslul import ProviderError

    def always_calls_tool() -> Response:
        return Response(
            text="",
            level_used=None,
            provider="fake",
            model="small",
            usage=Usage(),
            tool_calls=[ToolCall(id="c", name="add", input={})],
        )

    # a provider that never stops asking for the tool → the iteration guard must trip
    provider = ScriptedProvider("fake", [always_calls_tool() for _ in range(50)])
    router = Router(_config(), providers={"fake": provider})

    async def executor(call: ToolCall) -> str:
        return "again"

    req = Request(
        messages=[Message(role="user", content="loop")],
        tools=[ToolDef(name="add", description="add", input_schema={"type": "object"})],
        tool_executor=executor,
    )
    with pytest.raises(ProviderError):
        await router.complete(req, level=Level.SIMPLE)
