from __future__ import annotations

from typing import Any

import pytest

from maslul import (
    AuthError,
    ConfigError,
    Level,
    Message,
    ModelSpec,
    ModelUsage,
    RateLimited,
    Request,
    Response,
    Router,
    RoutingDecision,
    Strategy,
    ToolCall,
    ToolDef,
    Usage,
)
from tests.fakes import FakeProvider, FlakyProvider, ScriptedProvider


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


async def test_route_default_uses_default_level_when_no_level_pinned() -> None:
    router = Router(_config(), providers={"fake": FakeProvider("fake")})
    resp = await router.complete(_req())  # no level → ROUTE_DEFAULT → default_level (hard)
    assert resp.level_used is Level.HARD
    assert resp.model == "big"


async def test_model_pin_skips_routing() -> None:
    fake = FakeProvider("fake")
    router = Router(_config(), providers={"fake": fake})
    resp = await router.complete(_req(), model=ModelSpec(provider="fake", model="exact-model"))
    assert resp.model == "exact-model"
    assert resp.level_used is None  # a model pin is not a tier decision
    assert fake.calls[0][0].model == "exact-model"


async def test_hard_signal_escalates_to_hard() -> None:
    # default_level is SIMPLE here, so routing to HARD can only come from the hard-signal detector
    config = _config()
    config["maslul"]["default_level"] = "simple"
    router = Router(config, providers={"fake": FakeProvider("fake")})

    plain = Request(messages=[Message(role="user", content="hi there")])
    assert (await router.complete(plain)).level_used is Level.SIMPLE  # no signal → default

    hard = Request(messages=[Message(role="user", content="תחקור את הנושא הזה לעומק")])
    assert (await router.complete(hard)).level_used is Level.HARD  # intent verb → HARD


async def test_bypass_predicate_picks_tier() -> None:
    def bypass(req: Request) -> Level | None:
        return Level.SIMPLE if "hi" in req.messages[-1].content else None

    router = Router(_config(), providers={"fake": FakeProvider("fake")}, bypass_predicate=bypass)
    resp = await router.complete(Request(messages=[Message(role="user", content="hi")]))
    assert resp.level_used is Level.SIMPLE  # bypass fired (beats ROUTE_DEFAULT's hard)


async def test_custom_classifier_resolves_middle() -> None:
    async def classify(req: Request) -> Level:  # async classifier supported
        return Level.MEDIUM

    router = Router(_config(), providers={"fake": FakeProvider("fake")}, classifier=classify)
    resp = await router.complete(_req())
    assert resp.level_used is Level.MEDIUM
    assert resp.model == "mid"


async def test_route_and_complete_hooks_fire() -> None:
    decisions: list[RoutingDecision] = []
    completions: list[Response] = []
    router = Router(
        _config(),
        providers={"fake": FakeProvider("fake")},
        on_route=lambda _req, decision: decisions.append(decision),
        on_complete=completions.append,
    )
    await router.complete(_req())
    assert len(decisions) == 1 and decisions[0].reason == "strategy:route_default"
    assert decisions[0].level is Level.HARD
    assert len(completions) == 1 and completions[0].model == "big"


async def test_usage_records_expose_per_model_breakdown() -> None:
    router = Router(_config(), providers={"fake": FakeProvider("fake")})
    resp = await router.complete(_req(), level=Level.SIMPLE)
    assert resp.usage_records == [ModelUsage(provider="fake", model="small", usage=resp.usage)]


def _classify_config(
    strategy: str, *, classifier_model: str = "judge", min_tokens: int = 1
) -> dict[str, Any]:
    config = _config()
    config["maslul"]["strategy"] = strategy
    config["maslul"]["min_tokens_to_classify"] = min_tokens
    config["maslul"]["classifier"] = {"provider": "fake", "model": classifier_model}
    return config


async def test_classify_strategy_routes_via_classifier_model() -> None:
    fake = FakeProvider("fake", text='{"level": "medium"}')
    router = Router(_classify_config("classify"), providers={"fake": fake})
    resp = await router.complete(
        Request(messages=[Message(role="user", content="a real question")])
    )
    assert resp.level_used is Level.MEDIUM
    assert resp.model == "mid"  # the MEDIUM tier
    assert resp.classification_usage is not None
    assert {r.model for r in resp.usage_records} == {"judge", "mid"}  # classifier + answer


async def test_classify_budget_guard_skips_to_default() -> None:
    fake = FakeProvider("fake", text='{"level": "simple"}')
    router = Router(_classify_config("classify", min_tokens=1000), providers={"fake": fake})
    resp = await router.complete(Request(messages=[Message(role="user", content="hi")]))
    assert resp.level_used is Level.HARD  # below budget → default_level, classifier skipped
    assert all(r.model != "judge" for r in resp.usage_records)


async def test_classify_caches_by_prompt() -> None:
    fake = FakeProvider("fake", text='{"level": "simple"}')
    router = Router(_classify_config("classify"), providers={"fake": fake})
    req = Request(messages=[Message(role="user", content="identical question")])
    await router.complete(req)
    after_first = len(fake.calls)  # classify + answer
    await router.complete(req)
    assert len(fake.calls) - after_first == 1  # classify cached → only the answer call


async def test_classify_and_answer_answers_inline() -> None:
    fake = FakeProvider("fake", text="The answer is 42.")
    router = Router(
        _classify_config("classify_and_answer", classifier_model="floor"), providers={"fake": fake}
    )
    resp = await router.complete(_req())
    assert resp.text == "The answer is 42."
    assert resp.level_used is None  # the classifier answered; not a tier decision
    assert len(fake.calls) == 1  # a single combined call
    assert resp.usage_records[0].model == "floor"


async def test_classify_and_answer_escalates_on_sentinel() -> None:
    from maslul import ESCALATE_SENTINEL

    fake = FakeProvider("fake", text=ESCALATE_SENTINEL)
    router = Router(
        _classify_config("classify_and_answer", classifier_model="floor"), providers={"fake": fake}
    )
    resp = await router.complete(_req())
    assert resp.level_used is Level.HARD  # sentinel → re-dispatch to HARD tier
    assert resp.model == "big"
    assert resp.classification_usage is not None  # the declined classifier call still counts
    assert len(fake.calls) == 2  # classifier (sentinel) + escalated answer
    assert {r.model for r in resp.usage_records} == {"floor", "big"}


async def test_classify_and_answer_runs_tool_loop() -> None:
    # The inline answer is a FULL turn: the cheap model asks for a tool, the router executes it
    # and feeds the result back, and the model answers — tools are not dropped.
    provider = ScriptedProvider(
        "fake",
        [
            Response(
                text="",
                level_used=None,
                provider="fake",
                model="floor",
                usage=Usage(input_tokens=3, output_tokens=2),
                tool_calls=[ToolCall(id="c1", name="add", input={"a": 2, "b": 3})],
            ),
            Response(
                text="It's 5.",
                level_used=None,
                provider="fake",
                model="floor",
                usage=Usage(input_tokens=4, output_tokens=1),
            ),
        ],
    )
    router = Router(
        _classify_config("classify_and_answer", classifier_model="floor"),
        providers={"fake": provider},
    )
    executed: list[ToolCall] = []

    async def executor(call: ToolCall) -> str:
        executed.append(call)
        return "5"

    req = Request(
        messages=[Message(role="user", content="2+3?")],
        tools=[ToolDef(name="add", description="add", input_schema={"type": "object"})],
        tool_executor=executor,
    )
    resp = await router.complete(req)
    assert resp.text == "It's 5."  # answered via the tool loop, not a single bare call
    assert [c.name for c in executed] == ["add"]  # the tool actually ran
    assert [c.name for c in resp.tool_calls] == ["add"]
    assert resp.usage.output_tokens == 3  # 2 + 1 accumulated across the seeded loop
    assert resp.level_used is None  # answered inline


async def test_classify_without_classifier_config_raises() -> None:
    config = _config()
    config["maslul"]["strategy"] = "classify"
    router = Router(config, providers={"fake": FakeProvider("fake")})
    with pytest.raises(ConfigError):
        await router.complete(_req())


async def test_verify_cascade_requires_a_verifier() -> None:
    router = Router(_config(), providers={"fake": FakeProvider("fake")})
    with pytest.raises(ConfigError):
        await router.complete(_req(), strategy=Strategy.VERIFY_CASCADE)


async def test_verify_cascade_accepts_cheap_answer() -> None:
    fake = FakeProvider("fake", text="cheap answer")

    async def accept(_req: Request, _resp: Response) -> bool:  # async verifier supported
        return True

    router = Router(_config(), providers={"fake": fake}, verifier=accept)
    resp = await router.complete(_req(), strategy=Strategy.VERIFY_CASCADE)
    assert resp.text == "cheap answer"
    assert resp.level_used is Level.SIMPLE  # cheapest tier, accepted
    assert resp.model == "small"


async def test_verify_cascade_escalates_when_rejected() -> None:
    fake = FakeProvider("fake")
    verified: list[str] = []

    def reject(_req: Request, resp: Response) -> bool:
        verified.append(resp.model)
        return False  # the cheap answer isn't good enough → escalate

    router = Router(_config(), providers={"fake": fake}, verifier=reject)
    resp = await router.complete(_req(), strategy=Strategy.VERIFY_CASCADE)
    assert resp.level_used is Level.HARD  # escalated to the most capable tier
    assert resp.model == "big"
    assert resp.classification_usage is not None  # the cheap attempt still counts
    assert {r.model for r in resp.usage_records} == {"small", "big"}  # floor + escalated
    assert verified == ["small"]  # the verifier saw the cheap (SIMPLE) answer


async def test_missing_provider_raises_config_error() -> None:
    router = Router(_config(), providers={})
    with pytest.raises(ConfigError):
        await router.complete(_req(), level=Level.HARD)


def _grok_heavy_config() -> dict[str, Any]:
    return {
        "maslul": {
            "strategy": "classify_and_answer",
            "default_level": "medium",
            "classifier": {"provider": "grok", "model": "g-classify"},
            "tiers": {
                "simple": {"provider": "grok", "model": "g-small"},
                "medium": {"provider": "grok", "model": "g-mid"},
                "hard": {"provider": "fake", "model": "big"},
            },
        }
    }


async def test_degrade_remaps_unavailable_tiers_and_drops_classifier() -> None:
    # Only "fake" is available; the grok tiers + grok classifier must degrade gracefully.
    router = Router(
        _grok_heavy_config(),
        providers={"fake": FakeProvider("fake")},
        missing_provider="degrade",
    )
    cfg = router.config
    # grok tiers remap to the nearest available higher tier (HARD = fake:big)
    assert (cfg.tiers[Level.SIMPLE].provider, cfg.tiers[Level.SIMPLE].model) == ("fake", "big")
    assert cfg.tiers[Level.MEDIUM].model == "big"
    assert cfg.tiers[Level.HARD].model == "big"
    assert cfg.classifier is None  # the grok classifier is dropped …
    assert cfg.strategy is Strategy.ROUTE_DEFAULT  # … and the strategy downgrades
    resp = await router.complete(_req())  # routes without crashing
    assert resp.provider == "fake"


async def test_degrade_raises_when_no_tier_is_available() -> None:
    with pytest.raises(ConfigError, match="no configured tier"):
        Router(
            _grok_heavy_config(),
            providers={"other": FakeProvider("other")},
            missing_provider="degrade",
        )


async def test_error_mode_is_the_default_and_keeps_config_intact() -> None:
    router = Router(_grok_heavy_config(), providers={"fake": FakeProvider("fake")})
    assert router.config.tiers[Level.SIMPLE].provider == "grok"  # untouched
    with pytest.raises(ConfigError):  # missing grok surfaces at call time
        await router.complete(_req(), level=Level.SIMPLE)


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


# --- resilience (M5) ---------------------------------------------------------------------


async def test_transient_error_is_retried_then_succeeds() -> None:
    flaky = FlakyProvider("fake", fails=2)  # two RateLimited, then a real reply
    config = _config()
    config["maslul"]["max_retries"] = 2
    config["maslul"]["retry_base_delay"] = 0  # no real sleeping in tests
    router = Router(config, providers={"fake": flaky})
    resp = await router.complete(_req(), level=Level.SIMPLE)
    assert resp.text == "recovered"
    assert flaky.attempts == 3


async def test_falls_back_to_higher_tier_on_persistent_failure() -> None:
    flaky = FlakyProvider("flaky", fails=99)  # never recovers
    stable = FakeProvider("stable", text="from the hard tier")
    config = {
        "maslul": {
            "strategy": "route_default",
            "default_level": "simple",
            "max_retries": 0,
            "retry_base_delay": 0,
            "tiers": {
                "simple": {"provider": "flaky", "model": "small"},
                "hard": {"provider": "stable", "model": "big"},
            },
        }
    }
    router = Router(config, providers={"flaky": flaky, "stable": stable})
    resp = await router.complete(_req())  # SIMPLE fails → fall back to the HARD tier
    assert resp.text == "from the hard tier"
    assert resp.provider == "stable"


async def test_auth_error_is_not_retried_or_fallen_back() -> None:
    flaky = FlakyProvider("fake", fails=99, error=AuthError("bad key"))
    config = _config()
    config["maslul"]["max_retries"] = 3
    router = Router(config, providers={"fake": flaky})
    with pytest.raises(AuthError):
        await router.complete(_req(), level=Level.SIMPLE)
    assert flaky.attempts == 1  # config problem → fail fast, no retry/fallback


async def test_on_error_fires_per_failed_attempt() -> None:
    flaky = FlakyProvider("fake", fails=2)
    errors: list[Exception] = []
    config = _config()
    config["maslul"]["max_retries"] = 2
    config["maslul"]["retry_base_delay"] = 0
    router = Router(
        config,
        providers={"fake": flaky},
        on_error=lambda _req, _spec, exc: errors.append(exc),
    )
    await router.complete(_req(), level=Level.SIMPLE)
    assert len(errors) == 2  # two transient failures before the success
    assert all(isinstance(e, RateLimited) for e in errors)
