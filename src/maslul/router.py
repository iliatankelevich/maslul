"""The router — picks a model for each request and returns a normalized :class:`Response`.

Routing decision order (plan §1.3), highest precedence first:

    0. ``model=`` pinned        → that exact provider:model, no routing
    1. ``level=`` pinned        → that tier
    2. deterministic bypass     → injectable predicate picks a tier (e.g. greetings → SIMPLE)
    3. hard-signal detector     → HARD (UP-only; never routes down)
    4. ambiguous middle         → a caller-supplied classifier, else the configured Strategy

M4 part 1 implements 0–3, ``ROUTE_DEFAULT``, and the custom-classifier hook; the LLM
``CLASSIFY`` / ``CLASSIFY_AND_ANSWER`` strategies are part 2. The tool-use loop (M2) and
structured-output decode (M3) run underneath whichever model routing selects.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from maslul.config import RouterConfig
from maslul.errors import ConfigError, ProviderError
from maslul.providers import Provider, build_provider
from maslul.types import (
    BypassPredicate,
    Classifier,
    CompleteHook,
    HardSignal,
    Level,
    Message,
    ModelSpec,
    ModelUsage,
    Request,
    Response,
    RouteHook,
    RoutingDecision,
    Strategy,
    ToolCall,
    Usage,
)

# Guard against a runaway tool loop (mirrors Kippy's llm.py).
_MAX_TOOL_ITERATIONS = 8


class Router:
    """Routes a :class:`Request` to a provider/model and returns a normalized :class:`Response`.

    ``providers`` may be injected (the ``FakeProvider`` in tests); when omitted, the providers
    named by the configured tiers and classifier are auto-built from ``[maslul.providers.*]``.

    Injectable routing hooks: ``bypass_predicate`` (a deterministic tier fast-path),
    ``hard_signal`` (UP-only escalation; defaults to :func:`default_hard_signal`), and
    ``classifier`` (the caller's own classification method for the ambiguous middle).
    Observability hooks ``on_route`` / ``on_complete`` may be passed here or registered later;
    ``on_complete`` receives each :class:`Response` with its per-model ``usage_records``.
    """

    def __init__(
        self,
        config: RouterConfig | Mapping[str, Any],
        providers: Mapping[str, Provider] | None = None,
        *,
        bypass_predicate: BypassPredicate | None = None,
        hard_signal: HardSignal | None = None,
        classifier: Classifier | None = None,
        on_route: RouteHook | None = None,
        on_complete: CompleteHook | None = None,
    ) -> None:
        self._config = (
            config if isinstance(config, RouterConfig) else RouterConfig.from_dict(config)
        )
        if providers is None:
            providers = {
                name: build_provider(name, self._config.providers.get(name, {}))
                for name in self._provider_names()
            }
        self._providers: dict[str, Provider] = dict(providers)
        self._bypass = bypass_predicate
        self._hard_signal = hard_signal or default_hard_signal
        self._classifier = classifier
        self._on_route: list[RouteHook] = [on_route] if on_route else []
        self._on_complete: list[CompleteHook] = [on_complete] if on_complete else []

    @classmethod
    def from_toml(
        cls,
        path: str | os.PathLike[str],
        providers: Mapping[str, Provider] | None = None,
        **hooks: Any,
    ) -> Router:
        """Build a router from a TOML config file. Omit ``providers`` to auto-build them from
        the ``[maslul.providers.*]`` config; pass routing/observability hooks as keywords."""
        return cls(RouterConfig.from_toml(path), providers, **hooks)

    def on_route(self, callback: RouteHook) -> None:
        """Register a callback fired with ``(request, RoutingDecision)`` before each model call."""
        self._on_route.append(callback)

    def on_complete(self, callback: CompleteHook) -> None:
        """Register a callback fired with the final :class:`Response` (incl. usage_records)."""
        self._on_complete.append(callback)

    @property
    def config(self) -> RouterConfig:
        return self._config

    async def complete(
        self,
        req: Request,
        *,
        level: Level | None = None,
        model: str | ModelSpec | None = None,
        strategy: Strategy | None = None,
    ) -> Response:
        """Route ``req`` to a model and run it.

        ``model=`` pins an exact ``provider:model`` (skips routing); ``level=`` pins a tier;
        otherwise the routing brain decides (bypass → hard-signal → strategy/classifier). With
        ``req.tools`` + ``req.tool_executor`` the provider-agnostic tool loop runs underneath.
        """
        decision = await self._route(req, level=level, model=model, strategy=strategy)
        for cb in self._on_route:
            cb(req, decision)
        resp = await self._execute(decision.spec, req)
        resp.level_used = decision.level
        for cb in self._on_complete:
            cb(resp)
        return resp

    async def _route(
        self,
        req: Request,
        *,
        level: Level | None,
        model: str | ModelSpec | None,
        strategy: Strategy | None,
    ) -> RoutingDecision:
        if model is not None:  # 0. explicit model pin
            spec = model if isinstance(model, ModelSpec) else ModelSpec.parse(model)
            return RoutingDecision(spec=spec, level=None, reason="model_pinned")
        if level is not None:  # 1. explicit level pin
            return RoutingDecision(self._spec_for_level(level), level, "level_pinned")
        if self._bypass is not None:  # 2. deterministic bypass (any tier)
            bypass_level = self._bypass(req)
            if bypass_level is not None:
                return RoutingDecision(self._spec_for_level(bypass_level), bypass_level, "bypass")
        if self._hard_signal(req):  # 3. hard-signal (UP-only)
            return RoutingDecision(self._spec_for_level(Level.HARD), Level.HARD, "hard_signal")
        strat = strategy or self._config.strategy  # 4. ambiguous middle
        chosen = await self._resolve_middle(req, strat)
        return RoutingDecision(self._spec_for_level(chosen), chosen, f"strategy:{strat.value}")

    async def _resolve_middle(self, req: Request, strategy: Strategy) -> Level:
        """Resolve the ambiguous middle: a caller-supplied classifier wins; else the strategy."""
        if self._classifier is not None:
            result = self._classifier(req)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        if strategy is Strategy.ROUTE_DEFAULT:
            return self._config.default_level
        raise NotImplementedError(
            f"strategy {strategy.value!r} lands in M4 part 2 — "
            "use route_default or inject a classifier"
        )

    async def _execute(self, spec: ModelSpec, req: Request) -> Response:
        """Run the model (with the tool loop when tools are present), decode structured output,
        and attach the per-model usage breakdown."""
        provider = self._provider_for(spec)
        ledger: dict[tuple[str, str], Usage] = {}
        if not req.tools or req.tool_executor is None:
            resp = await provider.complete(spec, req)
            _record(ledger, resp)
        else:
            resp = await self._run_tool_loop(spec, provider, req, ledger)
        _apply_structured(req, resp)
        resp.usage = _total(ledger)
        resp.usage_records = [ModelUsage(p, m, u) for (p, m), u in ledger.items()]
        return resp

    async def _run_tool_loop(
        self,
        spec: ModelSpec,
        provider: Provider,
        req: Request,
        ledger: dict[tuple[str, str], Usage],
    ) -> Response:
        assert req.tool_executor is not None
        messages = list(req.messages)
        executed: list[ToolCall] = []
        for _ in range(_MAX_TOOL_ITERATIONS):
            resp = await provider.complete(spec, replace(req, messages=messages))
            _record(ledger, resp)
            if not resp.tool_calls:
                resp.tool_calls = executed  # surface the full tool I/O of the turn
                return resp
            messages.append(
                Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls)
            )
            for call in resp.tool_calls:
                executed.append(call)
                result = await req.tool_executor(call)
                messages.append(
                    Message(role="tool", content=result, tool_call_id=call.id, name=call.name)
                )
        raise ProviderError(f"tool loop exceeded {_MAX_TOOL_ITERATIONS} iterations")

    def _provider_names(self) -> set[str]:
        """Provider names referenced by the configured tiers and classifier."""
        names = {spec.provider for spec in self._config.tiers.values()}
        if self._config.classifier is not None:
            names.add(self._config.classifier.provider)
        return names

    def _spec_for_level(self, level: Level) -> ModelSpec:
        try:
            return self._config.tiers[level]
        except KeyError as e:
            raise ConfigError(f"no tier configured for level {level.name}") from e

    def _provider_for(self, spec: ModelSpec) -> Provider:
        try:
            return self._providers[spec.provider]
        except KeyError as e:
            raise ConfigError(
                f"no provider registered for {spec.provider!r} "
                f"(registered: {sorted(self._providers)})"
            ) from e


# Intent verbs that mark genuinely hard work — Hebrew + English (plan §1.2).
_HARD_SIGNAL_KEYWORDS = re.compile(
    r"תחקור|תנתח|נתח|השווה|הוכח|חקור|"
    r"\banaly[sz]e\b|\bresearch\b|\bprove\b|\bderive\b|\bcompare\b|\bevaluate\b",
    re.IGNORECASE,
)
_LONG_CONTEXT_CHARS = 8000


def default_hard_signal(req: Request) -> bool:
    """UP-only heuristic: route to HARD when there are explicit complexity markers — attached
    media, a very long context, fenced code, or an intent verb (Hebrew or English). Never a
    ``short ⇒ simple`` rule (plan §1.1): absence of a signal means *undecided*, not *easy*."""
    if req.media:
        return True
    text = "\n".join(m.content for m in req.messages if m.content)
    if len(text) > _LONG_CONTEXT_CHARS or "```" in text:
        return True
    return bool(_HARD_SIGNAL_KEYWORDS.search(text))


def _record(ledger: dict[tuple[str, str], Usage], resp: Response) -> None:
    _accumulate(ledger.setdefault((resp.provider, resp.model), Usage()), resp.usage)


def _total(ledger: dict[tuple[str, str], Usage]) -> Usage:
    total = Usage()
    for u in ledger.values():
        _accumulate(total, u)
    return total


def _accumulate(total: Usage, u: Usage) -> None:
    total.input_tokens += u.input_tokens
    total.output_tokens += u.output_tokens
    total.cache_read_input_tokens += u.cache_read_input_tokens
    total.cache_creation_input_tokens += u.cache_creation_input_tokens


def _apply_structured(req: Request, resp: Response) -> None:
    """When ``response_format`` was requested, parse the JSON answer into ``Response.structured``.
    The provider already constrained the model to emit JSON; here we just decode the text. A
    parse failure (e.g. a refusal) leaves ``structured`` as ``None`` for the caller to handle."""
    if req.response_format is None or resp.structured is not None or not resp.text:
        return
    with contextlib.suppress(ValueError, TypeError):
        resp.structured = json.loads(resp.text)
