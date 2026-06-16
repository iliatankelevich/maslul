"""The router — picks a model for each request and returns a normalized :class:`Response`.

Routing decision order (plan §1.3), highest precedence first:

    0. ``model=`` pinned        → that exact provider:model, no routing
    1. ``level=`` pinned        → that tier
    2. deterministic bypass     → injectable predicate picks a tier (e.g. greetings → SIMPLE)
    3. hard-signal detector     → HARD (UP-only; never routes down)
    4. ambiguous middle         → a caller-supplied classifier, else the configured Strategy

Strategies for the middle: ``ROUTE_DEFAULT`` (default-to-capable), ``CLASSIFY`` (a cheap
dedicated classifier model labels the level, cached + budget-guarded, then dispatch to that
tier), and ``CLASSIFY_AND_ANSWER`` (the classifier model answers directly, or emits the
escalation sentinel to bump to a stronger tier). ``VERIFY_CASCADE`` is M5.

The tool-use loop (M2) and structured-output decode (M3) run underneath whichever model
routing selects. Every model call (classify + answer + tool iterations) is recorded into the
response's per-model usage breakdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from maslul.config import RouterConfig
from maslul.errors import AuthError, ConfigError, MaslulError, ProviderError, RateLimited, Timeout
from maslul.providers import Provider, build_provider
from maslul.types import (
    BypassPredicate,
    Classifier,
    CompleteHook,
    ErrorHook,
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
# Headroom for a classify call — small, but enough for a thinking classifier to reason then label.
_CLASSIFY_MAX_TOKENS = 512

#: Emitted verbatim by a CLASSIFY_AND_ANSWER model that declines to answer and wants escalation.
ESCALATE_SENTINEL = "⟦MASLUL::ESCALATE::hard⟧"
_ESCALATE_RE = re.compile(r"^\s*⟦MASLUL::ESCALATE::(\w+)⟧")

_LEVEL_BY_NAME = {"simple": Level.SIMPLE, "medium": Level.MEDIUM, "hard": Level.HARD}
_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {"level": {"type": "string", "enum": ["simple", "medium", "hard"]}},
    "required": ["level"],
    "additionalProperties": False,
}
_CLASSIFY_INSTRUCTIONS = (
    "You are a routing classifier. Decide how much *reasoning capability* the assistant needs "
    "to answer the user's request CORRECTLY — judge intrinsic difficulty, not how long the "
    "answer should be (a short prompt can be very hard; a long paste can be trivial). "
    "simple = trivial lookups, greetings, reformatting; medium = moderate or short multi-step "
    "reasoning; hard = deep reasoning, research, proofs, analysis, ambiguous or high-stakes. "
    "When unsure, pick the harder level. Respond with the level only."
)
_CLASSIFY_AND_ANSWER_GUIDANCE = (
    "Answer the user's request directly if you can do so correctly. If answering it correctly "
    "needs a more capable model, do NOT attempt it — reply with EXACTLY this and nothing "
    f"else: {ESCALATE_SENTINEL}"
)


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
        on_error: ErrorHook | None = None,
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
        self._on_error: list[ErrorHook] = [on_error] if on_error else []
        self._classify_cache: dict[str, Level] = {}

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

    def on_error(self, callback: ErrorHook) -> None:
        """Register a callback fired on each failed model attempt (retry or fallback)."""
        self._on_error.append(callback)

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
        strat = strategy or self._config.strategy
        decision, prepared = await self._route(req, level=level, model=model, strategy=strat)
        for cb in self._on_route:
            cb(req, decision)
        if prepared is not None:  # CLASSIFY_AND_ANSWER answered inline
            resp = self._finalize(req, prepared, _ledger_of(prepared))
        else:
            resp = await self._execute(decision.spec, decision.level, req, decision.classification)
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
        strategy: Strategy,
    ) -> tuple[RoutingDecision, Response | None]:
        if model is not None:  # 0. explicit model pin
            spec = model if isinstance(model, ModelSpec) else ModelSpec.parse(model)
            return RoutingDecision(spec, None, "model_pinned"), None
        if level is not None:  # 1. explicit level pin
            return RoutingDecision(self._spec_for_level(level), level, "level_pinned"), None
        if self._bypass is not None:  # 2. deterministic bypass (any tier)
            bypass_level = self._bypass(req)
            if bypass_level is not None:
                return RoutingDecision(
                    self._spec_for_level(bypass_level), bypass_level, "bypass"
                ), None
        if self._hard_signal(req):  # 3. hard-signal (UP-only)
            return RoutingDecision(
                self._spec_for_level(Level.HARD), Level.HARD, "hard_signal"
            ), None
        return await self._resolve_middle(req, strategy)  # 4. ambiguous middle

    async def _resolve_middle(
        self, req: Request, strategy: Strategy
    ) -> tuple[RoutingDecision, Response | None]:
        """Resolve the ambiguous middle: a caller-supplied classifier wins; else the strategy."""
        if self._classifier is not None:
            result = self._classifier(req)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return RoutingDecision(self._spec_for_level(result), result, "classifier"), None
        if strategy is Strategy.ROUTE_DEFAULT:
            level = self._config.default_level
            return RoutingDecision(
                self._spec_for_level(level), level, "strategy:route_default"
            ), None
        if strategy is Strategy.CLASSIFY:
            return await self._classify(req), None
        if strategy is Strategy.CLASSIFY_AND_ANSWER:
            return await self._classify_and_answer(req)
        raise NotImplementedError(f"strategy {strategy.value!r} (VERIFY_CASCADE) lands in M5")

    async def _classify(self, req: Request) -> RoutingDecision:
        """CLASSIFY: a cheap dedicated classifier model labels the level (cached + budget-guarded),
        then we dispatch to that tier. Classify usage is attributed on the decision."""
        spec = self._require_classifier()
        text = _request_text(req)
        if _approx_tokens(text) < self._config.min_tokens_to_classify:  # budget guard
            level = self._config.default_level
            return RoutingDecision(self._spec_for_level(level), level, "classify:below_budget")
        cache_key = _classify_cache_key(spec, text)
        cached = self._classify_cache.get(cache_key)
        if cached is not None:
            return RoutingDecision(self._spec_for_level(cached), cached, "classify:cached")
        provider = self._provider_for(spec)
        classify_req = Request(
            messages=[Message(role="user", content=text)],
            system=[_CLASSIFY_INSTRUCTIONS],
            response_format=_LEVEL_SCHEMA,
            max_tokens=_CLASSIFY_MAX_TOKENS,
        )
        resp = await provider.complete(spec, classify_req)
        level = _parse_classified_level(resp.text)
        self._classify_cache[cache_key] = level
        record = ModelUsage(spec.provider, spec.model, resp.usage)
        return RoutingDecision(self._spec_for_level(level), level, "classify", record)

    async def _classify_and_answer(self, req: Request) -> tuple[RoutingDecision, Response | None]:
        """CLASSIFY_AND_ANSWER: one call to the classifier model. It either answers directly, or
        emits the escalation sentinel — then we re-dispatch the original request to a stronger
        tier."""
        spec = self._require_classifier()
        provider = self._provider_for(spec)
        guided = replace(req, system=[_CLASSIFY_AND_ANSWER_GUIDANCE, *(req.system or [])])
        resp = await provider.complete(spec, guided)
        target = _parse_escalation(resp.text)
        if target is not None:  # declined → escalate; the classifier call still cost tokens
            record = ModelUsage(spec.provider, spec.model, resp.usage)
            decision = RoutingDecision(
                self._spec_for_level(target), target, "classify_and_answer:escalate", record
            )
            return decision, None
        return RoutingDecision(spec, None, "classify_and_answer:answered"), resp

    async def _execute(
        self,
        spec: ModelSpec,
        level: Level | None,
        req: Request,
        classification: ModelUsage | None = None,
    ) -> Response:
        """Run the answer with resilience: try the model (retrying transient errors), and on
        persistent failure fall back to the next-higher tier. ``AuthError`` never retries or
        falls back — it's a configuration problem."""
        last: MaslulError | None = None
        for fallback_spec in self._fallback_chain(spec, level):
            for attempt in range(self._config.max_retries + 1):
                try:
                    return await self._execute_once(fallback_spec, req, classification)
                except AuthError:
                    raise
                except (RateLimited, Timeout) as e:  # transient → retry this model, then fall back
                    last = e
                    for cb in self._on_error:
                        cb(req, fallback_spec, e)
                    if attempt < self._config.max_retries:
                        await self._sleep_backoff(attempt)
                        continue
                    break
                except ProviderError as e:  # not retryable here → fall back to the next model
                    last = e
                    for cb in self._on_error:
                        cb(req, fallback_spec, e)
                    break
        raise last if last is not None else ProviderError("no model available to execute request")

    async def _execute_once(
        self, spec: ModelSpec, req: Request, classification: ModelUsage | None
    ) -> Response:
        provider = self._provider_for(spec)
        ledger: dict[tuple[str, str], Usage] = {}
        if classification is not None:
            _record_usage(
                ledger, classification.provider, classification.model, classification.usage
            )
        if not req.tools or req.tool_executor is None:
            resp = await self._invoke(provider, spec, req)
            _record(ledger, resp)
        else:
            resp = await self._run_tool_loop(spec, provider, req, ledger)
        return self._finalize(req, resp, ledger, classification)

    async def _invoke(self, provider: Provider, spec: ModelSpec, req: Request) -> Response:
        """One provider call, bounded by the per-call timeout (mapped to a retryable Timeout)."""
        if self._config.request_timeout is None:
            return await provider.complete(spec, req)
        try:
            return await asyncio.wait_for(
                provider.complete(spec, req), self._config.request_timeout
            )
        except TimeoutError as e:
            raise Timeout(f"provider call exceeded {self._config.request_timeout}s") from e

    async def _sleep_backoff(self, attempt: int) -> None:
        base = self._config.retry_base_delay
        delay = min(self._config.retry_max_delay, base * (2**attempt))
        await asyncio.sleep(delay + random.uniform(0, delay))  # full jitter

    def _fallback_chain(self, spec: ModelSpec, level: Level | None) -> Sequence[ModelSpec]:
        """The primary spec, then (when fallback is enabled and a tier was chosen) each
        higher tier — which may be a different provider, giving cross-provider resilience."""
        chain = [spec]
        if self._config.fallback and level is not None:
            for higher in Level:
                if higher > level:
                    tier = self._config.tiers.get(higher)
                    if tier is not None and tier not in chain:
                        chain.append(tier)
        return chain

    def _finalize(
        self,
        req: Request,
        resp: Response,
        ledger: dict[tuple[str, str], Usage],
        classification: ModelUsage | None = None,
    ) -> Response:
        _apply_structured(req, resp)
        resp.usage = _total(ledger)
        resp.usage_records = [ModelUsage(p, m, u) for (p, m), u in ledger.items()]
        if classification is not None:
            resp.classification_usage = classification.usage
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
            resp = await self._invoke(provider, spec, replace(req, messages=messages))
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

    def _require_classifier(self) -> ModelSpec:
        if self._config.classifier is None:
            raise ConfigError("this strategy needs a [maslul.classifier] config entry")
        return self._config.classifier

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
    text = _request_text(req)
    if len(text) > _LONG_CONTEXT_CHARS or "```" in text:
        return True
    return bool(_HARD_SIGNAL_KEYWORDS.search(text))


def _request_text(req: Request) -> str:
    return "\n".join(m.content for m in req.messages if m.content)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough; the budget guard only needs an order-of-magnitude


def _classify_cache_key(spec: ModelSpec, text: str) -> str:
    digest = hashlib.sha256(text.encode()).hexdigest()
    return f"{spec.provider}:{spec.model}:{digest}"


def _parse_classified_level(text: str) -> Level:
    """Read a classifier reply into a Level. Unknown/garbled → HARD (escalate up, never down)."""
    try:
        name = json.loads(text).get("level", "")
    except (ValueError, TypeError, AttributeError):
        name = text
    return _LEVEL_BY_NAME.get(str(name).strip().lower(), Level.HARD)


def _parse_escalation(text: str) -> Level | None:
    """The escalation sentinel names a target level; tolerate leading whitespace + a trailing
    reason line. Returns None when the text is a real answer (no sentinel)."""
    match = _ESCALATE_RE.match(text or "")
    if match is None:
        return None
    return _LEVEL_BY_NAME.get(match.group(1).lower(), Level.HARD)


def _ledger_of(resp: Response) -> dict[tuple[str, str], Usage]:
    ledger: dict[tuple[str, str], Usage] = {}
    _record(ledger, resp)
    return ledger


def _record(ledger: dict[tuple[str, str], Usage], resp: Response) -> None:
    _record_usage(ledger, resp.provider, resp.model, resp.usage)


def _record_usage(
    ledger: dict[tuple[str, str], Usage], provider: str, model: str, usage: Usage
) -> None:
    _accumulate(ledger.setdefault((provider, model), Usage()), usage)


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
