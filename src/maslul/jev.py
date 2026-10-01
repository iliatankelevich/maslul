"""TypeSafe Jev — optional :data:`~maslul.types.Classifier` / :data:`~maslul.types.Verifier` hooks
backed by TypeSafe's System One API (https://docs.typesafe.ai).

Jev is **not an LLM provider** — it returns typed, calibrated decisions (here: a ``choice`` between
the router's levels, and a ``noul`` yes/no probability), never free text, so it cannot serve a
:class:`~maslul.providers.base.Provider` completion. It plugs into the *hook* contract instead:
:class:`JevClassifier` is a :data:`~maslul.types.Classifier` (step 4 of routing,
``Router(classifier=...)``), and :class:`JevVerifier` is a :data:`~maslul.types.Verifier`
(``VERIFY_CASCADE``, ``Router(verifier=...)``).

**English is Jev's primary training language; Hebrew and other languages are "handled but not
equally well"** (TypeSafe's own docs). Both hooks default to a conservative confidence gate, never
raise into the router on failure, and expose :meth:`JevClassifier.decide` /
:meth:`JevVerifier.judge` so a caller can log Jev's raw decisions in shadow mode and calibrate
``min_confidence`` / ``min_yes`` on real traffic before acting on them.

Requires the ``maslul[jev]`` extra (``httpx``) — not imported by :mod:`maslul` itself, so core
``import maslul`` stays free of it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from maslul.errors import ConfigError
from maslul.types import Classifier, Level, Request, Response, Usage, Verifier

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.typesafe.ai"
_DEFAULT_MODEL = "jev-latest"
_DEFAULT_TIMEOUT = 5.0  # on the hot path of every chat turn; fail fast and defer, don't stall it

_LEVEL_OPTION = {Level.SIMPLE: "simple", Level.MEDIUM: "medium", Level.HARD: "hard"}
_OPTION_LEVEL = {v: k for k, v in _LEVEL_OPTION.items()}

# Concrete examples per level, not adjectives. Abstract descriptions ("moderate reasoning", "deep
# reasoning") never let HARD win: measured 2026-10-01, a week-by-week moving plan, a savings split
# and an insurance comparison all came back MEDIUM at 0.48-0.78, in English and Hebrew alike. With
# these, the same requests came back HARD at 0.97-0.99 and the easy ones stayed SIMPLE. The wording
# moves the decision far more than the language does, which is why it is configurable.
_DEFAULT_LEVEL_CRITERIA: dict[Level, str] = {
    Level.SIMPLE: (
        "Small talk, thanks or a greeting; a single fact or definition; a short translation; "
        "a one-step action such as setting a reminder or editing a list."
    ),
    Level.MEDIUM: (
        "A short explanation; drafting a short message; looking something up in the user's own "
        "records; a recommendation among a few options."
    ),
    Level.HARD: (
        "Any request that needs a multi-part plan, a comparison of several options with "
        "trade-offs, financial, legal or medical advice, or reasoning over many documents or "
        "messages — even when it is phrased briefly."
    ),
}
_DEFAULT_CLASSIFY_QUESTION = (
    "How much reasoning capability does an assistant need to answer the user's latest message in "
    "this conversation CORRECTLY? Judge intrinsic difficulty, not how long the answer should be — "
    "a short message can be very hard, and a long paste can be trivial."
)
_DEFAULT_VERIFY_QUESTION = (
    "Does the assistant's reply fully and correctly handle the user's latest message in this "
    "conversation?"
)

_DEFAULT_HISTORY_WINDOW = 6
_DEFAULT_STATE_CHAR_CAP = 20_000  # well under Jev's 32k-token state budget, in characters


class JevClient:
    """Thin async HTTP client for ``POST /v1/systemone``.

    Not a :class:`~maslul.providers.base.Provider` (see module docstring). Construct one and share
    it between a :class:`JevClassifier` and a :class:`JevVerifier` to reuse the connection pool.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "TYPESAFE_API_KEY",
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        timeout: float = _DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """``api_key_env`` names the environment variable Jev's own docs use
        (``TYPESAFE_API_KEY``) by default; override it to point at a different one. Pass
        ``client`` or ``transport`` to inject a fake for tests — ``client`` takes an already-built
        ``httpx.AsyncClient`` (you own its auth header); ``transport`` is spliced into a client
        this constructor builds (base URL, timeout and the bearer header still apply)."""
        self.model = model
        if client is not None:
            self._client = client
            self._owns_client = False
            return
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"no Jev API key (pass api_key= or set {api_key_env})")
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}"},
            transport=transport,
        )
        self._owns_client = True

    async def systemone(
        self, state: str | Mapping[str, Any] | list[Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """``POST /v1/systemone``. Raises ``httpx.HTTPError`` on a transport failure or a non-2xx
        response — callers (:class:`JevClassifier` / :class:`JevVerifier`) catch it and defer."""
        resp = await self._client.post(
            "/v1/systemone", json={"state": state, "model": self.model, "questions": questions}
        )
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


@dataclass(frozen=True)
class JevClassification:
    """Jev's raw answer to the classify Choice question — returned by :meth:`JevClassifier.decide`
    for shadow-mode logging/calibration *before* the confidence gate in ``__call__`` is applied.

    ``level`` is Jev's top choice whenever the call succeeded, even below ``min_confidence`` —
    that is the point of shadow mode: see what Jev would have said. It is ``None`` only when the
    call failed or came back malformed.
    """

    level: Level | None
    probabilities: dict[Level, float]
    confidence: float | None
    model: str | None
    usage: Usage | None


@dataclass(frozen=True)
class JevVerdict:
    """Jev's raw answer to the verify Noul question — returned by :meth:`JevVerifier.judge` for
    shadow-mode logging before the ``min_yes`` gate in ``__call__`` is applied. ``p_yes`` is
    ``None`` only when the call failed or came back malformed."""

    p_yes: float | None
    model: str | None
    usage: Usage | None


class JevClassifier:
    """A :data:`~maslul.types.Classifier` backed by Jev's ``choice`` primitive: one Choice question
    whose options are the router's three levels.

    Any transport error, timeout, rate limit (429), overload (529), or an answer that doesn't parse
    as expected is logged as a warning and resolved as ``None`` — deferring to the router's
    configured :class:`~maslul.Strategy`. An outage of this optional classifier must never break
    routing.
    """

    def __init__(
        self,
        client: JevClient | None = None,
        *,
        min_confidence: float = 0.7,
        level_criteria: Mapping[Level, str] | None = None,
        question: str = _DEFAULT_CLASSIFY_QUESTION,
        history_window: int = _DEFAULT_HISTORY_WINDOW,
        state_char_cap: int = _DEFAULT_STATE_CHAR_CAP,
        on_decision: Callable[[JevClassification], None] | None = None,
        **client_kwargs: Any,
    ) -> None:
        """``min_confidence`` gates ``__call__`` only — :meth:`decide` always reports Jev's raw
        pick, so a caller can log it in shadow mode and tune the threshold (per language — Hebrew
        and other non-English traffic needs its own, likely higher, bar) before trusting it.
        Extra keyword args build a :class:`JevClient` when ``client`` is omitted."""
        self._client = client or JevClient(**client_kwargs)
        self._min_confidence = min_confidence
        self._criteria = dict(level_criteria) if level_criteria else _DEFAULT_LEVEL_CRITERIA
        self._question = question
        self._history_window = history_window
        self._state_char_cap = state_char_cap
        self._on_decision = on_decision

    async def decide(self, req: Request) -> JevClassification:
        """Run the classify question and return Jev's raw decision, uncapped by
        ``min_confidence`` — the shadow-mode entry point."""
        state = _conversation_state(req, self._history_window, self._state_char_cap)
        questions = {
            "level": {
                "type": "choice",
                "instructions": self._question,
                "criteria": {_LEVEL_OPTION[lvl]: desc for lvl, desc in self._criteria.items()},
            }
        }
        try:
            raw = await self._client.systemone(state, questions)
            decision = _parse_classification(raw)
        except Exception as e:  # noqa: BLE001 - any failure defers to the strategy, never raises
            logger.warning("Jev classify call failed: %s", e)
            decision = JevClassification(
                level=None, probabilities={}, confidence=None, model=None, usage=None
            )
        if self._on_decision is not None:
            self._on_decision(decision)
        return decision

    async def __call__(self, req: Request) -> Level | None:
        decision = await self.decide(req)
        if decision.level is None or decision.confidence is None:
            return None
        if decision.confidence < self._min_confidence:
            return None
        return decision.level


class JevVerifier:
    """A :data:`~maslul.types.Verifier` backed by Jev's ``noul`` (yes/no probability) primitive —
    for ``VERIFY_CASCADE``: does the cheap tier's candidate reply hold up?

    On failure, ``on_error`` decides the outcome: the default ``"accept"`` means an outage of this
    quality check never doubles cost/latency by escalating every turn while Jev is down.
    """

    def __init__(
        self,
        client: JevClient | None = None,
        *,
        min_yes: float = 0.7,
        question: str = _DEFAULT_VERIFY_QUESTION,
        history_window: int = _DEFAULT_HISTORY_WINDOW,
        state_char_cap: int = _DEFAULT_STATE_CHAR_CAP,
        on_error: Literal["accept", "reject"] = "accept",
        on_decision: Callable[[JevVerdict], None] | None = None,
        **client_kwargs: Any,
    ) -> None:
        self._client = client or JevClient(**client_kwargs)
        self._min_yes = min_yes
        self._question = question
        self._history_window = history_window
        self._state_char_cap = state_char_cap
        self._on_error = on_error
        self._on_decision = on_decision

    async def judge(self, req: Request, resp: Response) -> JevVerdict:
        """Run the verify question and return Jev's raw ``p_yes``, uncapped by ``min_yes`` — the
        shadow-mode entry point."""
        base = _conversation_state(req, self._history_window, self._state_char_cap)
        state = _append_reply(base, resp.text, self._state_char_cap)
        questions = {"ok": {"type": "noul", "instructions": self._question}}
        try:
            raw = await self._client.systemone(state, questions)
            verdict = _parse_verdict(raw)
        except Exception as e:  # noqa: BLE001 - any failure falls back to on_error, never raises
            logger.warning("Jev verify call failed: %s", e)
            verdict = JevVerdict(p_yes=None, model=None, usage=None)
        if self._on_decision is not None:
            self._on_decision(verdict)
        return verdict

    async def __call__(self, req: Request, resp: Response) -> bool:
        verdict = await self.judge(req, resp)
        if verdict.p_yes is None:
            return self._on_error == "accept"
        return verdict.p_yes >= self._min_yes


@dataclass(frozen=True)
class JevHooks:
    """Hooks built by :func:`hooks_from_config`; pass straight into ``Router(classifier=hooks
    .classifier, verifier=hooks.verifier)``. Either may be ``None`` when not requested, and
    ``client`` is ``None`` when neither was. Call :meth:`aclose` when the router is discarded."""

    classifier: Classifier | None
    verifier: Verifier | None
    client: JevClient | None = None

    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.aclose()


def hooks_from_config(config: Mapping[str, Any]) -> JevHooks:
    """Build :class:`JevClassifier` / :class:`JevVerifier` from a ``[maslul.jev]`` config table.

    ``Router``/``RouterConfig`` have no hook-plugin registry — ``classifier``/``verifier`` are plain
    constructor keywords, not names resolved from config — and one optional dependency does not
    justify adding one. So this factory reads the table and returns the hooks, and the caller passes
    them into ``Router(...)``, where they behave like any other injected hook. A hook the caller
    passes directly always wins, because ``Router`` never reads this table itself.

    ```toml
    [maslul.jev]
    api_key_env = "TYPESAFE_API_KEY"   # optional; this is Jev's own default
    model = "jev-latest"               # optional
    timeout = 5.0                      # optional, seconds
    classifier = true                  # build a JevClassifier
    min_confidence = 0.7               # optional, JevClassifier gate
    question = "..."                   # optional, the classify question
    verifier = true                    # build a JevVerifier
    min_yes = 0.7                      # optional, JevVerifier gate
    on_error = "accept"                # optional, JevVerifier failure policy
    verify_question = "..."            # optional, the verify question

    [maslul.jev.criteria]              # optional, what each level means; replaces the defaults
    simple = "..."
    medium = "..."
    hard = "..."
    ```

    Raises :class:`~maslul.errors.ConfigError` on a ``criteria`` key that is not a level, rather
    than silently classifying against a description nobody wrote.
    """
    root = config.get("maslul", config)
    raw = root.get("jev")
    # A table that asks for neither hook must not demand an API key just to build a client.
    if not raw or not (raw.get("classifier") or raw.get("verifier")):
        return JevHooks(classifier=None, verifier=None)
    criteria = _criteria_from_config(raw.get("criteria"))
    client_kwargs = {k: raw[k] for k in ("api_key_env", "base_url", "model", "timeout") if k in raw}
    client = JevClient(**client_kwargs)
    classifier = (
        JevClassifier(
            client,
            min_confidence=float(raw.get("min_confidence", 0.7)),
            level_criteria=criteria,
            question=raw.get("question", _DEFAULT_CLASSIFY_QUESTION),
        )
        if raw.get("classifier")
        else None
    )
    verifier = (
        JevVerifier(
            client,
            min_yes=float(raw.get("min_yes", 0.7)),
            on_error=raw.get("on_error", "accept"),
            question=raw.get("verify_question", _DEFAULT_VERIFY_QUESTION),
        )
        if raw.get("verifier")
        else None
    )
    return JevHooks(classifier=classifier, verifier=verifier, client=client)


def _criteria_from_config(raw: Mapping[str, Any] | None) -> dict[Level, str] | None:
    if not raw:
        return None
    unknown = set(raw) - set(_OPTION_LEVEL)
    if unknown:
        raise ConfigError(f"[maslul.jev.criteria] has unknown level(s): {sorted(unknown)}")
    return {_OPTION_LEVEL[name]: str(text) for name, text in raw.items()}


def _conversation_state(req: Request, history_window: int, char_cap: int) -> str:
    """Text-only transcript of the last ``history_window`` turns (media is dropped — Jev is
    text-only), tail-truncated to ``char_cap`` so the latest message always survives the cut."""
    turns = [m for m in req.messages if m.content]
    if history_window > 0:
        turns = turns[-history_window:]
    text = "\n".join(f"{m.role}: {m.content}" for m in turns)
    return text[-char_cap:] if len(text) > char_cap else text


def _append_reply(state: str, reply: str, char_cap: int) -> str:
    combined = f"{state}\nassistant (candidate reply): {reply}"
    return combined[-char_cap:] if len(combined) > char_cap else combined


def _parse_classification(raw: dict[str, Any]) -> JevClassification:
    answer = raw["answers"]["level"]
    if answer.get("type") != "choice":
        raise ValueError(f"expected a choice answer, got {answer.get('type')!r}")
    choice = answer["choice"]
    level = _OPTION_LEVEL[choice]  # KeyError on an unknown option -> caught, treated as a failure
    probabilities = {
        _OPTION_LEVEL[opt]: p for opt, p in answer["probabilities"].items() if opt in _OPTION_LEVEL
    }
    return JevClassification(
        level=level,
        probabilities=probabilities,
        confidence=float(answer["confidence"]),
        model=raw.get("model"),
        usage=_usage(raw.get("usage")),
    )


def _parse_verdict(raw: dict[str, Any]) -> JevVerdict:
    answer = raw["answers"]["ok"]
    if answer.get("type") != "noul":
        raise ValueError(f"expected a noul answer, got {answer.get('type')!r}")
    return JevVerdict(
        p_yes=float(answer["noul"]), model=raw.get("model"), usage=_usage(raw.get("usage"))
    )


def _usage(raw: Mapping[str, Any] | None) -> Usage | None:
    if raw is None:
        return None
    return Usage(
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
    )
