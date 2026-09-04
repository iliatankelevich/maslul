"""Core contracts — the normalized request/response types every provider speaks, plus
the routing primitives (:class:`Level`, :class:`Strategy`, :class:`ModelSpec`).

These are provider-agnostic by construction: a :class:`Request` / :class:`Response` pair
has the same shape whether it was served by Anthropic, Gemini, or Grok.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Literal

from maslul.errors import ConfigError, MaslulError

#: Provider names Maslul knows how to dispatch to. The ``provider`` prefix of a
#: ``"provider:model"`` spec must be one of these.
KNOWN_PROVIDERS: frozenset[str] = frozenset({"anthropic", "gemini", "grok", "openai"})

#: A JSON Schema document — used for structured output and tool input schemas.
JsonSchema = dict[str, Any]

#: A conversation role. Richer content blocks (tool results, media) arrive with the
#: tool-use loop in M2.
Role = Literal["user", "assistant", "tool"]


class Level(IntEnum):
    """Difficulty tier. ``IntEnum`` so ``SIMPLE < MEDIUM < HARD`` holds for escalation."""

    SIMPLE = 1
    MEDIUM = 2
    HARD = 3


class Strategy(StrEnum):
    """How the ambiguous middle (step 4 of the routing order) is resolved.

    Values match the strings used in the ``[maslul] strategy`` config key.
    """

    ROUTE_DEFAULT = "route_default"
    CLASSIFY = "classify"
    CLASSIFY_AND_ANSWER = "classify_and_answer"
    VERIFY_CASCADE = "verify_cascade"


@dataclass(frozen=True)
class ModelSpec:
    """A single resolved model: which provider's SDK to dispatch to, and the model id."""

    provider: str
    model: str
    max_tokens: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, spec: str) -> ModelSpec:
        """Parse the canonical ``"provider:model"`` string.

        The ``provider`` prefix tells Maslul
        which SDK to dispatch to. Raises :class:`ConfigError` on a malformed spec or an
        unknown provider.
        """
        provider, sep, model = spec.partition(":")
        if not sep or not model or provider not in KNOWN_PROVIDERS:
            raise ConfigError(f"bad model spec {spec!r} — expected 'provider:model'")
        return cls(provider=provider, model=model)


@dataclass
class Message:
    """One conversation turn.

    Plain text uses ``role`` + ``content``. Tool use (M2): an ``assistant`` turn may carry
    ``tool_calls``; a ``role="tool"`` turn holds the executor's output in ``content`` with the
    ``tool_call_id`` it answers (``name`` is the tool, for providers that match results by name).
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class MediaPart:
    """An image or PDF attachment: raw bytes plus its MIME type."""

    mime_type: str
    data: bytes


@dataclass
class ContextCache:
    """Declares which parts of a request are stable enough for a provider to reuse across calls.

    **This is prompt caching, not the response cache.** The response cache (``[maslul.cache]``,
    :mod:`maslul.cache`) answers *without calling the model at all*. A ``ContextCache`` still calls
    the model — it just asks the provider to bill the part of the prompt it has already seen at the
    cache-read rate (~0.1x on Anthropic). Different layer, and they compose.

    A **hint, not a command**: the caller declares *what is stable and for how long*, never a
    mechanism. Anthropic honours it with explicit ``cache_control`` breakpoints; Gemini, OpenAI and
    Grok cache a matching prefix automatically and honour it through **prompt layout** (stable
    content first) plus, where available, an affinity key. A provider that can do nothing with a
    field ignores it — the request still succeeds, it just costs full price. Savings are always
    reported the same way, in ``Usage.cache_read_input_tokens``.

    ⚠️ **The cache is model-scoped; the router picks the model.** A cache written on the ``simple``
    tier is cold on ``hard``, so an escalating strategy (``CLASSIFY_AND_ANSWER``,
    ``VERIFY_CASCADE``, or tier fallback) can pay the write premium twice for nothing. Pair
    ``media=True`` with a pinned model (``complete(req, model=...)`` or ``level=...``); the router
    does **not** second-guess you.

    ⚠️ **Any byte that changes early invalidates everything after it**, on every provider. A
    timestamp or a per-request id in ``system`` — or a tool set that varies per user — silently
    kills every downstream hit. Verify with ``Usage.cache_read_input_tokens``, not by inspection.
    """

    #: Cache the system prefix (persona + guidance). On Anthropic this also covers **tools**, which
    #: render before system. Cheap to re-write, so it stays on even when the model may escalate.
    system: bool = True
    #: Cache attached media — the expensive one (a 100k-token PDF). Also **moves media to the first
    #: user message** so it sits in the stable prefix rather than the most volatile slot; that is a
    #: behaviour change, which is why it is opt-in. Pair with a pinned model (see above).
    media: bool = False
    #: Cache the conversation prefix, for long multi-turn chats.
    history: bool = False
    #: ``None`` = the provider's default (Anthropic 5 min, OpenAI ~5–10 min). ``>= 3600`` opts into
    #: Anthropic's 1-hour TTL (2x write premium instead of 1.25x) and OpenAI's 24-hour retention.
    ttl_seconds: int | None = None
    #: Affinity key — a stable per-conversation or per-document id, routing like prompts to the
    #: same cache. Used by OpenAI (``prompt_cache_key``). **Ignored by Anthropic, Gemini and
    #: Grok**, which expose no such knob (see the provider docstrings). OpenAI needs roughly
    #: 15 req/min on a key to keep it warm, so a per-document key is useful, a per-message key is
    #: not.
    key: str | None = None


@dataclass
class ToolDef:
    """A tool the model may call: a name, a description, and a JSON-Schema input."""

    name: str
    description: str
    input_schema: JsonSchema


@dataclass
class ToolCall:
    """A model's request to invoke a tool, normalized across providers."""

    id: str
    name: str
    input: dict[str, Any]
    #: Opaque provider state that the model attached to THIS call and requires back, unmodified,
    #: when the tool loop replays the call alongside its result. maslul never interprets it.
    #:
    #: It exists because a tool call is not always reconstructible from ``(name, input)``. Gemini 3
    #: attaches a **thought signature** to the ``functionCall`` part; replay it without the
    #: signature and the *next* request is rejected outright — ``400 INVALID_ARGUMENT: Function
    #: call is missing a thought_signature in functionCall parts``. So a provider that mints one
    #: must capture it here, and re-attach it when it rebuilds the turn. Providers with no such
    #: state (Anthropic, OpenAI, Grok) leave it ``None``.
    signature: bytes | None = None


#: Runs a tool the model asked for and returns its result as text. Supplied by the caller;
#: the router drives the tool-use loop (M2).
ToolExecutor = Callable[[ToolCall], Awaitable[str]]


@dataclass
class Request:
    """A normalized completion request — the same shape for every provider."""

    messages: list[Message]
    system: list[str] | None = None
    tools: list[ToolDef] | None = None
    tool_executor: ToolExecutor | None = None
    # Raw provider-native server-side tool specs (e.g. Anthropic web search) the provider runs
    # itself — no client executor. Merged alongside ``tools``; unsupported providers ignore them.
    server_tools: list[dict[str, Any]] | None = None
    # Normalized web search: set ``web_search=True`` and every provider enables its own grounding
    # (Anthropic web_search tool, Gemini Google Search, Grok Live Search) — the caller doesn't pick
    # a provider-specific mechanism. Citations land in :attr:`Response.sources` uniformly.
    web_search: bool = False
    web_search_max_uses: int | None = None  # cap searches/turn where the provider supports it
    response_format: JsonSchema | None = None
    media: list[MediaPart] | None = None
    # Prompt caching (NOT the response cache — see ContextCache). Declares which parts of this
    # request are stable enough for the provider to reuse. ``None`` = today's behaviour, byte for
    # byte: no cache markers, no affinity key, and media stays on the last user message.
    context_cache: ContextCache | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stop: list[str] | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    """Token accounting, normalized across providers (cache fields are 0 when N/A).

    **The three input fields are disjoint** — each names a different *price*, so cost is one
    formula on every provider:

    - ``input_tokens`` — tokens billed at **full** price.
    - ``cache_read_input_tokens`` — tokens billed at the **discounted** cache-read price (~0.1x).
    - ``cache_creation_input_tokens`` — tokens billed at the **write premium** (Anthropic only).

    The prompt's true size is their **sum**; never add ``cache_read`` to ``input_tokens`` expecting
    a total-cost figure. Providers disagree natively — Anthropic reports ``input_tokens`` excluding
    cached tokens, while OpenAI, Grok and Gemini report a prompt total that *includes* them —
    so maslul subtracts the cached count on those three (``providers._common.split_input``) and
    normalizes on Anthropic's shape, the one that maps to money.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class ModelUsage:
    """Tokens attributed to one ``provider:model`` within a request — a request can span
    several (a classifier model, the answer model, tool-loop iterations). The per-model
    breakdown the usage-metrics hook reports."""

    provider: str
    model: str
    usage: Usage


@dataclass
class Response:
    """A normalized completion result."""

    text: str
    level_used: Level | None
    provider: str
    model: str
    usage: Usage  # total across every model call in the request (sum of usage_records)
    structured: Any | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    sources: list[str] = field(default_factory=list)
    classification_usage: Usage | None = None
    usage_records: list[ModelUsage] = field(default_factory=list)  # per-model breakdown
    cached: bool = False  # served from the response cache — zero new tokens were spent
    raw: Any = None


@dataclass(frozen=True)
class RoutingDecision:
    """Why the router picked this model — passed to the ``on_route`` hook for observability."""

    spec: ModelSpec
    level: Level | None  # None when a model was pinned directly or a classifier answered inline
    reason: str  # model_pinned | level_pinned | bypass | hard_signal | classifier | strategy:*
    classification: ModelUsage | None = None  # tokens spent on a separate classify call, if any


#: Resolves the difficulty tier for a request, or returns None to defer to the configured
#: strategy. The caller's own classification method (may be sync or async).
Classifier = Callable[[Request], "Level | None | Awaitable[Level | None]"]
#: Deterministic fast-path: pick a tier with no model judgment (e.g. greetings → SIMPLE), or None.
BypassPredicate = Callable[[Request], "Level | None"]
#: UP-only escalation signal: True routes the request to HARD without a classifier call.
HardSignal = Callable[[Request], bool]
#: Decides whether a cheap answer is good enough; False triggers escalation (VERIFY_CASCADE).
Verifier = Callable[[Request, Response], bool | Awaitable[bool]]
#: Observability hooks.
RouteHook = Callable[[Request, RoutingDecision], None]
CompleteHook = Callable[[Response], None]
#: Fired on each failed model attempt (a retry or a fallback to another model).
ErrorHook = Callable[[Request, ModelSpec, MaslulError], None]
