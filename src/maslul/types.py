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

from maslul.errors import ConfigError

#: Provider names Maslul knows how to dispatch to. The ``provider`` prefix of a
#: ``"provider:model"`` spec must be one of these.
KNOWN_PROVIDERS: frozenset[str] = frozenset({"anthropic", "gemini", "grok"})

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

        The same format as Kippy's ``[models]`` table: the ``provider`` prefix tells Maslul
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
    response_format: JsonSchema | None = None
    media: list[MediaPart] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stop: list[str] | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    """Token accounting, normalized across providers (cache fields are 0 when N/A)."""

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
    raw: Any = None


@dataclass(frozen=True)
class RoutingDecision:
    """Why the router picked this model — passed to the ``on_route`` hook for observability."""

    spec: ModelSpec
    level: Level | None  # None when a model was pinned directly
    reason: str  # model_pinned | level_pinned | bypass | hard_signal | strategy:<name>


#: Resolves the difficulty tier for a request, or returns None to defer to the configured
#: strategy. The caller's own classification method (may be sync or async).
Classifier = Callable[[Request], "Level | None | Awaitable[Level | None]"]
#: Deterministic fast-path: pick a tier with no model judgment (e.g. greetings → SIMPLE), or None.
BypassPredicate = Callable[[Request], "Level | None"]
#: UP-only escalation signal: True routes the request to HARD without a classifier call.
HardSignal = Callable[[Request], bool]
#: Observability hooks.
RouteHook = Callable[[Request, RoutingDecision], None]
CompleteHook = Callable[[Response], None]
