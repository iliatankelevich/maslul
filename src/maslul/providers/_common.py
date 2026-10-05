"""Small translation helpers shared by the provider drivers. Imports only core types, never
an SDK, so it doesn't affect the optional-extras isolation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from maslul.errors import ConfigError
from maslul.types import (
    KNOWN_PROVIDERS,
    REASONING_LEVELS,
    TOOL_CHOICE_MODES,
    ContextCache,
    MediaPart,
    Message,
    ModelSpec,
    Reasoning,
    Request,
    ToolChoice,
)


@dataclass(frozen=True)
class GenerationParams:
    """The generic parameters for one model call, resolved: the request's value, else the tier's
    default. Each provider translates these into its own SDK keywords."""

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    reasoning: Reasoning | None = None
    tool_choice: ToolChoice | None = None


def generation_params(spec: ModelSpec, req: Request) -> GenerationParams:
    """Resolve ``req``'s generic parameters over ``spec``'s defaults (a request value wins).

    Raises :class:`~maslul.ConfigError` for a ``reasoning`` level outside
    :data:`~maslul.types.REASONING_LEVELS`, or a ``tool_choice`` that names a tool the request
    does not define — both would otherwise surface as a provider 400 with a less useful message.
    """
    reasoning = req.reasoning if req.reasoning is not None else spec.reasoning
    if reasoning is not None and reasoning not in REASONING_LEVELS:
        raise ConfigError(
            f"unknown reasoning level {reasoning!r} — expected one of {sorted(REASONING_LEVELS)}"
        )
    choice = req.tool_choice
    if (
        choice is not None
        and choice not in TOOL_CHOICE_MODES
        and choice not in {t.name for t in req.tools or []}
    ):
        raise ConfigError(f"tool_choice {choice!r} names no tool in the request")
    return GenerationParams(
        max_tokens=req.max_tokens or spec.max_tokens,
        temperature=req.temperature if req.temperature is not None else spec.temperature,
        top_p=req.top_p if req.top_p is not None else spec.top_p,
        top_k=req.top_k if req.top_k is not None else spec.top_k,
        seed=req.seed if req.seed is not None else spec.seed,
        stop=req.stop,
        reasoning=reasoning,
        tool_choice=choice,
    )


def raw_options(provider: str, options: Mapping[str, Any], *, unscoped: bool) -> dict[str, Any]:
    """The raw SDK keywords ``provider`` should merge from ``Request.provider_options``.

    That is its own scoped entry (``{"gemini": {...}}``), on top of — only when ``unscoped`` — the
    legacy unscoped keys. Another provider's entry is never returned. Anthropic and OpenAI pass
    ``unscoped=True`` for back-compat; Gemini and xAI never applied unscoped keys, and callers put
    Anthropic-shaped values there (a content-block ``system``), so they pass ``False``.
    """
    out = {k: v for k, v in options.items() if k not in KNOWN_PROVIDERS} if unscoped else {}
    out.update(options.get(provider) or {})
    return out


def last_user_index(messages: list[Message]) -> int:
    """Index of the last ``user``-role message, or -1 if none."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            return i
    return -1


def first_user_index(messages: list[Message]) -> int:
    """Index of the first ``user``-role message, or -1 if none."""
    for i, m in enumerate(messages):
        if m.role == "user":
            return i
    return -1


def media_index(
    messages: list[Message],
    media: list[MediaPart] | None,
    cache: ContextCache | None,
) -> int:
    """Which message ``media`` attaches to — the one lever that works on every provider.

    Default (**no context cache**): the **last** user message, preserving the historical
    behaviour byte for byte.

    With ``ContextCache(media=True)``: the **first** user message. Every provider caches a
    *prefix* — Anthropic explicitly, Gemini/OpenAI/xAI implicitly — so a 100k-token PDF parked
    after the conversation shifts position on every new turn and re-bills in full each time.
    Moving it to the front makes it part of the stable prefix. Single-turn requests are
    unaffected (first user message *is* the last one).
    """
    if not media:
        return -1
    if cache is not None and cache.media:
        return first_user_index(messages)
    return last_user_index(messages)


def media_first(cache: ContextCache | None) -> bool:
    """Whether media blocks are emitted **before** the message's text, inside the message they
    attach to.

    Default (**no context cache**): ``False`` — text then media, exactly as before.

    With ``ContextCache(media=True)``: ``True``. This is the other half of stable-first layout, and
    without it the first half is worthless on the flow that matters most. A document-QA turn is a
    *single* user message holding both the question and the PDF, so "attach media to the first user
    message" is a no-op there (the first user message *is* the last one) — but the rendered blocks
    are ``[text(question), document(pdf)]``, which puts the **volatile question inside the prefix**
    that any breakpoint on the document would cache. Every question is then a fresh prefix and the
    cache never hits. Emitting ``[document(pdf), text(question)]`` puts the stable 100k-token
    document first, where a prefix cache can actually reach it.

    Anthropic independently recommends placing a ``document`` block before the text block, so this
    is not a quality trade-off — but it *is* a wire-format change, hence gated behind ``media``.
    """
    return cache is not None and cache.media


def split_input(total: int, cached: int) -> tuple[int, int]:
    """A cached-*inclusive* prompt total → maslul's disjoint :class:`~maslul.Usage` convention.

    OpenAI, xAI and Gemini all report a prompt total that already *contains* the cached tokens;
    Anthropic reports them separately. maslul normalizes on Anthropic's shape — ``input_tokens``
    means "billed at full price" — so the caller can price any provider with one formula
    (see :class:`~maslul.Usage`). Returns ``(billable, cached)``.

    Clamped: a provider that ever reports ``cached > total`` must not yield negative billable
    tokens.
    """
    return max(total - cached, 0), cached
