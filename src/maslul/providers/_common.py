"""Small translation helpers shared by the provider drivers. Imports only core types, never
an SDK, so it doesn't affect the optional-extras isolation."""

from __future__ import annotations

from maslul.types import ContextCache, MediaPart, Message


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
    *prefix* — Anthropic explicitly, Gemini/OpenAI/Grok implicitly — so a 100k-token PDF parked
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

    OpenAI, Grok and Gemini all report a prompt total that already *contains* the cached tokens;
    Anthropic reports them separately. maslul normalizes on Anthropic's shape — ``input_tokens``
    means "billed at full price" — so the caller can price any provider with one formula
    (see :class:`~maslul.Usage`). Returns ``(billable, cached)``.

    Clamped: a provider that ever reports ``cached > total`` must not yield negative billable
    tokens.
    """
    return max(total - cached, 0), cached
