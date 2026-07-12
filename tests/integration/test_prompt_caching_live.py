"""Live prompt-caching tests — the only ones that prove the feature works.

A unit test can assert that a ``cache_control`` marker landed on the right block; it cannot assert
that the provider *honoured* it. That takes two real calls: the first writes the cache, the second
must read it back. Everything else is testing our own mock.

Gated on ``ANTHROPIC_API_KEY`` and skipped cleanly without it. Each test costs a few cents.
"""

from __future__ import annotations

import os

import pytest

from maslul.providers.anthropic import AnthropicProvider
from maslul.types import ContextCache, Message, ModelSpec, Request

requires_anthropic = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)

# Sonnet's minimum cacheable prefix is 2,048 tokens (Opus wants 4,096) — below it the API silently
# ignores the marker. This clears both with room to spare, and is byte-stable across runs: a single
# varying character anywhere in the prefix would invalidate the cache and fail the test for the
# wrong reason.
_STABLE_SYSTEM = (
    "You are a meticulous claims adjuster. Reference material follows.\n\n"
    + "\n".join(
        f"Clause {i}: The insured party shall notify the administrator within thirty days of any "
        f"qualifying event. Failure to do so may result in partial forfeiture of benefits under "
        f"section {i}, subject to the exceptions enumerated in the appendix."
        for i in range(400)
    )
)
_MODEL = ModelSpec(
    provider="anthropic", model=os.getenv("MASLUL_ANTHROPIC_MODEL", "claude-sonnet-4-6")
)


def _ask(question: str, cache: ContextCache | None) -> Request:
    return Request(
        messages=[Message(role="user", content=question)],
        system=[_STABLE_SYSTEM],
        context_cache=cache,
        max_tokens=16,
    )


@requires_anthropic
async def test_anthropic_writes_then_reads_the_prompt_cache() -> None:
    """The acceptance test from the plan: send the same large prefix twice; the second call must be
    served from cache. Also proves the disjoint-``Usage`` convention on real numbers — the cached
    tokens land in ``cache_read_input_tokens``, NOT in ``input_tokens``."""
    provider = AnthropicProvider()
    cache = ContextCache(system=True)

    first = await provider.complete(_MODEL, _ask("How many days to notify?", cache))
    # The write may already be a read if a previous run warmed it inside the 5-minute TTL — either
    # way the prefix must be accounted for as cached, not billed at full price.
    assert first.usage.cache_creation_input_tokens + first.usage.cache_read_input_tokens > 2000

    second = await provider.complete(_MODEL, _ask("What is in the appendix?", cache))
    assert second.usage.cache_read_input_tokens > 2000, "second call was not served from cache"
    # The disjoint convention: only the question itself is billed at full price.
    assert second.usage.input_tokens < 100, (
        f"input_tokens={second.usage.input_tokens} — the cached prefix leaked into the "
        "full-price bucket, so Usage is not disjoint"
    )


@requires_anthropic
async def test_anthropic_does_not_cache_without_a_context_cache() -> None:
    """The opt-in guarantee, live: no ContextCache → no marker → the provider caches nothing, even
    though the prefix is identical to the test above and is sitting warm in Anthropic's cache."""
    resp = await AnthropicProvider().complete(_MODEL, _ask("How many days to notify?", None))
    assert resp.usage.cache_creation_input_tokens == 0
    assert resp.usage.cache_read_input_tokens == 0
    assert resp.usage.input_tokens > 2000  # the whole prefix, billed at full price
