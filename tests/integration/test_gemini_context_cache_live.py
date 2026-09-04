"""Live Gemini explicit-caching tests — the only ones that prove phase 4 works.

A unit test can assert that ``cached_content`` was sent instead of ``system_instruction``; it cannot
assert that Vertex *honoured* it, that ``google_search`` survives being cached, or that the 4,096
floor behaves as documented. Those need real calls.

Gated on ``VERTEX_PROJECT`` (plus working ADC — ``gcloud auth application-default login``) and
skipped cleanly without it. Each test costs a fraction of a cent.

⚠️ The prefixes here are deliberately sized around the **4,096-token floor**, because that boundary
is the whole reason phase 4 exists: below it explicit caching is refused, and between it and roughly
10,000 tokens implicit caching does nothing, so explicit is the only thing that works at all.
"""

from __future__ import annotations

import os

import pytest

from maslul.providers.gemini import GeminiProvider
from maslul.types import ContextCache, Message, ModelSpec, Request, ToolDef

requires_vertex = pytest.mark.skipif(
    not os.getenv("VERTEX_PROJECT"), reason="VERTEX_PROJECT not set"
)

_MODEL = os.getenv("MASLUL_TEST_GEMINI_MODEL", "gemini-3.8-flash")

# One repeated sentence, so the prefix is byte-stable across runs — a single varying character
# would invalidate the cache and fail these tests for the wrong reason.
_SENTENCE = (
    "The administrator shall retain every notice of loss for a period of seven years, "
    "and shall make it available to the insured party on request. "
)
_ABOVE_FLOOR = _SENTENCE * 420  # comfortably over 4,096 tokens
_BELOW_FLOOR = _SENTENCE * 40  # comfortably under it


def _provider() -> GeminiProvider:
    return GeminiProvider(
        vertex_project=os.environ["VERTEX_PROJECT"],
        vertex_location=os.getenv("VERTEX_LOCATION", "global"),
    )


def _req(system: str, **over: object) -> Request:
    kwargs: dict = {
        "messages": [Message(role="user", content="Reply with the single word: ok.")],
        "system": [system],
        "context_cache": ContextCache(system=True, ttl_seconds=120),
        "max_tokens": 32,
    }
    kwargs.update(over)
    return Request(**kwargs)


@requires_vertex
async def test_explicit_cache_is_read_back_on_the_second_call() -> None:
    """The whole point: two calls, and the second is served from the cache.

    Implicit caching returns 0 at this size — that is what makes this test meaningful rather than
    a restatement of what Gemini would have done anyway.
    """
    provider = _provider()
    spec = ModelSpec(provider="gemini", model=_MODEL)

    first = await provider.complete(spec, _req(_ABOVE_FLOOR))
    second = await provider.complete(spec, _req(_ABOVE_FLOOR))

    assert second.usage.cache_read_input_tokens > 4000, (
        f"expected the prefix to be served from cache, got "
        f"{second.usage.cache_read_input_tokens} (first call: "
        f"{first.usage.cache_read_input_tokens})"
    )
    # Disjoint accounting (phase 1): the billable remainder is the volatile tail only.
    assert second.usage.input_tokens < 100


@requires_vertex
async def test_a_prefix_under_the_floor_still_answers_at_full_price() -> None:
    """Below 4,096 tokens Vertex refuses to create the cache. That must cost money, not the request
    — the same contract as an Anthropic breakpoint below the minimum being silently ignored."""
    out = await _provider().complete(
        # Not asserting on `text`: a thinking model can spend a small budget entirely on thought and
        # return nothing, which would make this flaky for a reason that has nothing to do with
        # caching. The claim under test is "the call went through and was billed in full".
        ModelSpec(provider="gemini", model=_MODEL),
        _req(_BELOW_FLOOR, max_tokens=256),
    )
    assert out.usage.input_tokens > 500  # the whole prefix was billed
    assert out.usage.cache_read_input_tokens == 0  # nothing was cached


@requires_vertex
async def test_google_search_survives_being_cached() -> None:
    """``web_search`` moves *into* the cache, because Vertex rejects a request that sets both
    ``cached_content`` and ``tools``. If grounding stopped working there, it would be a silent
    regression — a cached search tool that never searches."""
    provider = _provider()
    spec = ModelSpec(provider="gemini", model=_MODEL)
    req = _req(
        _ABOVE_FLOOR,
        messages=[Message(role="user", content="Who won the 2026 FIFA World Cup? Search the web.")],
        web_search=True,
        max_tokens=1024,  # a starved budget ends the turn before the search resolves
    )
    await provider.complete(spec, req)  # warm the cache
    out = await provider.complete(spec, req)

    assert out.usage.cache_read_input_tokens > 4000
    assert out.sources, "a cached google_search tool must still ground the answer"


@requires_vertex
async def test_web_search_does_not_share_a_cache_with_a_non_search_request() -> None:
    """Two requests differing only in ``web_search`` must not share a handle: the search tool is
    inside the cache, so sharing would silently give one of them the other's tool set."""
    provider = _provider()
    spec = ModelSpec(provider="gemini", model=_MODEL)
    tool = ToolDef(name="noop", description="does nothing", input_schema={"type": "object"})

    await provider.complete(spec, _req(_ABOVE_FLOOR, tools=[tool], web_search=False))
    await provider.complete(spec, _req(_ABOVE_FLOOR, tools=[tool], web_search=True))

    # Two distinct cache handles for the two tool sets.
    assert len({name for name, _ in provider._handles.values()}) == 2  # noqa: SLF001
