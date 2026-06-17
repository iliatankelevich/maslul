"""Unit tests for the response cache (exact + semantic)."""

from __future__ import annotations

import pytest

import maslul.cache as cache_mod
from maslul.cache import CacheConfig, ResponseCache
from maslul.errors import ConfigError
from maslul.types import ModelUsage, Response, Usage


def _resp(text: str = "answer") -> Response:
    usage = Usage(input_tokens=10, output_tokens=5)
    return Response(
        text=text,
        level_used=None,
        provider="fake",
        model="m",
        usage=usage,
        usage_records=[ModelUsage("fake", "m", usage)],
    )


async def test_exact_hit_is_a_zeroed_cached_copy() -> None:
    cache = ResponseCache(CacheConfig(mode="exact"))
    assert await cache.get("k") is None  # miss
    await cache.put("k", _resp("hi"))
    hit = await cache.get("k")
    assert hit is not None
    assert hit.text == "hi" and hit.cached is True
    assert (hit.usage.input_tokens, hit.usage.output_tokens) == (0, 0)  # no new tokens spent
    assert hit.usage_records == []


async def test_exact_miss_on_different_key() -> None:
    cache = ResponseCache(CacheConfig(mode="exact"))
    await cache.put("a", _resp())
    assert await cache.get("b") is None


def test_off_mode_is_disabled() -> None:
    assert ResponseCache(CacheConfig(mode="off")).enabled is False
    assert ResponseCache(CacheConfig(mode="exact")).enabled is True


def test_unknown_mode_raises() -> None:
    with pytest.raises(ConfigError):
        ResponseCache(CacheConfig(mode="fuzzy"))


async def test_ttl_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: clock[0])
    cache = ResponseCache(CacheConfig(mode="exact", ttl_seconds=10))
    await cache.put("k", _resp())
    assert await cache.get("k") is not None  # within TTL
    clock[0] += 11
    assert await cache.get("k") is None  # expired


async def test_lru_eviction() -> None:
    cache = ResponseCache(CacheConfig(mode="exact", max_entries=2))
    await cache.put("a", _resp("A"))
    await cache.put("b", _resp("B"))
    await cache.get("a")  # touch a → b becomes least-recently-used
    await cache.put("c", _resp("C"))  # over capacity → evict b
    assert await cache.get("b") is None
    a, c = await cache.get("a"), await cache.get("c")
    assert a is not None and a.text == "A"
    assert c is not None and c.text == "C"


async def test_semantic_hit_above_threshold_and_miss_below() -> None:
    vectors = {"cat": [1.0, 0.0], "kitty": [0.99, 0.02], "dog": [0.0, 1.0]}

    def embed(text: str) -> list[float]:
        return vectors[text]

    cache = ResponseCache(CacheConfig(mode="semantic", similarity_threshold=0.9), embed=embed)
    await cache.put("cat", _resp("meow"))
    near = await cache.get("kitty")  # exact miss, but semantically close to "cat"
    assert near is not None and near.text == "meow" and near.cached is True
    assert await cache.get("dog") is None  # orthogonal → below threshold


async def test_semantic_async_embed_supported() -> None:
    async def embed(_text: str) -> list[float]:
        return [1.0, 0.0]

    cache = ResponseCache(CacheConfig(mode="semantic", similarity_threshold=0.5), embed=embed)
    await cache.put("x", _resp("hit"))
    got = await cache.get("y")  # different key, same embedding → semantic hit
    assert got is not None and got.text == "hit"


def test_semantic_requires_embed() -> None:
    with pytest.raises(ConfigError, match="embed"):
        ResponseCache(CacheConfig(mode="semantic"))


def test_cosine_helper() -> None:
    assert cache_mod._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cache_mod._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cache_mod._cosine([], [1.0]) == 0.0  # mismatched/empty → 0
