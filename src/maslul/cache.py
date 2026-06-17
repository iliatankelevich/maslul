"""Response cache - a cost lever that returns a prior :class:`Response` instead of calling a model.

Two modes, built over the same prompt-hash the CLASSIFY strategy already uses:

- ``exact``    - an identical request (same prompt/system/schema/pins) returns the stored response.
- ``semantic`` - on an exact miss, embed the request and return the nearest stored response whose
  cosine similarity clears ``similarity_threshold``. Needs an injected ``embed`` function (maslul
  has no embeddings of its own), so it stays provider-agnostic.

In-memory, LRU-bounded (``max_entries``) with an optional ``ttl_seconds``. A cache hit is returned
as a copy with **zeroed usage** and ``cached=True`` - so observability sees the saving, not a
phantom re-spend. The router only caches **tool-free** completions (a cached ``tool_call`` would be
re-executed against stale state).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from maslul.errors import ConfigError
from maslul.types import Response, Usage

#: Embeds a request's text to a vector for semantic lookup. Sync or async; caller-supplied.
Embedder = Callable[[str], "list[float] | Awaitable[list[float]]"]


@dataclass
class CacheConfig:
    """``[maslul.cache]`` config. ``mode`` off/exact/semantic; ``similarity_threshold`` is the
    cosine cut-off for a semantic hit (0-1, higher = stricter)."""

    mode: str = "off"
    max_entries: int = 512
    ttl_seconds: float | None = None
    similarity_threshold: float = 0.95


@dataclass
class _Entry:
    response: Response
    embedding: list[float] | None = None
    expires_at: float | None = None


class ResponseCache:
    """In-memory exact + (optionally) semantic response cache. Not safe to share across event
    loops, but fine for a single asyncio app: ``get``/``put`` only await the injected embedder."""

    def __init__(self, config: CacheConfig, embed: Embedder | None = None) -> None:
        if config.mode not in ("off", "exact", "semantic"):
            raise ConfigError(f"unknown cache mode {config.mode!r} (off|exact|semantic)")
        if config.mode == "semantic" and embed is None:
            raise ConfigError("semantic cache needs an embed function - Router(embed=...)")
        self._config = config
        self._embed = embed
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._config.mode in ("exact", "semantic")

    async def get(self, key: str) -> Response | None:
        """Return a cached response for ``key`` (exact, then semantic), or ``None``."""
        now = time.monotonic()
        digest = _hash(key)
        entry = self._entries.get(digest)
        if entry is not None and not _expired(entry, now):
            self._entries.move_to_end(digest)
            return _serve(entry.response)
        if self._config.mode == "semantic" and self._embed is not None:
            vector = await _resolve(self._embed(key))
            best = self._nearest(vector, now)
            if best is not None:
                return _serve(best)
        return None

    async def put(self, key: str, response: Response) -> None:
        """Store ``response`` under ``key`` (embedding it too, in semantic mode)."""
        embedding = None
        if self._config.mode == "semantic" and self._embed is not None:
            embedding = await _resolve(self._embed(key))
        ttl = self._config.ttl_seconds
        digest = _hash(key)
        self._entries[digest] = _Entry(
            response=replace(response),  # snapshot - later caller mutations don't leak in
            embedding=embedding,
            expires_at=(time.monotonic() + ttl) if ttl else None,
        )
        self._entries.move_to_end(digest)
        while len(self._entries) > self._config.max_entries:
            self._entries.popitem(last=False)  # evict the least-recently-used

    def _nearest(self, vector: list[float], now: float) -> Response | None:
        best_response: Response | None = None
        best_sim = self._config.similarity_threshold
        for entry in list(self._entries.values()):
            if entry.embedding is None or _expired(entry, now):
                continue
            sim = _cosine(vector, entry.embedding)
            if sim >= best_sim:
                best_sim, best_response = sim, entry.response
        return best_response


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _expired(entry: _Entry, now: float) -> bool:
    return entry.expires_at is not None and now >= entry.expires_at


def _serve(response: Response) -> Response:
    """A copy marked ``cached`` with zeroed usage - a hit spent no new tokens."""
    return replace(
        response, usage=Usage(), usage_records=[], classification_usage=None, cached=True
    )


async def _resolve(value: list[float] | Awaitable[list[float]]) -> list[float]:
    if isinstance(value, list):
        return value
    return await value


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
