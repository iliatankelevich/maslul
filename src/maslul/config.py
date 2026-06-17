"""Router configuration — parsed from a plain dict (OSS users) or a TOML file
(``maslul.toml``). The shape mirrors §5 of the implementation plan.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from maslul.cache import CacheConfig
from maslul.errors import ConfigError
from maslul.types import Level, ModelSpec, Strategy


@dataclass
class RouterConfig:
    """Resolved router configuration: the tier→model map plus routing knobs.

    ``providers`` holds the raw ``[maslul.providers.*]`` sub-tables verbatim; they are
    consumed in M1 to build real provider instances (M0 injects providers directly).
    """

    strategy: Strategy
    default_level: Level
    tiers: dict[Level, ModelSpec]
    classifier: ModelSpec | None = None
    min_tokens_to_classify: int = 40
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Resilience (M5): per-call timeout, retry+backoff on transient errors, tier fallback.
    request_timeout: float | None = None
    max_retries: int = 2
    retry_base_delay: float = 0.5
    retry_max_delay: float = 8.0
    fallback: bool = True
    # Response cache (M8): off by default. Semantic mode also needs Router(embed=...).
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RouterConfig:
        """Build from a config mapping. Tolerates either the ``[maslul]`` wrapper or a
        bare inner dict.
        """
        root = data.get("maslul", data)

        raw_strategy = root.get("strategy", Strategy.ROUTE_DEFAULT.value)
        try:
            strategy = Strategy(raw_strategy)
        except ValueError as e:
            raise ConfigError(f"unknown strategy {raw_strategy!r}") from e

        default_level = _parse_level(root.get("default_level", "hard"))
        min_tokens = int(root.get("min_tokens_to_classify", 40))

        tiers = {
            _parse_level(name): _model_spec(entry) for name, entry in root.get("tiers", {}).items()
        }

        classifier_raw = root.get("classifier")
        classifier = _model_spec(classifier_raw) if classifier_raw else None

        timeout = root.get("request_timeout")
        return cls(
            strategy=strategy,
            default_level=default_level,
            tiers=tiers,
            classifier=classifier,
            min_tokens_to_classify=min_tokens,
            providers=dict(root.get("providers", {})),
            request_timeout=float(timeout) if timeout is not None else None,
            max_retries=int(root.get("max_retries", 2)),
            retry_base_delay=float(root.get("retry_base_delay", 0.5)),
            retry_max_delay=float(root.get("retry_max_delay", 8.0)),
            fallback=bool(root.get("fallback", True)),
            cache=_cache_config(root.get("cache", {})),
        )

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str]) -> RouterConfig:
        """Load and parse a TOML config file (stdlib ``tomllib``)."""
        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))


def _cache_config(entry: Mapping[str, Any]) -> CacheConfig:
    """Parse ``[maslul.cache]`` (mode off/exact/semantic + knobs)."""
    ttl = entry.get("ttl_seconds")
    return CacheConfig(
        mode=str(entry.get("mode", "off")),
        max_entries=int(entry.get("max_entries", 512)),
        ttl_seconds=float(ttl) if ttl is not None else None,
        similarity_threshold=float(entry.get("similarity_threshold", 0.95)),
    )


def _parse_level(value: str | Level) -> Level:
    if isinstance(value, Level):
        return value
    try:
        return Level[str(value).upper()]
    except KeyError as e:
        raise ConfigError(f"unknown level {value!r} — expected simple/medium/hard") from e


def _model_spec(entry: Mapping[str, Any]) -> ModelSpec:
    """A tier/classifier entry: either split ``provider``+``model`` keys, or a single
    ``model = "provider:model"`` shorthand (identical to Kippy's ``[models]`` format).
    """
    if "provider" in entry:
        return ModelSpec(
            provider=entry["provider"],
            model=entry["model"],
            max_tokens=entry.get("max_tokens"),
            options=dict(entry.get("options", {})),
        )
    if "model" in entry:
        return ModelSpec.parse(entry["model"])
    raise ConfigError(
        f"tier/classifier needs 'provider'+'model' or a 'provider:model' shorthand, "
        f"got {dict(entry)!r}"
    )
