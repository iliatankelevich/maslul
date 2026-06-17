"""Provider drivers and the factory that builds them from ``[maslul.providers.*]`` config.

This package is imported as part of ``import maslul`` (the router depends on it), so it must
stay SDK-free. That is why ``build_provider`` defers importing each concrete provider module to
the moment it's actually built — each provider module imports its SDK at top level, so importing
``maslul.providers.anthropic`` requires the ``anthropic`` extra, importing
``maslul.providers.gemini`` requires ``gemini``, etc. Deferring keeps ``import maslul`` working
with only the extras you installed (locked in by ``tests/test_import_isolation.py``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from maslul.errors import ConfigError
from maslul.providers.base import Provider
from maslul.types import KNOWN_PROVIDERS

__all__ = ["Provider", "build_provider"]


def build_provider(name: str, config: Mapping[str, Any]) -> Provider:
    """Construct the provider named ``name`` from its config sub-table.

    Secrets are referenced by env-var name (``api_key_env``), never inlined. Used by
    :meth:`maslul.Router.from_toml` to auto-wire providers when none are injected.

    The per-branch imports are deliberately deferred (see this module's docstring): they pull in
    the SDK only for the provider you actually build, keeping ``import maslul`` SDK-free.
    """
    if name == "anthropic":
        from maslul.providers.anthropic import AnthropicProvider

        key = _env(config.get("api_key_env")) or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigError("anthropic not configured (set api_key_env or ANTHROPIC_API_KEY)")
        return AnthropicProvider(api_key=key)
    if name == "gemini":
        from maslul.providers.gemini import GeminiProvider

        project = config.get("vertex_project")
        key = _env(config.get("api_key_env")) or os.environ.get("GOOGLE_API_KEY")
        if not project and not key:
            raise ConfigError("gemini not configured (set vertex_project or an api key)")
        return GeminiProvider(
            vertex_project=project,
            vertex_location=config.get("vertex_location", "global"),
            api_key=key,
        )
    if name == "grok":
        from maslul.providers.grok import GrokProvider

        key = _env(config.get("api_key_env")) or os.environ.get("XAI_API_KEY")
        if not key:
            raise ConfigError("grok not configured (set api_key_env or XAI_API_KEY)")
        return GrokProvider(api_key=key)
    raise ConfigError(f"unknown provider {name!r} — expected one of {sorted(KNOWN_PROVIDERS)}")


def _env(var_name: str | None) -> str | None:
    return os.environ.get(var_name) if var_name else None
