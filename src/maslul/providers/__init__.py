"""Provider drivers and the factory that builds them from ``[maslul.providers.*]`` config.

``build_provider`` imports each backend's SDK lazily (inside the matching branch), so importing
this package — or ``maslul`` itself — never pulls in a provider SDK you didn't install.
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
    """
    if name == "anthropic":
        from maslul.providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key=_env(config.get("api_key_env")))
    if name == "gemini":
        from maslul.providers.gemini import GeminiProvider

        return GeminiProvider(
            vertex_project=config.get("vertex_project"),
            vertex_location=config.get("vertex_location", "global"),
            api_key=_env(config.get("api_key_env")),
        )
    if name == "grok":
        from maslul.providers.grok import GrokProvider

        return GrokProvider(api_key=_env(config.get("api_key_env")))
    raise ConfigError(f"unknown provider {name!r} — expected one of {sorted(KNOWN_PROVIDERS)}")


def _env(var_name: str | None) -> str | None:
    return os.environ.get(var_name) if var_name else None
