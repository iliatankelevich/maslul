"""The router. In M0 it does explicit ``level=`` dispatch only — internal judgment
(deterministic bypass, the hard-signal detector, the classify strategies) lands in M4.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from maslul.config import RouterConfig
from maslul.errors import ConfigError
from maslul.providers.base import Provider
from maslul.types import Level, ModelSpec, Request, Response


class Router:
    """Routes a :class:`Request` to a provider/model and returns a normalized
    :class:`Response`.

    M0 supports only an explicit ``level=`` pin. The provider registry is injected
    (the ``FakeProvider`` in tests); auto-building real providers from the
    ``[maslul.providers.*]`` config lands in M1.
    """

    def __init__(
        self,
        config: RouterConfig | Mapping[str, Any],
        providers: Mapping[str, Provider],
    ) -> None:
        self._config = (
            config if isinstance(config, RouterConfig) else RouterConfig.from_dict(config)
        )
        self._providers: dict[str, Provider] = dict(providers)

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str], providers: Mapping[str, Provider]) -> Router:
        """Build a router from a TOML config file. ``providers`` is injected in M0;
        auto-building real provider instances from config lands in M1.
        """
        return cls(RouterConfig.from_toml(path), providers)

    @property
    def config(self) -> RouterConfig:
        return self._config

    async def complete(self, req: Request, *, level: Level | None = None) -> Response:
        """Run a completion.

        M0 supports only an explicit ``level=`` pin: it dispatches to that tier's model.
        Internal judgment (omitting ``level``) is not implemented until M4.
        """
        if level is None:
            raise NotImplementedError(
                "internal routing (no level=) lands in M4 — pin level= for now"
            )
        spec = self._spec_for_level(level)
        provider = self._provider_for(spec)
        resp = await provider.complete(spec, req)
        resp.level_used = level
        return resp

    def _spec_for_level(self, level: Level) -> ModelSpec:
        try:
            return self._config.tiers[level]
        except KeyError as e:
            raise ConfigError(f"no tier configured for level {level.name}") from e

    def _provider_for(self, spec: ModelSpec) -> Provider:
        try:
            return self._providers[spec.provider]
        except KeyError as e:
            raise ConfigError(
                f"no provider registered for {spec.provider!r} "
                f"(registered: {sorted(self._providers)})"
            ) from e
