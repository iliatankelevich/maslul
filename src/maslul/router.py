"""The router. In M0 it does explicit ``level=`` dispatch only — internal judgment
(deterministic bypass, the hard-signal detector, the classify strategies) lands in M4.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from maslul.config import RouterConfig
from maslul.errors import ConfigError, ProviderError
from maslul.providers import Provider, build_provider
from maslul.types import Level, Message, ModelSpec, Request, Response, ToolCall, Usage

# Guard against a runaway tool loop (mirrors Kippy's llm.py).
_MAX_TOOL_ITERATIONS = 8


class Router:
    """Routes a :class:`Request` to a provider/model and returns a normalized
    :class:`Response`.

    M0 supports only an explicit ``level=`` pin. ``providers`` may be injected (the
    ``FakeProvider`` in tests); when omitted, the providers named by the configured tiers
    and classifier are auto-built from the ``[maslul.providers.*]`` config.
    """

    def __init__(
        self,
        config: RouterConfig | Mapping[str, Any],
        providers: Mapping[str, Provider] | None = None,
    ) -> None:
        self._config = (
            config if isinstance(config, RouterConfig) else RouterConfig.from_dict(config)
        )
        if providers is None:
            providers = {
                name: build_provider(name, self._config.providers.get(name, {}))
                for name in self._provider_names()
            }
        self._providers: dict[str, Provider] = dict(providers)

    @classmethod
    def from_toml(
        cls,
        path: str | os.PathLike[str],
        providers: Mapping[str, Provider] | None = None,
    ) -> Router:
        """Build a router from a TOML config file. Omit ``providers`` to auto-build them
        from the ``[maslul.providers.*]`` config; inject them for tests or custom wiring.
        """
        return cls(RouterConfig.from_toml(path), providers)

    def _provider_names(self) -> set[str]:
        """Provider names referenced by the configured tiers and classifier."""
        names = {spec.provider for spec in self._config.tiers.values()}
        if self._config.classifier is not None:
            names.add(self._config.classifier.provider)
        return names

    @property
    def config(self) -> RouterConfig:
        return self._config

    async def complete(self, req: Request, *, level: Level | None = None) -> Response:
        """Run a completion at the pinned ``level``.

        With ``req.tools`` + ``req.tool_executor``, the router drives the provider-agnostic
        tool-use loop: call the model, run any requested tools, feed results back, repeat
        until the model stops calling tools (or the iteration guard trips). Internal judgment
        (omitting ``level``) is not implemented until M4.
        """
        if level is None:
            raise NotImplementedError(
                "internal routing (no level=) lands in M4 — pin level= for now"
            )
        spec = self._spec_for_level(level)
        provider = self._provider_for(spec)
        if not req.tools or req.tool_executor is None:
            resp = await provider.complete(spec, req)
            resp.level_used = level
            return resp
        return await self._run_tool_loop(spec, provider, req, level)

    async def _run_tool_loop(
        self, spec: ModelSpec, provider: Provider, req: Request, level: Level
    ) -> Response:
        assert req.tool_executor is not None
        messages = list(req.messages)
        total = Usage()
        executed: list[ToolCall] = []
        for _ in range(_MAX_TOOL_ITERATIONS):
            resp = await provider.complete(spec, replace(req, messages=messages))
            _accumulate(total, resp.usage)
            if not resp.tool_calls:
                resp.usage = total
                resp.tool_calls = executed  # surface the full tool I/O of the turn
                resp.level_used = level
                return resp
            messages.append(
                Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls)
            )
            for call in resp.tool_calls:
                executed.append(call)
                result = await req.tool_executor(call)
                messages.append(
                    Message(role="tool", content=result, tool_call_id=call.id, name=call.name)
                )
        raise ProviderError(f"tool loop exceeded {_MAX_TOOL_ITERATIONS} iterations")

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


def _accumulate(total: Usage, u: Usage) -> None:
    total.input_tokens += u.input_tokens
    total.output_tokens += u.output_tokens
    total.cache_read_input_tokens += u.cache_read_input_tokens
    total.cache_creation_input_tokens += u.cache_creation_input_tokens
