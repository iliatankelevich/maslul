from __future__ import annotations

from typing import Any

import pytest

from maslul import ConfigError, Level, Message, Request, Router, Strategy
from tests.fakes import FakeProvider


def _config() -> dict[str, Any]:
    return {
        "maslul": {
            "strategy": "route_default",
            "default_level": "hard",
            "tiers": {
                "simple": {"provider": "fake", "model": "small"},
                "medium": {"provider": "fake", "model": "mid"},
                "hard": {"provider": "fake", "model": "big"},
            },
        }
    }


def _req() -> Request:
    return Request(messages=[Message(role="user", content="hello")])


async def test_level_dispatch_returns_normalized_response() -> None:
    fake = FakeProvider("fake", text="hi")
    router = Router(_config(), providers={"fake": fake})

    resp = await router.complete(_req(), level=Level.SIMPLE)

    assert resp.text == "hi"
    assert resp.level_used is Level.SIMPLE
    assert resp.provider == "fake"
    assert resp.model == "small"  # the SIMPLE tier
    assert len(fake.calls) == 1
    assert fake.calls[0][0].model == "small"


async def test_complete_without_level_is_not_implemented_yet() -> None:
    router = Router(_config(), providers={"fake": FakeProvider("fake")})
    with pytest.raises(NotImplementedError):
        await router.complete(_req())


async def test_missing_provider_raises_config_error() -> None:
    router = Router(_config(), providers={})
    with pytest.raises(ConfigError):
        await router.complete(_req(), level=Level.HARD)


def test_from_dict_parses_tiers_and_routing_knobs() -> None:
    cfg = Router(_config(), providers={"fake": FakeProvider("fake")}).config
    assert cfg.strategy is Strategy.ROUTE_DEFAULT
    assert cfg.default_level is Level.HARD
    assert cfg.tiers[Level.SIMPLE].model == "small"
    assert cfg.tiers[Level.HARD].model == "big"
