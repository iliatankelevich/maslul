from __future__ import annotations

import pytest

from maslul import ConfigError, Level, ModelSpec, Strategy


def test_model_spec_parse_valid() -> None:
    spec = ModelSpec.parse("anthropic:claude-sonnet-4-6")
    assert spec.provider == "anthropic"
    assert spec.model == "claude-sonnet-4-6"


@pytest.mark.parametrize("bad", ["claude-sonnet", "cohere:command", "anthropic:", ":model", ""])
def test_model_spec_parse_rejects_bad(bad: str) -> None:
    with pytest.raises(ConfigError):
        ModelSpec.parse(bad)


def test_level_is_ordered_for_escalation() -> None:
    assert Level.SIMPLE < Level.MEDIUM < Level.HARD


def test_strategy_values_match_config_strings() -> None:
    assert Strategy("route_default") is Strategy.ROUTE_DEFAULT
    assert Strategy.CLASSIFY_AND_ANSWER.value == "classify_and_answer"
