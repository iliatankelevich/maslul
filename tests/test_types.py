from __future__ import annotations

import pytest

from maslul import ConfigError, Level, ModelSpec, RouterConfig, Strategy


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


@pytest.mark.parametrize(
    "entry",
    [
        {"model": "anthropic:claude-haiku-4-5"},
        {"provider": "anthropic", "model": "claude-haiku-4-5"},
    ],
)
def test_tier_max_tokens_and_options_apply_in_both_forms(entry: dict[str, str]) -> None:
    tier = {**entry, "max_tokens": 2048, "options": {"temperature": 0.2}}
    spec = RouterConfig.from_dict({"tiers": {"hard": tier}}).tiers[Level.HARD]
    assert (spec.provider, spec.model) == ("anthropic", "claude-haiku-4-5")
    assert spec.max_tokens == 2048
    assert spec.options == {"temperature": 0.2}


# --- deprecated "grok" provider name (renamed "xai" in 0.6.0) ----------------------------


def test_grok_spec_is_renamed_to_xai_with_a_warning() -> None:
    with pytest.warns(DeprecationWarning, match="'grok' is deprecated"):
        spec = ModelSpec.parse("grok:grok-4.3")
    assert (spec.provider, spec.model) == ("xai", "grok-4.3")
    with pytest.warns(DeprecationWarning):
        assert ModelSpec(provider="grok", model="grok-4.3").provider == "xai"


def test_grok_config_tier_and_provider_table_are_renamed() -> None:
    with pytest.warns(DeprecationWarning):
        cfg = RouterConfig.from_dict(
            {
                "tiers": {"hard": {"provider": "grok", "model": "grok-4.3"}},
                "providers": {"grok": {"api_key_env": "XAI_API_KEY"}},
            }
        )
    assert cfg.tiers[Level.HARD].provider == "xai"
    assert cfg.providers == {"xai": {"api_key_env": "XAI_API_KEY"}}


def test_xai_spec_does_not_warn(recwarn: pytest.WarningsRecorder) -> None:
    assert ModelSpec.parse("xai:grok-4.3").provider == "xai"
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
