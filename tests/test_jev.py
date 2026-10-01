"""Jev hooks — no network: every call goes through ``httpx.MockTransport``.

Covers the contract that matters most for an optional, injected hook: it must *never* break
routing. A confident answer resolves normally; anything else (low confidence, malformed JSON,
429/500, a timeout) must come back as ``None`` for the classifier and ``on_error``'s policy for
the verifier, never an exception.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from maslul.jev import JevClassifier, JevClient, JevVerifier, hooks_from_config
from maslul.types import Level, Message, Request, Response, Usage


def _req(*messages: str) -> Request:
    return Request(messages=[Message(role="user", content=m) for m in messages])


def _resp(text: str) -> Response:
    return Response(text=text, level_used=None, provider="fake", model="fake", usage=Usage())


def _client(handler: Any, **kwargs: Any) -> JevClient:
    return JevClient(api_key="test-key", transport=httpx.MockTransport(handler), **kwargs)


def _choice_handler(choice: str, confidence: float, probabilities: dict[str, float] | None = None):
    default_probs = {"simple": 0.0, "medium": 0.0, "hard": 0.0}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "level": {
                        "type": "choice",
                        "choice": choice,
                        "probabilities": probabilities or default_probs,
                        "confidence": confidence,
                    }
                },
                "usage": {"input_tokens": 42, "output_tokens": 5},
            },
        )

    return handler


def _noul_handler(noul: float):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {"ok": {"type": "noul", "noul": noul}},
                "usage": {"input_tokens": 30, "output_tokens": 4},
            },
        )

    return handler


def _status_handler(status: int):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "nope"})

    return handler


def _timeout_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.TimeoutException("simulated timeout", request=request)


async def test_confident_choice_returns_level() -> None:
    client = _client(_choice_handler("hard", 0.91, {"simple": 0.03, "medium": 0.06, "hard": 0.91}))
    classifier = JevClassifier(client, min_confidence=0.7)
    assert await classifier(_req("plan my entire tax strategy for next year")) is Level.HARD


async def test_low_confidence_defers_to_strategy() -> None:
    client = _client(_choice_handler("hard", 0.4))
    classifier = JevClassifier(client, min_confidence=0.7)
    assert await classifier(_req("hmm")) is None


async def test_decide_reports_raw_pick_below_threshold() -> None:
    """Shadow mode needs the raw pick even when it wouldn't clear the gate."""
    client = _client(_choice_handler("medium", 0.4, {"simple": 0.3, "medium": 0.4, "hard": 0.3}))
    classifier = JevClassifier(client, min_confidence=0.7)
    decision = await classifier.decide(_req("hmm"))
    assert decision.level is Level.MEDIUM
    assert decision.confidence == 0.4
    assert decision.probabilities[Level.MEDIUM] == 0.4
    assert decision.model == "jev-1.13.0"
    assert decision.usage == Usage(input_tokens=42, output_tokens=5)


@pytest.mark.parametrize("status", [429, 500, 529])
async def test_http_errors_return_none(status: int) -> None:
    client = _client(_status_handler(status))
    classifier = JevClassifier(client)
    assert await classifier(_req("anything")) is None


async def test_timeout_returns_none() -> None:
    client = _client(_timeout_handler)
    classifier = JevClassifier(client)
    assert await classifier(_req("anything")) is None


@pytest.mark.parametrize(
    "answers",
    [
        {
            "level": {
                "type": "choice",
                "choice": "unknown-option",
                "probabilities": {},
                "confidence": 0.9,
            }
        },
        {"level": {"type": "noul", "noul": 0.5}},  # wrong primitive entirely
        {"nope": {}},  # missing the expected question key
    ],
)
async def test_malformed_answer_returns_none(answers: dict[str, Any]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers, "usage": {}})

    classifier = JevClassifier(_client(handler))
    assert await classifier(_req("anything")) is None


async def test_on_decision_fires_even_on_failure() -> None:
    seen = []
    classifier = JevClassifier(_client(_status_handler(500)), on_decision=seen.append)
    await classifier(_req("x"))
    assert len(seen) == 1
    assert seen[0].level is None


async def test_state_window_and_char_cap_keep_the_latest_message() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "level": {
                        "type": "choice",
                        "choice": "simple",
                        "probabilities": {"simple": 1.0, "medium": 0.0, "hard": 0.0},
                        "confidence": 0.99,
                    }
                },
                "usage": {},
            },
        )

    client = _client(handler)
    classifier = JevClassifier(client, history_window=2, state_char_cap=50)
    long_tail = "x" * 100
    await classifier(_req("turn one", "turn two", f"latest: {long_tail}"))
    state = captured["body"]["state"]
    assert isinstance(state, str)
    assert len(state) <= 50
    assert "turn one" not in state  # outside the history window
    assert state.endswith(long_tail[-10:])  # tail-truncated: the newest content survives the cap


async def test_hebrew_state_passes_through_unmangled() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "level": {
                        "type": "choice",
                        "choice": "simple",
                        "probabilities": {"simple": 1.0, "medium": 0.0, "hard": 0.0},
                        "confidence": 0.99,
                    }
                },
                "usage": {},
            },
        )

    client = _client(handler)
    classifier = JevClassifier(client)
    hebrew = "תודה קיפי! מה השעה עכשיו?"
    await classifier(_req(hebrew))
    assert hebrew in captured["body"]["state"]


async def test_confident_yes_accepts() -> None:
    verifier = JevVerifier(_client(_noul_handler(0.95)), min_yes=0.7)
    assert await verifier(_req("question"), _resp("a good answer")) is True


async def test_low_yes_rejects() -> None:
    verifier = JevVerifier(_client(_noul_handler(0.2)), min_yes=0.7)
    assert await verifier(_req("question"), _resp("a bad answer")) is False


async def test_verifier_error_defaults_to_accept() -> None:
    verifier = JevVerifier(_client(_status_handler(500)))
    assert await verifier(_req("q"), _resp("a")) is True


async def test_verifier_on_error_reject() -> None:
    verifier = JevVerifier(_client(_status_handler(500)), on_error="reject")
    assert await verifier(_req("q"), _resp("a")) is False


async def test_verifier_timeout_defaults_to_accept() -> None:
    verifier = JevVerifier(_client(_timeout_handler))
    assert await verifier(_req("q"), _resp("a")) is True


def test_hooks_from_config_empty_without_jev_table(monkeypatch: pytest.MonkeyPatch) -> None:
    hooks = hooks_from_config({"maslul": {}})
    assert hooks.classifier is None
    assert hooks.verifier is None


def test_hooks_from_config_builds_requested_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    hooks = hooks_from_config(
        {
            "maslul": {
                "jev": {
                    "classifier": True,
                    "verifier": True,
                    "min_confidence": 0.8,
                    "min_yes": 0.6,
                    "on_error": "reject",
                }
            }
        }
    )
    assert isinstance(hooks.classifier, JevClassifier)
    assert isinstance(hooks.verifier, JevVerifier)


def test_hooks_from_config_classifier_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    hooks = hooks_from_config({"maslul": {"jev": {"classifier": True}}})
    assert hooks.classifier is not None
    assert hooks.verifier is None


def test_hooks_from_config_without_a_requested_hook_needs_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    hooks = hooks_from_config({"maslul": {"jev": {"timeout": 2.0}}})
    assert hooks.classifier is None and hooks.verifier is None and hooks.client is None


async def test_hooks_from_config_client_can_be_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    hooks = hooks_from_config({"maslul": {"jev": {"classifier": True, "verifier": True}}})
    assert hooks.client is not None
    await hooks.aclose()
    assert hooks.client._client.is_closed
