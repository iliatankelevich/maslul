"""Prompt caching (``ContextCache``) — the disjoint-``Usage`` convention, stable-first layout, and
Anthropic's ``cache_control`` breakpoint placement.

These assert *placement*, not just presence: which block carries the marker is the whole feature.
The most important test in the file is the no-regression one — with ``context_cache=None`` the
rendered request must be byte-for-byte what it was before prompt caching existed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from maslul.providers._common import split_input
from maslul.providers.anthropic import AnthropicProvider
from maslul.providers.gemini import GeminiProvider
from maslul.providers.grok import GrokProvider
from maslul.providers.openai import OpenAIProvider
from maslul.types import ContextCache, MediaPart, Message, ModelSpec, Request
from tests.test_providers_unit import _FakeAnthropic, _FakeGemini, _FakeGrok, _FakeOpenAI

_ANTHROPIC = ModelSpec(provider="anthropic", model="claude-sonnet-4-6")
_PDF = MediaPart(mime_type="application/pdf", data=b"%PDF-fake")


def _anthropic_resp() -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        stop_reason="end_turn",
    )


def _oai_ok() -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None, annotations=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None),
    )


def _gemini_ok() -> Any:
    return SimpleNamespace(
        text="ok",
        usage_metadata=SimpleNamespace(
            prompt_token_count=1, candidates_token_count=1, cached_content_token_count=0
        ),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
    )


def _grok_ok() -> Any:
    return SimpleNamespace(
        content="ok",
        tool_calls=[],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, cached_prompt_text_tokens=0),
        finish_reason="stop",
    )


def _blocks(sent: dict[str, Any]) -> list[dict[str, Any]]:
    """Every content block across the rendered messages, flattened."""
    out: list[dict[str, Any]] = []
    for m in sent["messages"]:
        content = m["content"]
        out.extend(content if isinstance(content, list) else [])
    return out


def _marked(sent: dict[str, Any]) -> list[dict[str, Any]]:
    """Every block carrying a cache_control marker — system blocks included."""
    system = sent.get("system")
    blocks = list(system) if isinstance(system, list) else []
    return [b for b in blocks + _blocks(sent) if "cache_control" in b]


# --- the disjoint Usage convention (phase 1) ---------------------------------------------
# Each provider's raw usage shape goes in; input_tokens + cache_read must equal the provider's
# own prompt total. Anthropic already reports them disjoint; the other three do not.


def test_split_input_is_disjoint_and_clamped() -> None:
    assert split_input(100, 30) == (70, 30)  # cached-inclusive total → disjoint
    assert split_input(100, 0) == (100, 0)
    assert split_input(10, 40) == (0, 40)  # never negative, however the provider misreports


async def test_anthropic_usage_is_left_alone() -> None:
    """Anthropic already excludes cached tokens from input_tokens — don't double-subtract."""
    resp = _anthropic_resp()
    resp.usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=90,
        cache_creation_input_tokens=0,
    )
    out = await AnthropicProvider(client=_FakeAnthropic(resp)).complete(
        _ANTHROPIC, Request(messages=[Message(role="user", content="hi")])
    )
    assert (out.usage.input_tokens, out.usage.cache_read_input_tokens) == (10, 90)


async def test_openai_usage_subtracts_cached_from_the_prompt_total() -> None:
    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None, annotations=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,  # includes the 90 cached
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=90),
        ),
    )
    out = await OpenAIProvider(client=_FakeOpenAI(resp)).complete(
        ModelSpec(provider="openai", model="gpt-4o"),
        Request(messages=[Message(role="user", content="hi")]),
    )
    assert (out.usage.input_tokens, out.usage.cache_read_input_tokens) == (10, 90)
    assert out.usage.input_tokens + out.usage.cache_read_input_tokens == 100  # no double count


async def test_grok_usage_subtracts_cached_from_the_prompt_total() -> None:
    resp = SimpleNamespace(
        content="ok",
        tool_calls=[],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5, cached_prompt_text_tokens=90),
        finish_reason="stop",
    )
    out = await GrokProvider(client=_FakeGrok(resp)).complete(
        ModelSpec(provider="grok", model="grok-4.3"),
        Request(messages=[Message(role="user", content="hi")]),
    )
    assert (out.usage.input_tokens, out.usage.cache_read_input_tokens) == (10, 90)


async def test_gemini_usage_subtracts_cached_from_the_prompt_total() -> None:
    resp = SimpleNamespace(
        text="ok",
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,  # includes the 90 cached
            candidates_token_count=5,
            cached_content_token_count=90,
        ),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
    )
    out = await GeminiProvider(client=_FakeGemini(resp)).complete(
        ModelSpec(provider="gemini", model="gemini-2.5-flash"),
        Request(messages=[Message(role="user", content="hi")]),
    )
    assert (out.usage.input_tokens, out.usage.cache_read_input_tokens) == (10, 90)


# --- stable-first layout (phase 2) — the free win on every provider ------------------------
# Media must move from the LAST user message (the most volatile slot) to the FIRST when
# ContextCache(media=True), or the prefix shifts on every new turn and nothing ever caches.

_CHAT = [
    Message(role="user", content="here is the doc"),
    Message(role="assistant", content="got it"),
    Message(role="user", content="what does clause 4 say?"),
]


async def test_anthropic_media_moves_to_the_first_user_message_when_cached() -> None:
    fake = _FakeAnthropic(_anthropic_resp())
    await AnthropicProvider(client=fake).complete(
        _ANTHROPIC,
        Request(messages=_CHAT, media=[_PDF], context_cache=ContextCache(media=True)),
    )
    sent = fake.calls[0]["messages"]
    assert any(b.get("type") == "document" for b in sent[0]["content"])  # first user turn
    # ...and the volatile last turn is untouched — still a plain string, no media, no rewrite.
    assert sent[2]["content"] == "what does clause 4 say?"


async def test_anthropic_media_stays_last_without_a_context_cache() -> None:
    """The no-regression guarantee: absent a ContextCache, nothing about the layout changes."""
    fake = _FakeAnthropic(_anthropic_resp())
    await AnthropicProvider(client=fake).complete(_ANTHROPIC, Request(messages=_CHAT, media=[_PDF]))
    sent = fake.calls[0]["messages"]
    assert sent[0]["content"] == "here is the doc"  # untouched plain string
    assert any(b.get("type") == "document" for b in sent[2]["content"])  # still the last turn


async def test_no_context_cache_leaves_plain_string_content_alone() -> None:
    """Plain-string content must NOT be rewritten into a text-block list just because the history
    breakpoint *could* have gone there. Marking rewrites the block; not asking must not."""
    fake = _FakeAnthropic(_anthropic_resp())
    await AnthropicProvider(client=fake).complete(_ANTHROPIC, Request(messages=_CHAT))
    assert fake.calls[0]["messages"] == [
        {"role": "user", "content": "here is the doc"},
        {"role": "assistant", "content": "got it"},
        {"role": "user", "content": "what does clause 4 say?"},
    ]


async def test_openai_media_moves_to_the_first_user_message_when_cached() -> None:
    fake = _FakeOpenAI(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None, annotations=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None),
        )
    )
    await OpenAIProvider(client=fake).complete(
        ModelSpec(provider="openai", model="gpt-4o"),
        Request(
            messages=_CHAT,
            media=[MediaPart(mime_type="image/png", data=b"\x89PNG")],
            context_cache=ContextCache(media=True),
        ),
    )
    sent = fake.calls[0]["messages"]
    assert any(p.get("type") == "image_url" for p in sent[0]["content"])
    assert sent[2]["content"] == "what does clause 4 say?"  # volatile turn stays plain


async def test_gemini_media_moves_to_the_first_user_message_when_cached() -> None:
    fake = _FakeGemini(
        SimpleNamespace(
            text="ok",
            usage_metadata=SimpleNamespace(
                prompt_token_count=1, candidates_token_count=1, cached_content_token_count=0
            ),
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        )
    )
    await GeminiProvider(client=fake).complete(
        ModelSpec(provider="gemini", model="gemini-2.5-flash"),
        Request(messages=_CHAT, media=[_PDF], context_cache=ContextCache(media=True)),
    )
    contents = fake.calls[0]["contents"]
    assert any(getattr(p, "inline_data", None) is not None for p in contents[0].parts)
    assert not any(getattr(p, "inline_data", None) is not None for p in contents[2].parts)


async def test_grok_media_moves_to_the_first_user_message_when_cached() -> None:
    fake = _FakeGrok(
        SimpleNamespace(
            content="ok",
            tool_calls=[],
            usage=SimpleNamespace(
                prompt_tokens=1, completion_tokens=1, cached_prompt_text_tokens=0
            ),
            finish_reason="stop",
        )
    )
    await GrokProvider(client=fake).complete(
        ModelSpec(provider="grok", model="grok-4.3"),
        Request(
            messages=_CHAT,
            media=[MediaPart(mime_type="image/png", data=b"\x89PNG")],
            context_cache=ContextCache(media=True),
        ),
    )
    sent = fake.calls[0]["messages"]
    assert any(c.HasField("image_url") for c in sent[0].content)
    assert not any(c.HasField("image_url") for c in sent[2].content)


# --- OpenAI affinity key (phase 2) ---------------------------------------------------------


async def test_openai_media_stays_last_without_a_context_cache() -> None:
    """No-regression, OpenAI: media must NOT move unless a ContextCache asks for it."""
    fake = _FakeOpenAI(_oai_ok())
    await OpenAIProvider(client=fake).complete(
        ModelSpec(provider="openai", model="gpt-4o"),
        Request(messages=_CHAT, media=[MediaPart(mime_type="image/png", data=b"\x89PNG")]),
    )
    sent = fake.calls[0]["messages"]
    assert sent[0]["content"] == "here is the doc"  # first turn untouched
    assert any(p.get("type") == "image_url" for p in sent[2]["content"])  # still the last turn


async def test_gemini_media_stays_last_without_a_context_cache() -> None:
    """No-regression, Gemini."""
    fake = _FakeGemini(_gemini_ok())
    await GeminiProvider(client=fake).complete(
        ModelSpec(provider="gemini", model="gemini-2.5-flash"),
        Request(messages=_CHAT, media=[_PDF]),
    )
    contents = fake.calls[0]["contents"]
    assert not any(getattr(p, "inline_data", None) is not None for p in contents[0].parts)
    assert any(getattr(p, "inline_data", None) is not None for p in contents[2].parts)


async def test_grok_media_stays_last_without_a_context_cache() -> None:
    """No-regression, Grok."""
    fake = _FakeGrok(_grok_ok())
    await GrokProvider(client=fake).complete(
        ModelSpec(provider="grok", model="grok-4.3"),
        Request(messages=_CHAT, media=[MediaPart(mime_type="image/png", data=b"\x89PNG")]),
    )
    sent = fake.calls[0]["messages"]
    assert not any(c.HasField("image_url") for c in sent[0].content)
    assert any(c.HasField("image_url") for c in sent[2].content)


async def test_openai_short_ttl_does_not_request_extended_retention() -> None:
    """Only ttl_seconds >= 3600 opts into 24h retention — a 5-minute ttl must not."""
    fake = _FakeOpenAI(_oai_ok())
    await OpenAIProvider(client=fake).complete(
        ModelSpec(provider="openai", model="gpt-4o"),
        Request(
            messages=[Message(role="user", content="hi")],
            context_cache=ContextCache(ttl_seconds=300),
        ),
    )
    assert "prompt_cache_retention" not in fake.calls[0]


async def test_openai_forwards_the_affinity_key_and_extended_retention() -> None:
    fake = _FakeOpenAI(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None, annotations=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None),
        )
    )
    spec = ModelSpec(provider="openai", model="gpt-4o")
    provider = OpenAIProvider(client=fake)
    await provider.complete(
        spec,
        Request(
            messages=[Message(role="user", content="hi")],
            context_cache=ContextCache(key="doc-42", ttl_seconds=3600),
        ),
    )
    assert fake.calls[0]["prompt_cache_key"] == "doc-42"
    assert fake.calls[0]["prompt_cache_retention"] == "24h"

    await provider.complete(spec, Request(messages=[Message(role="user", content="hi")]))
    assert "prompt_cache_key" not in fake.calls[1]  # absent without a ContextCache
    assert "prompt_cache_retention" not in fake.calls[1]


# --- Anthropic cache_control placement (phase 3) --------------------------------------------


async def test_anthropic_places_no_markers_without_a_context_cache() -> None:
    """The no-regression guarantee: system stays a plain string, no block carries a marker."""
    fake = _FakeAnthropic(_anthropic_resp())
    await AnthropicProvider(client=fake).complete(
        _ANTHROPIC, Request(messages=_CHAT, media=[_PDF], system=["persona"])
    )
    sent = fake.calls[0]
    assert sent["system"] == "persona"  # folded to a string, exactly as before
    assert _marked(sent) == []


async def test_anthropic_marks_the_last_system_block_only() -> None:
    """One breakpoint on the *last* system block — it caches tools + system together, since tools
    render first and a breakpoint caches everything before it."""
    fake = _FakeAnthropic(_anthropic_resp())
    await AnthropicProvider(client=fake).complete(
        _ANTHROPIC,
        Request(
            messages=[Message(role="user", content="hi")],
            system=["guidance", "persona"],
            context_cache=ContextCache(),  # system=True is the default
        ),
    )
    sent = fake.calls[0]
    assert [b["text"] for b in sent["system"]] == ["guidance", "persona"]
    assert "cache_control" not in sent["system"][0]
    assert sent["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert len(_marked(sent)) == 1


async def test_anthropic_marks_the_last_media_block() -> None:
    fake = _FakeAnthropic(_anthropic_resp())
    await AnthropicProvider(client=fake).complete(
        _ANTHROPIC,
        Request(
            messages=_CHAT,
            media=[_PDF, _PDF],
            system=["persona"],
            context_cache=ContextCache(media=True),
        ),
    )
    first_turn = fake.calls[0]["messages"][0]["content"]
    docs = [b for b in first_turn if b.get("type") == "document"]
    assert "cache_control" not in docs[0]  # only the LAST media block is a breakpoint
    assert docs[1]["cache_control"] == {"type": "ephemeral"}


async def test_anthropic_history_marks_the_last_content_block() -> None:
    fake = _FakeAnthropic(_anthropic_resp())
    await AnthropicProvider(client=fake).complete(
        _ANTHROPIC,
        Request(
            messages=_CHAT,
            system=["persona"],
            context_cache=ContextCache(system=False, history=True),
        ),
    )
    sent = fake.calls[0]
    last = sent["messages"][-1]["content"]
    assert last[-1]["text"] == "what does clause 4 say?"
    assert last[-1]["cache_control"] == {"type": "ephemeral"}
    assert len(_marked(sent)) == 1


async def test_anthropic_long_ttl_opts_into_the_one_hour_cache() -> None:
    fake = _FakeAnthropic(_anthropic_resp())
    await AnthropicProvider(client=fake).complete(
        _ANTHROPIC,
        Request(
            messages=[Message(role="user", content="hi")],
            system=["persona"],
            context_cache=ContextCache(ttl_seconds=3600),
        ),
    )
    assert fake.calls[0]["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_anthropic_never_exceeds_four_breakpoints() -> None:
    """All three markers at once on a single-turn document-QA request — the exact shape Kippy
    sends. Three distinct blocks get marked, which is inside the budget of four."""
    fake = _FakeAnthropic(_anthropic_resp())
    await AnthropicProvider(client=fake).complete(
        _ANTHROPIC,
        Request(
            messages=[Message(role="user", content="summarize")],
            media=[_PDF],
            system=["persona"],
            context_cache=ContextCache(system=True, media=True, history=True),
        ),
    )
    sent = fake.calls[0]
    marked = _marked(sent)
    assert len(marked) == 3 <= 4  # system + media + history, each a distinct block
    # The document leads the message and carries the breakpoint; the volatile text trails it and
    # is therefore *outside* the prefix that breakpoint caches. That ordering is the whole feature.
    content = sent["messages"][0]["content"]
    assert [b["type"] for b in content] == ["document", "text"]
    assert content[0]["cache_control"] == {"type": "ephemeral"}


async def test_anthropic_counts_caller_markers_against_the_budget() -> None:
    """A caller hand-building a cached system through provider_options (Kippy's escape hatch) still
    owns those breakpoints; ours must budget around them rather than 400 the request."""
    fake = _FakeAnthropic(_anthropic_resp())
    pinned = [
        {"type": "text", "text": f"block {i}", "cache_control": {"type": "ephemeral"}}
        for i in range(4)  # the caller already spent the entire budget
    ]
    await AnthropicProvider(client=fake).complete(
        _ANTHROPIC,
        Request(
            messages=_CHAT,
            media=[_PDF],
            system=["persona"],
            provider_options={"system": pinned},
            context_cache=ContextCache(system=True, media=True, history=True),
        ),
    )
    sent = fake.calls[0]
    assert sent["system"] is pinned  # provider_options still wins
    assert len(_marked(sent)) == 4  # no marker of ours was added on top
    assert not any("cache_control" in b for b in _blocks(sent))


async def test_anthropic_pinned_system_does_not_waste_a_breakpoint() -> None:
    """When provider_options overrides our system, our system block never reaches the API. Marking
    that orphan would silently burn one of the four breakpoints — and could cost us the media one,
    which is the whole point of the feature. Budget must go to media instead."""
    fake = _FakeAnthropic(_anthropic_resp())
    pinned = [{"type": "text", "text": "persona", "cache_control": {"type": "ephemeral"}}]
    await AnthropicProvider(client=fake).complete(
        _ANTHROPIC,
        Request(
            messages=_CHAT,
            media=[_PDF],
            system=["ours — overridden"],
            provider_options={"system": pinned},
            context_cache=ContextCache(system=True, media=True, history=True),
        ),
    )
    sent = fake.calls[0]
    docs = [b for b in _blocks(sent) if b.get("type") == "document"]
    assert docs[-1]["cache_control"] == {"type": "ephemeral"}  # media still got its breakpoint
    assert len(_marked(sent)) == 3  # caller's 1 + media + history — within budget, none wasted


# --- router interaction: the tool loop must not lose the cache ------------------------------


async def test_context_cache_survives_the_router_tool_loop() -> None:
    """``_run_tool_loop`` rebuilds the Request each iteration with ``dataclasses.replace``. If
    ``context_cache`` were dropped there, every tool-loop turn would re-send an unmarked prompt and
    re-bill the whole document at full price — silently, since nothing would error. This is exactly
    the flow that matters (a document-QA turn that calls a tool), so pin it."""
    from maslul import Level, Router
    from maslul.types import ToolCall, ToolDef

    seen: list[ContextCache | None] = []

    class _Recorder:
        name = "anthropic"

        async def complete(self, spec: ModelSpec, req: Request) -> Any:
            from maslul.types import Response, Usage

            seen.append(req.context_cache)
            calls = (
                [ToolCall(id="c1", name="lookup", input={})] if len(seen) == 1 else []
            )  # tool on turn 1, answer on turn 2
            return Response(
                text="" if calls else "done",
                level_used=None,
                provider="anthropic",
                model=spec.model,
                usage=Usage(),
                tool_calls=calls,
            )

        async def healthcheck(self, spec: ModelSpec) -> None: ...

    router = Router(
        config={
            "maslul": {
                "strategy": "route_default",
                "default_level": "hard",
                "tiers": {"hard": {"model": "anthropic:claude-sonnet-4-6"}},
            }
        },
        providers={"anthropic": _Recorder()},
    )
    cache = ContextCache(media=True)
    resp = await router.complete(
        Request(
            messages=[Message(role="user", content="summarize")],
            media=[_PDF],
            context_cache=cache,
            tools=[ToolDef(name="lookup", description="l", input_schema={"type": "object"})],
            tool_executor=lambda call: _result(),
        ),
        level=Level.HARD,
    )
    assert resp.text == "done"
    assert len(seen) == 2  # the loop really did run a second turn
    assert seen == [cache, cache], "context_cache was dropped inside the tool loop"


async def _result() -> str:
    return "42"


# --- block order INSIDE the media message — the flow the whole feature exists for -------------
# Kippy's document-QA turn is a SINGLE user message carrying both the question and the PDF. So
# "attach media to the first user message" is a no-op there (first == last), and what actually
# decides whether the cache can ever hit is the order of blocks *within* that message: a breakpoint
# on the document caches everything before it, so a question emitted first lands inside the prefix
# and every new question is a fresh, never-hit prefix.

_DOC_QA = [Message(role="user", content="What does clause 4 say?")]  # question + PDF, one message


async def test_anthropic_media_precedes_the_volatile_question_when_cached() -> None:
    fake = _FakeAnthropic(_anthropic_resp())
    await AnthropicProvider(client=fake).complete(
        _ANTHROPIC,
        Request(messages=_DOC_QA, media=[_PDF], context_cache=ContextCache(media=True)),
    )
    content = fake.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in content] == ["document", "text"], (
        "the question must come AFTER the document, or it sits inside the cached prefix and "
        "every new question misses the cache"
    )
    # ...and the breakpoint is on the document, so the cached prefix ends before the question.
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in content[1]


async def test_anthropic_text_precedes_media_without_a_context_cache() -> None:
    """No-regression: the historical order is text-then-media. Only opting in flips it."""
    fake = _FakeAnthropic(_anthropic_resp())
    await AnthropicProvider(client=fake).complete(
        _ANTHROPIC, Request(messages=_DOC_QA, media=[_PDF])
    )
    content = fake.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in content] == ["text", "document"]


async def test_openai_media_precedes_the_question_when_cached() -> None:
    fake = _FakeOpenAI(_oai_ok())
    img = MediaPart(mime_type="image/png", data=b"\x89PNG")
    provider = OpenAIProvider(client=fake)
    spec = ModelSpec(provider="openai", model="gpt-4o")
    await provider.complete(
        spec, Request(messages=_DOC_QA, media=[img], context_cache=ContextCache(media=True))
    )
    assert [p["type"] for p in fake.calls[0]["messages"][0]["content"]] == ["image_url", "text"]

    await provider.complete(spec, Request(messages=_DOC_QA, media=[img]))  # no cache → unchanged
    assert [p["type"] for p in fake.calls[1]["messages"][0]["content"]] == ["text", "image_url"]


async def test_gemini_media_precedes_the_question_when_cached() -> None:
    fake = _FakeGemini(_gemini_ok())
    provider = GeminiProvider(client=fake)
    spec = ModelSpec(provider="gemini", model="gemini-2.5-flash")
    await provider.complete(
        spec, Request(messages=_DOC_QA, media=[_PDF], context_cache=ContextCache(media=True))
    )
    parts = fake.calls[0]["contents"][0].parts
    assert getattr(parts[0], "inline_data", None) is not None  # media first
    assert getattr(parts[-1], "text", None)  # question last

    await provider.complete(spec, Request(messages=_DOC_QA, media=[_PDF]))  # no cache → unchanged
    parts = fake.calls[1]["contents"][0].parts
    assert getattr(parts[0], "text", None)


async def test_grok_media_precedes_the_question_when_cached() -> None:
    fake = _FakeGrok(_grok_ok())
    img = MediaPart(mime_type="image/png", data=b"\x89PNG")
    provider = GrokProvider(client=fake)
    spec = ModelSpec(provider="grok", model="grok-4.3")
    await provider.complete(
        spec, Request(messages=_DOC_QA, media=[img], context_cache=ContextCache(media=True))
    )
    assert fake.calls[0]["messages"][0].content[0].HasField("image_url")  # media first

    await provider.complete(spec, Request(messages=_DOC_QA, media=[img]))  # no cache → unchanged
    assert not fake.calls[1]["messages"][0].content[0].HasField("image_url")
