# maslul — Architecture

This document describes how maslul is built and why. For usage, see the [README](../README.md).

maslul does exactly two things, and nothing else:

1. **Routing** — decide which model *tier* (or exact model) handles each request.
2. **Provider normalization** — present one `Request`/`Response` shape over the Anthropic,
   Google Gemini, and xAI Grok SDKs, so callers write the loop once.

Everything below serves those two jobs. There is no server, no CLI, and no agent framework.

---

## Package layout & the SDK-free core

```
src/maslul/
├─ __init__.py        public API (re-exports; __version__)
├─ types.py           the contracts — Request/Response/Usage/ModelUsage, Level, Strategy,
│                       ModelSpec, ToolDef/ToolCall, MediaPart, RoutingDecision, hook aliases
├─ errors.py          MaslulError hierarchy
├─ config.py          RouterConfig — parse from TOML or a dict
├─ router.py          Router — the routing brain, the tool loop, resilience, hooks
└─ providers/
   ├─ base.py         Provider Protocol
   ├─ __init__.py     build_provider() factory (lazy, SDK-free)
   ├─ _common.py      tiny shared helpers (no SDK)
   ├─ anthropic.py    AnthropicProvider   (extra: anthropic)
   ├─ gemini.py       GeminiProvider      (extra: gemini)
   └─ grok.py         GrokProvider        (extra: grok)
```

**The core is stdlib-only and import-light.** `import maslul` pulls in *no* provider SDK — a
regression test (`tests/test_import_isolation.py`) asserts that in a subprocess. This is what lets
`pip install maslul[anthropic]` work without `google-genai` or `xai-sdk` present.

The mechanism: each provider *module* imports its SDK at module top level, but nothing on the
`import maslul` path imports those modules. `router.py` reaches `providers/__init__.py` (SDK-free),
and `build_provider()` **defers** importing the concrete provider module until the moment a provider
is actually constructed:

```python
def build_provider(name, config):
    if name == "anthropic":
        from maslul.providers.anthropic import AnthropicProvider   # ← SDK imported only here
        return AnthropicProvider(api_key=_env(config.get("api_key_env")))
    ...
```

Clients are constructed once, when you build the `Router` (or when you inject them), and reused for
every request — including across tool-loop iterations. Nothing is reconstructed per call.

---

## The normalized contract

Every provider speaks the same `Request` → `Response`:

```python
Request(messages, system, tools, tool_executor, response_format, media, server_tools,
        web_search, web_search_max_uses, max_tokens, temperature, stop, provider_options, metadata)

Response(text, level_used, provider, model, usage, structured, tool_calls, finish_reason,
         sources, classification_usage, usage_records, raw)
```

Two escape hatches keep the abstraction from ever boxing you in:

- **`provider_options`** — a raw passthrough merged into the underlying SDK call. Anything maslul
  doesn't model first-class (Anthropic prompt caching / `thinking` / `effort`, etc.) goes here.
- **`Response.raw`** — the untouched SDK response object.

`Difficulty is not readable from surface features.` A short prompt can be very hard and a long
paste trivial, so maslul **never** applies a `short ⇒ simple` rule. That principle shapes the whole
routing design below.

---

## The routing brain (`router.py`)

`Router.complete(req, *, level=None, model=None, strategy=None)` resolves a model in a fixed
precedence order, then runs it:

```
0. model= pinned         → that exact provider:model, no routing
1. level= pinned         → that tier
2. deterministic bypass  → an injectable predicate picks a tier (e.g. greetings → SIMPLE)
3. hard-signal detector  → HARD, UP-ONLY (intent verbs HE+EN, fenced code, media, long context)
4. ambiguous middle      → the configured Strategy
```

The pre-stage (2, 3) is **asymmetric**: bypass may pick any tier, but the hard-signal detector only
ever escalates *up* to HARD — misrouting up costs money, misrouting down costs correctness, so when
uncertain maslul prefers the capable tier. Both the bypass predicate and the hard-signal detector
are injectable; `default_hard_signal` is the built-in.

### Strategies (step 4 only)

| Strategy | Model calls | Mechanism |
|---|---|---|
| `ROUTE_DEFAULT` | 0 | Use `default_level` (default-to-capable). |
| `CLASSIFY` | 1 classify + 1 answer | A dedicated cheap classifier model emits a constrained `{"level": …}`, cached by prompt hash and skipped below `min_tokens_to_classify`; then dispatch to that tier. |
| `CLASSIFY_AND_ANSWER` | 1 | The classifier model answers directly, or emits `⟦MASLUL::ESCALATE::hard⟧` as its whole output to bump to a stronger tier (the original request is re-dispatched). |
| `VERIFY_CASCADE` | 1 cheap + verify | Answer at the cheapest tier, run a caller-supplied `verifier`; if it rejects, escalate to the most capable tier. |

A caller-supplied **`classifier`** (sync or async) takes precedence over the configured strategy for
the middle — that's how you plug in your own difficulty logic. The result of routing is a
`RoutingDecision` (`spec`, `level`, `reason`, optional `classification` usage), handed to the
`on_route` hook before the model runs.

---

## Provider abstraction — the hard 80%

A backend implements one Protocol and nothing else:

```python
class Provider(Protocol):
    name: str
    async def complete(self, spec: ModelSpec, req: Request) -> Response: ...
    async def healthcheck(self, spec: ModelSpec) -> None: ...
```

**Two loops, two owners** — this split is the key design decision:

- **Client-side tools** (your functions) are looped by the **`Router`**, provider-agnostically: call
  the model, surface `tool_calls`, run `req.tool_executor`, append the results as normalized
  messages, repeat until the model stops (bounded by an iteration guard). Because it works on
  *normalized* messages, the same loop drives all three providers.
- **Server-side tools** (web search) are resolved **inside the provider**. The normalized way to ask
  for it is **`Request.web_search=True`** — every provider then enables its own grounding and
  normalizes citations into `Response.sources`, so the caller never picks a provider-specific
  mechanism (swap the answering model and search keeps working):
  - `anthropic` → the `web_search` server tool; loops on `stop_reason == "pause_turn"`, echoing back
    the raw assistant content (which the normalized loop deliberately doesn't carry) and accumulating
    usage across resumes.
  - `gemini` → a `google_search` grounding `Tool`; sources from the grounding metadata.
  - `grok` → the xAI Agent Tools `web_search` tool; sources from the response citations.
  The raw `Request.server_tools` passthrough remains for advanced Anthropic-specific specs.

The two compose: a single `provider.complete` call resolves any web searches internally, then
returns either a client `tool_use` (which the Router loop handles) or the final answer. The
`CLASSIFY_AND_ANSWER` strategy's inline answer runs that same Router loop, so the cheap answerer is a
full-capability turn (client tools + web search), still able to escalate via the sentinel.

Each provider also normalizes: messages/system, tool defs + the call/result round-trip, structured
output (`response_format` → `Response.structured`), vision (`MediaPart` image/PDF), token `Usage`
(incl. cache tokens), and **errors** → the `MaslulError` hierarchy so the resilience layer can act
on them.

| Provider | SDK | Auth | Notes |
|---|---|---|---|
| `anthropic` | `anthropic` | `ANTHROPIC_API_KEY` | `web_search` server tool (pause/resume), prompt caching via `provider_options` |
| `gemini` | `google-genai` | Vertex AI + ADC, or API key | `function_call`/`function_response`, `response_json_schema`, `google_search` grounding |
| `grok` | `xai-sdk` (gRPC) | `XAI_API_KEY` | stateful chat reconstructed each turn; Agent Tools `web_search` |

---

## Resilience (`router.py`)

The answer is wrapped in router-level resilience:

- **Retry** transient errors (`RateLimited`, `Timeout`) with exponential backoff + full jitter.
- **Per-call timeout** (`request_timeout`) mapped to a retryable `Timeout`.
- **Fallback**: on persistent failure, escalate to the next-higher tier — which may be a *different
  provider*, giving cross-provider failover from the tier config alone.
- **`AuthError` fails fast** — a key/permission problem won't be cured by retrying or falling back.

The `MaslulError` hierarchy: `ConfigError` and `ProviderError` under `MaslulError`, with
`RateLimited` / `Timeout` / `AuthError` under `ProviderError`. Providers map their SDK exceptions
into this hierarchy so the router treats every backend uniformly.

---

## Observability

Three hooks (constructor args or `router.on_*()` registration):

- `on_route(request, RoutingDecision)` — what was chosen and why (fires before the model call).
- `on_complete(Response)` — the final response.
- `on_error(request, ModelSpec, MaslulError)` — each failed attempt (retry or fallback).

For cost/usage monitoring, `Response.usage_records` is a **per-`(provider, model)` breakdown** — a
single request can span several models (a classifier + an answer model + tool-loop iterations), and
each is attributed separately; `Response.usage` is their sum.

---

## Configuration (`config.py`)

A TOML file (`Router.from_toml`) or a plain dict (`Router(config={...})`). Tiers map a `Level` to a
`ModelSpec`; the classifier is a separate entry (it's a cheap dedicated model under `CLASSIFY`, the
capable floor under `CLASSIFY_AND_ANSWER`). Models use the `provider:model` shorthand or split
`provider`/`model` keys. Secrets are referenced by **env-var name**, never inlined. Resilience knobs
(`request_timeout`, `max_retries`, `retry_base_delay`, `retry_max_delay`, `fallback`) live under
`[maslul]`. Pointing a capability at a different model or provider is a one-line config change.

When `providers` aren't injected, the Router auto-builds the ones named by the configured tiers and
classifier via `build_provider`. Inject them (`Router(config, providers={...})`) for tests or to
reuse an existing SDK client.

---

## Extending maslul: a new provider

1. Implement the `Provider` Protocol in `providers/<name>.py` (import its SDK at module top).
2. Translate `Request` ↔ the SDK and normalize the `Response` (text, `tool_calls`, `Usage`,
   `finish_reason`, errors → `MaslulError`).
3. Add the name to `KNOWN_PROVIDERS`, a branch in `build_provider`, and an optional-dependency extra.
4. Keep it out of the `import maslul` path so the SDK-free-core guarantee holds.

---

## Design principles

- **Difficulty is intrinsic, not surface-readable** — never `short ⇒ simple`; default to capable.
- **Escalate up, never silently down** — the pre-stage and fallbacks only add capability.
- **The core stays SDK-free** — providers are optional extras, imported lazily.
- **Don't box the caller in** — `provider_options` passthrough and `Response.raw` are always there.
- **Single purpose** — routing and provider normalization; everything else is out of scope.
