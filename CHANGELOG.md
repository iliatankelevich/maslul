# Changelog

All notable changes to **maslul** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2026-08-03

### Fixed

- **Gemini rejected every turn in which the model called two tools at once.** Gemini matches tool
  results **by position within a turn**, not by id: a `model` turn holding N `functionCall` parts
  must be answered by exactly **one** turn holding N `functionResponse` parts. `_contents` emitted
  one `Content` per tool result, so a two-call turn produced two single-part contents and the
  request was rejected outright:

  ```
  400 INVALID_ARGUMENT: Please ensure that the number of function response parts is equal to
  the number of function call parts of the function call turn.
  ```

  Consecutive `tool` messages now collapse into a single `tool` content — the same grouping the
  Anthropic provider has always done for `tool_result` blocks. Grok and OpenAI are unaffected: both
  key results by `tool_call_id` and take one message each, which is their correct wire format.

  Sequential tool use was never affected (1 call ↔ 1 result), which is why this survived 0.3.1 —
  only a model *choosing* to parallelize hit it, and then the whole turn was lost. Caught in
  production on `gemini-3.5-flash`, where a document-filing agent naturally called two tools in one
  turn. `tests/integration/test_providers_live.py::test_gemini_parallel_tool_loop_live` reproduces
  the 400 against Vertex without the fix and passes with it.

## [0.3.1] - 2026-07-14

### Fixed

- **Gemini 3 tool calling was broken after the first iteration.** Gemini 3 mints a **thought
  signature** on the `functionCall` part it emits, and requires it back, unmodified, when the tool
  loop replays that call alongside its result. `GeminiProvider` rebuilt the part from
  `ToolCall(name, input)`, so the signature was never captured and never returned — and the
  **second** request of the loop was rejected outright:

  ```
  400 INVALID_ARGUMENT: Function call is missing a thought_signature in functionCall parts.
  ```

  A single-shot call was fine; any turn that actually *used* a tool died. That made
  `gemini-3.5-flash` unusable as a routable tier for any agent with tools, which is the whole point
  of the tool loop. Gemini 2.x mints no signature and is unaffected.

  The signature could not be seen from where the code was looking: `_tool_calls` read the
  `resp.function_calls` convenience accessor, which yields bare `FunctionCall` objects, while the
  signature lives on the **`Part`** wrapping the call. It now walks the response parts.

### Added

- **`ToolCall.signature: bytes | None`** — opaque provider state a model attached to a specific
  tool call and requires back on replay. maslul never interprets it. Additive and optional;
  providers that mint no such state (Anthropic, OpenAI, Grok) leave it `None`.

## [0.3.0] - 2026-07-12

Portable **prompt caching** — phases 1–3 of [docs/prompt-caching-plan.md](docs/prompt-caching-plan.md).
Not to be confused with the **response cache** (`[maslul.cache]`), which skips the model call
entirely; this makes the model call but bills the prefix it has already seen at the cache-read rate.

### Changed

- **⚠️ BREAKING — `Usage.input_tokens` now means "tokens billed at full price" on every provider.**
  The three input fields are **disjoint**: `input_tokens` (full price) + `cache_read_input_tokens`
  (~0.1x) + `cache_creation_input_tokens` (write premium, Anthropic only) = the prompt's true size.
  Anthropic already reported it this way; **OpenAI, Grok and Gemini report a prompt total that
  *includes* the cached tokens**, and maslul passed that straight through — so
  `input_tokens + cache_read_input_tokens` **double-counted on three of the four providers**, by
  exactly the size of the cache hit. Any cost or context-size math built on `Usage` was wrong, and
  wrong in a way that got *worse* the better caching worked. Those three now subtract the cached
  count back out ([`providers/_common.py::split_input`](src/maslul/providers/_common.py)).
  **If you sum `input_tokens` today you were over-counting; after this you are not.** Consumers that
  add `input_tokens + cache_read_input_tokens` to get a prompt size are now correct on all four.

### Added

- **`ContextCache` + `Request.context_cache`** — a portable prompt-caching contract. The caller
  declares *what is stable and for how long* (`system`, `media`, `history`, `ttl_seconds`, `key`),
  never a mechanism, and each provider does the best it can. `context_cache=None` (the default) is
  today's behaviour byte for byte: no markers, no affinity key, media unmoved.
- **Stable-first layout — the free win, and the largest one.** Media used to attach to the **last**
  user message on *every* provider: the most expensive, most stable object in the request parked in
  the most volatile slot, so a second question about the same document shifted its position and
  re-billed all of it. With `ContextCache(media=True)` it moves to the **first** user message, into
  the prefix that Anthropic, Gemini, OpenAI and Grok all key their caches off. No provider-specific
  code, no runtime cost.
  **Two halves, and the second one is what makes it work.** `media=True` *also* emits the media
  blocks **before** the message's own text. A document-QA turn is a single user message holding both
  the question and the PDF — so moving media "to the first user message" is a no-op there (the first
  *is* the last), and the rendered blocks were `[text(question), document(pdf)]`. A breakpoint on the
  document therefore cached a prefix that **began with the volatile question**, so every new question
  was a fresh prefix and the hit rate was **zero**. Emitting `[document, text]` puts the stable
  document first, where a prefix cache can reach it. Measured against the live API on the exact
  Kippy shape: a follow-up question about a 79k-token PDF went **$0.237 → $0.024 (9.9×)**. Anthropic
  independently recommends document-before-text, so this costs no answer quality.
- **Anthropic `cache_control` breakpoints** — the only provider that requires explicit markers.
  `system` marks the last system block (which caches **tools + system** together, since tools render
  first and a breakpoint caches everything before it); `media` marks the last media block; `history`
  marks the last content block of the last turn. maslul owns placement within the API's
  **4-breakpoint budget**, counting any markers the caller hand-placed via `provider_options` and
  dropping the cheapest of its own rather than letting the request 400. `ttl_seconds >= 3600` opts
  into the 1-hour cache (2x write premium instead of 1.25x).
- **OpenAI affinity key** — `ContextCache.key` → `prompt_cache_key`; `ttl_seconds >= 3600` →
  `prompt_cache_retention="24h"`.

### Notes

- **`ContextCache.key` is a documented no-op on Grok** (and Anthropic and Gemini): `xai_sdk`'s
  `chat.create()` exposes no per-request headers, and its `conversation_id` is only an OpenTelemetry
  span attribute — not a cache-affinity knob. Grok still gets the full ordering win. Faking the key
  would have been worse than not having it.
- **The cache is model-scoped; the router picks the model.** A cache written on the `simple` tier is
  cold on `hard`, so an escalating strategy can pay the write premium twice for nothing. Pair
  `media=True` with a pinned model. maslul emits exactly the markers you ask for and does **not**
  silently drop them.
- Below a model's minimum cacheable prefix (2,048–4,096 tokens) Anthropic **silently ignores** a
  marker rather than erroring. maslul does not try to predict that — it surfaces as
  `cache_creation_input_tokens == 0`.
- Gemini *explicit* `CachedContent` (plan phase 4) is deliberately **not** built: Gemini caches
  implicitly, so measure whether the ordering fix already captured the win before adding
  server-side state maslul has never had.

## [0.2.1] - 2026-06-17

### Fixed
- **`CLASSIFY_AND_ANSWER` now self-escalates even when the caller pins the system via
  `provider_options["system"]`.** The escalate-or-answer guidance was only added to `req.system`,
  but Anthropic's prompt-caching pattern passes the system through `provider_options`, which
  *overrides* `req.system` — so a cached Anthropic classifier never saw the guidance and answered
  everything instead of escalating. The guidance is now also prepended to the pinned
  `provider_options["system"]` (string or content-block list), preserving the original system and
  its cache markers. Other providers are unaffected (they read `req.system`).

## [0.2.0] - 2026-06-17

A fourth provider (OpenAI), web-search parity across all providers, a response cache, graceful
provider fallback, and `CLASSIFY_AND_ANSWER` upgraded to a full turn.

### Added
- **OpenAI provider** (`maslul[openai]`) — text, tool use, structured output, vision, and web
  search (`web_search_options`) with `url_citation` annotations into `Response.sources`.
- **Normalized web search across every provider.** Set `Request.web_search=True` (optional
  `web_search_max_uses`) and each provider enables its own grounding — Anthropic's `web_search`
  server tool, Gemini's Google Search, Grok's Agent Tools `web_search`, OpenAI's `web_search_options`
  — with citations normalized into `Response.sources`. The caller never picks a provider-specific
  mechanism, so swapping the answering model keeps web search working.
- **Response cache** (`[maslul.cache]`) — `exact` or `semantic` (nearest request above a cosine
  threshold, via an injected `Router(embed=...)`); a hit returns with `cached=True` and zeroed usage.
  Tool-using requests are never cached.
- **Graceful provider fallback** — `Router(..., missing_provider="degrade")` remaps a tier (or
  classifier) whose provider isn't configured to the nearest available tier; `build_provider` now
  raises `ConfigError` when a credential is absent.

### Changed
- **`CLASSIFY_AND_ANSWER` is now a full-capability turn.** Its inline answer runs the
  provider-agnostic tool loop and honors web search (previously a single bare call that silently
  dropped client tool calls). The escalate-or-answer decision is still read from the first
  response; an answer then continues the loop seeded by it (no extra model call).
- The `Request.server_tools` raw passthrough remains for advanced Anthropic use; `web_search=True`
  is the portable path.

### Fixed
- `VERIFY_CASCADE`'s accepted cheap answer no longer gets re-finalized, preserving its per-model
  `usage_records` breakdown.

### Removed
- The Grok provider's deprecated xAI **Live Search** (`SearchParameters`); replaced by the Agent
  Tools API `web_search` tool.

## [0.1.0] - 2026-06-17

Initial release — an async, fully-typed LLM router across Anthropic, Gemini, and xAI Grok.

### Added
- **Routing brain.** One `Router.complete(...)` call. Pin an exact `model=`, pin a difficulty
  `level=`, or let the router decide: deterministic bypass → hard-signal detector (up-only,
  Hebrew + English) → strategy. Never a `short ⇒ simple` rule.
- **Strategies** for the ambiguous middle: `ROUTE_DEFAULT`, `CLASSIFY` (cheap dedicated classifier
  model, prompt-hash cache + `min_tokens_to_classify` budget guard), `CLASSIFY_AND_ANSWER`
  (escalation sentinel), and `VERIFY_CASCADE` (injectable verifier). Plus injectable
  `bypass_predicate`, `hard_signal`, and `classifier` hooks.
- **Provider normalization** behind one `Request`/`Response`: a provider-agnostic tool-use loop,
  structured output (JSON schema → `Response.structured`), and vision (image / PDF).
- **Anthropic server-side web search** — `pause_turn` resume + citations into `Response.sources`,
  via `Request.server_tools`.
- **Resilience** — retry with exponential backoff on transient errors, per-call timeout, and
  cross-tier (cross-provider) fallback on persistent failure; `AuthError` fails fast.
- **Observability** — `on_route` / `on_complete` / `on_error` hooks and a per-model token
  breakdown (`Response.usage_records`).
- **Providers behind extras**: `maslul[anthropic]`, `maslul[gemini]` (Vertex AI + ADC or API key),
  `maslul[grok]`. The core is stdlib-only; `import maslul` pulls in no provider SDK.
- Config from TOML or a plain dict (`Router.from_toml` / `Router(config=...)`), with the
  `provider:model` shorthand and env-var-referenced secrets.

[Unreleased]: https://github.com/iliatankelevich/maslul/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/iliatankelevich/maslul/releases/tag/v0.2.0
[0.1.0]: https://github.com/iliatankelevich/maslul/releases/tag/v0.1.0
