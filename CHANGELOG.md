# Changelog

All notable changes to **maslul** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **A retry or a fallback re-ran every tool the request had already run.** Each attempt restarted
  the tool loop from the request's own messages, so a `RateLimited` or `Timeout` after a tool
  call ran the tool again on the retry, and again on each tier the request fell back to. That is a
  wasted query for a read and a repeated side effect for a write. The loop now resumes: the next
  attempt continues from the results already in hand, on the same model or on the fallback tier. A
  tool that raised a retryable error runs again, and the calls beside it that returned do not. A
  tier's retries share its eight-iteration cap, and a fallback tier starts its own on the resumed
  transcript (before, every retry restarted the cap). A forced `tool_choice` is not forced again on
  a resumed turn, and `usage` / `usage_records` now include the model calls of an attempt that
  failed later, since their output is part of the transcript the answer was built on.
  `classify_and_answer`'s inline answer and `verify_cascade`'s escalation are unchanged.
- **Gemini gave two calls of one tool the same id.** With no id from the model, a call's id was the
  tool's name. It is now `<name>-<random>`, unique per call; an id the model sends is kept.
  Gemini still matches results by name, so only callers that keyed on the id see a difference.

### Added

- **Gemini 3 accepts tool calls another provider made.** Gemini 3 refuses a step of the current
  turn whose first `functionCall` carries no thought signature, which is what a fallback that
  resumes another provider's tool loop sends. Such a call now carries Vertex's documented
  validator-skip value. A call with its own signature, a call in an earlier turn, and every
  model before Gemini 3 are sent exactly as before.

## [0.6.0] - 2026-10-05

### Added

- **Generic generation parameters, translated per provider.** `Request` gains `top_p`, `top_k`,
  `seed`, `reasoning` (`"low"` / `"medium"` / `"high"`) and `tool_choice` (`"auto"` /
  `"required"` / `"none"` / a tool name), next to the existing `max_tokens`, `temperature` and
  `stop`. Each provider maps them onto its own SDK call and ignores the ones it has no equivalent
  for (`seed` on Anthropic, `top_k` on OpenAI and xAI). `reasoning` becomes adaptive thinking +
  `output_config.effort` on Claude 4.6 and later, a `budget_tokens` on older Claude models (Haiku
  4.5), `reasoning_effort` on OpenAI and xAI, and `thinking_config.thinking_level` on Gemini. With
  `reasoning` set, the sampling parameters reasoning models refuse are dropped instead of sent to a
  guaranteed 400 (`temperature`/`top_p`/`top_k` on Anthropic, `temperature`/`top_p`/`stop` on
  OpenAI, `stop` on xAI), and the default output cap rises, since thinking counts against it: from
  1,024 to 16,000 tokens on OpenAI and Claude 4.6+, and to the thinking budget plus 1,024 on older
  Claude models. A forced `tool_choice` applies to the tool loop's
  first turn only. An unknown `reasoning` level, or a `tool_choice` naming a tool the request does
  not define, raises `ConfigError`.
- **The same generic keys on a tier** (`temperature`, `top_p`, `top_k`, `seed`, `reasoning`, next
  to `max_tokens`), as that model's defaults; a request that sets the same key wins. Precedence,
  lowest to highest: tier generic keys, request generic fields, tier `options`, request
  `provider_options`.
- **Provider-scoped `provider_options`.** `{"anthropic": {...}, "gemini": {...}}`: each provider
  reads only its own entry, so one request can carry raw options for every tier it might be routed
  to — and this works on all four providers. The unscoped form keeps its old behaviour (applied on
  Anthropic and OpenAI, ignored on Gemini and xAI). `classify_and_answer`'s guidance also reaches a
  scoped `provider_options["anthropic"]["system"]`.

### Changed

- **The `grok` provider is now `xai`.** Providers are named for the API vendor, like `anthropic`
  and `openai`; Grok is xAI's model family, and the SDK (`xai-sdk`) and key (`XAI_API_KEY`) were
  already named for xAI. Model specs become `xai:grok-4.3`, the config table
  `[maslul.providers.xai]`, the extra `maslul[xai]`, and the driver
  `maslul.providers.xai.XAIProvider`. Model ids are unchanged.
- **Tier `options` now reach Gemini and xAI.** `[maslul.tiers.*] options` (and
  `[maslul.classifier] options`) were applied only by Anthropic and OpenAI; Gemini now merges them
  into `GenerateContentConfig` (`thinking_config`, `candidate_count`, …) and xAI into
  `chat.create()` (`reasoning_effort`, `parallel_tool_calls`, …).
- **The response cache keys on the generic parameters.** Two requests that differ only in
  `temperature`, `top_p`, `top_k`, `seed`, `stop` or `reasoning` no longer share a cached answer.
  Requests that set none of them keep the key they had.
- **Gemini's explicit cache skips a request with a forced `tool_choice`.** Vertex refuses
  `tool_config` next to `cached_content`, and a forced choice changes every turn, so the request
  goes out uncached rather than minting a `CachedContent` per turn.

### Fixed

- **The `model = "provider:model"` tier shorthand dropped `max_tokens` and `options`.** Only the
  split `provider` + `model` form read them, so a shorthand tier silently ignored both. Both forms
  now accept them.
- **`Request.stop` was ignored on xAI.** It is now sent as `chat.create(stop=...)`, like the other
  three providers.

### Deprecated

- **`grok` as a provider name.** It still works everywhere it used to (a `provider:model` spec, a
  tier's `provider` key, a `[maslul.providers.grok]` table, a `Router(providers={"grok": ...})`
  key, `build_provider("grok", ...)`, the `maslul[grok]` extra and `maslul.providers.grok`), is
  mapped to `xai`, and emits a `DeprecationWarning`. It will be removed in a future release.

### Docs

- README: drop the stale *Status* section (it still said `0.2.x` and three providers) in favour
  of PyPI, Python-version, CI and license badges; document all four routing hooks
  (`bypass_predicate`, `hard_signal`, `classifier`, `verifier`); list the `openai` and `jev`
  extras; add a *Model parameters* section (the generic-to-provider mapping table, raw options,
  precedence, and the model-specific refusals to know about); the Gemini config example moves off
  the access-limited 2.5 line.

## [0.5.0] - 2026-10-01

### Added

- **TypeSafe Jev hooks (`maslul[jev]`, `httpx`).** `JevClassifier` and `JevVerifier` — a
  `classifier` and a `verifier` hook backed by TypeSafe's System One API
  (https://docs.typesafe.ai). Jev is not an LLM provider (it returns typed, calibrated decisions,
  never text), so it plugs into the hook contract rather than `Provider`. `JevClassifier` asks one
  Choice question over the router's levels and returns a `Level` only above a configurable
  `min_confidence`; `JevVerifier` asks one Noul (yes/no probability) question for
  `VERIFY_CASCADE` and gates on `min_yes`. Both build a bounded, text-only conversation window as
  Jev's `state` (char-capped well under its 32k-token budget) and never raise into the router — any
  transport error, timeout, 429/529, or malformed answer resolves to `None` (classifier) or
  `on_error`'s policy (verifier, default `accept`, so an outage never doubles cost by escalating
  every turn). `decide()` / `judge()` expose the raw decision (probabilities, confidence, model,
  usage) for shadow-mode logging and threshold calibration — Jev's docs note English is its primary
  training language and other languages, including Hebrew, are "handled but not equally well".
  `maslul.jev.hooks_from_config()` builds both from a `[maslul.jev]` config table; `Router` has no
  generic hook-plugin registry, so this is a documented factory rather than a config-wired path,
  and an explicitly-injected `Router(classifier=..., verifier=...)` always wins.
- **Level descriptions are concrete examples, and configurable** (`[maslul.jev.criteria]`,
  `question`, `verify_question`). Abstract wording never let `hard` win: three clearly hard requests
  came back `medium` at 0.48 to 0.78 in English and Hebrew alike, and came back `hard` at 0.97 to
  0.99 with the new defaults. An unknown level name in `criteria` raises `ConfigError`.

## [0.4.0] - 2026-09-04

### Added

- **Gemini explicit context caching (`CachedContent`) — prompt-caching plan, phase 4.** Opt in by
  setting both `ContextCache(system=True, ttl_seconds=...)`; without a TTL nothing changes, because
  this is the one path that creates server-side state and that should be asked for, not inferred.

  Phase 4 was deliberately left unbuilt in 0.3.0 on the reasoning that Gemini "caches implicitly, and
  phase 2 may already capture all of it", gated on measuring it first. **Measured against
  `gemini-3.8-flash` on Vertex, it captures none of it at the size an agent actually runs at:**

  | stable prefix | implicit `cache_read` | explicit `CachedContent` |
  |---|---|---|
  | 2,283 tok | 0 | refused — `400: minimum token count to start explicit caching is 4,096` |
  | 4,203 tok | 0 | **4,203 cached** |
  | 4,563 tok | 0 | **4,563 cached** |
  | 5,043 tok | 0 | **5,043 cached** |
  | ~10,800 tok | 8,163 | (implicit already works) |

  There is a **dead band between the 4,096-token explicit floor and roughly 10,000 tokens** where
  implicit caching returns nothing and explicit returns essentially everything — and that band is
  exactly where a system prompt plus tool declarations lands.

  Behaviour worth knowing:
  - **The whole stable half moves into the cache, system instruction *and* tools.** Vertex rejects a
    request that sets `cached_content` together with `tools` or `system_instruction`
    (`"Tool config, tools and system instruction should not be set"`), so it is all-or-nothing.
    `google_search` caches fine and still grounds — verified, not assumed.
  - **`web_search` is part of the cache key.** The search tool lives inside the cache, so two
    otherwise-identical requests that differ only in `web_search` must not share a handle.
  - **A caching failure is never the caller's failure.** Under the model's minimum, out of quota, or
    any other error, the request is sent the ordinary way at full price — the same contract as an
    Anthropic breakpoint below the minimum being silently ignored.
  - **A cache that expires mid-flight is retried uncached, once.** TTL expiry is a race the library
    introduced, so the library pays for it rather than surfacing an error nobody can act on.
  - Handles are kept in a bounded per-provider LRU (32) and deleted server-side on eviction; a leaked
    cache expires on its own TTL.

### Changed

- `docs/prompt-caching-plan.md` records the measurement that re-opened phase 4, and corrects the
  provider table: the documented 4,096-token Gemini minimum holds for **explicit** caching, where the
  API enforces it with a 400. For **implicit** caching it is optimistic — nothing cached below
  ~10,000 tokens in these runs. Also documented: tool declarations do **not** break implicit caching
  (an open `vercel/ai` issue claims they do; it does not reproduce on Vertex).

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
  document first, where a prefix cache can reach it. Measured against the live API on a real
  document-QA workload: a follow-up question about a 79k-token PDF went **$0.237 → $0.024
  (9.9×)**. Anthropic
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

[Unreleased]: https://github.com/iliatankelevich/maslul/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/iliatankelevich/maslul/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/iliatankelevich/maslul/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/iliatankelevich/maslul/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/iliatankelevich/maslul/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/iliatankelevich/maslul/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/iliatankelevich/maslul/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/iliatankelevich/maslul/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/iliatankelevich/maslul/releases/tag/v0.2.0
[0.1.0]: https://github.com/iliatankelevich/maslul/releases/tag/v0.1.0
