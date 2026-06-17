# Changelog

All notable changes to **maslul** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-17

Provider capability parity for **web search**, and `CLASSIFY_AND_ANSWER` upgraded to a full turn.

### Added
- **Normalized web search across every provider.** Set `Request.web_search=True` (optional
  `web_search_max_uses`) and each provider enables its own grounding — Anthropic's `web_search`
  server tool, Gemini's Google Search, Grok's Agent Tools `web_search` — with citations normalized
  into `Response.sources`. The caller never picks a provider-specific mechanism, so swapping the
  answering model keeps web search working.

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
