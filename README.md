<p align="center">
  <img src="https://raw.githubusercontent.com/iliatankelevich/maslul/main/docs/assets/maslul-logo2.png" alt="maslul" width="700">
</p>

# maslul

**Smart LLM router — one call, the right model.**

Async and fully typed, across Anthropic,
Gemini, and xAI Grok — routing each request to the right model tier by difficulty. Stop
hardcoding model choices and stop re-writing the tool-use / structured-output / retry plumbing
for every provider.

`maslul` (Hebrew *מסלול*, "route / lane") is a small library that does exactly two things:
**routing** (pick a model tier per request, or pin one) and **provider normalization** (one
`Request`/`Response` shape for all three SDKs). No server, no CLI, no heavy ML deps — providers
live behind extras, and the core is stdlib-only.

```python
import asyncio
from maslul import Router, Request, Message

router = Router.from_toml("maslul.toml")           # tiers + classifier + providers, from config

async def main() -> None:
    resp = await router.complete(Request(messages=[Message(role="user", content="Hello!")]))
    print(resp.text, "·", resp.level_used, "·", resp.usage.output_tokens, "tokens")

asyncio.run(main())
```

## Install

```bash
pip install "maslul[anthropic,gemini,grok]"     # or just the providers you use
```

Each provider's SDK lives behind an extra, so `import maslul` pulls in **none** of them — you
only install what you route to. `maslul[anthropic]` → `anthropic`; `maslul[gemini]` →
`google-genai`; `maslul[grok]` → `xai-sdk`; `maslul[openai]` → `openai`.

## Routing

Difficulty is **not** readable from surface features — a short prompt can be very hard, a long
paste trivial — so maslul never applies a `short ⇒ simple` rule. You choose how each request is
routed, in this precedence order:

```python
from maslul import Level

await router.complete(req, model="anthropic:claude-opus-4-8")  # 0. pin an exact model
await router.complete(req, level=Level.HARD)                   # 1. pin a difficulty tier
await router.complete(req)                                     # 2-4. let the router decide
```

When you don't pin, the **routing brain** runs: a deterministic **bypass** (your fast-path, e.g.
greetings → SIMPLE) → a **hard-signal** detector (intent verbs, code, attachments, long context →
HARD, *up-only*) → the configured **strategy** for the ambiguous middle:

| Strategy | Cost for the middle | What it does |
|---|---|---|
| `ROUTE_DEFAULT` | 0 calls | Default-to-capable (`default_level`). Best for low volume. |
| `CLASSIFY` | 1 classify + 1 answer | A cheap dedicated classifier model labels the level (cached + budget-guarded), then dispatch. |
| `CLASSIFY_AND_ANSWER` | 1 call | The classifier model answers directly, or emits an escalation sentinel to bump to a stronger tier. |
| `VERIFY_CASCADE` | 1 cheap + verify | Answer cheap, run your verifier, escalate if it rejects — catches silent under-escalation. |

All three injection points are yours to supply:

```python
def my_classifier(req):      # your own difficulty call (sync or async); None defers to the strategy
    return Level.SIMPLE if is_trivial(req) else None

def my_verifier(req, resp):  # VERIFY_CASCADE: True keeps the cheap answer, False escalates
    return "I don't know" not in resp.text

router = Router.from_toml("maslul.toml", classifier=my_classifier, verifier=my_verifier)
```

## One shape for every capability

The same `Request`/`Response` works across all three providers:

```python
from maslul import Request, Message, ToolDef, ToolCall, MediaPart

# Tools — the router runs a provider-agnostic tool-use loop
async def get_weather(call: ToolCall) -> str:
    return f"18°C in {call.input['city']}"

req = Request(
    messages=[Message(role="user", content="Weather in Paris?")],
    tools=[ToolDef(name="get_weather", description="Current weather for a city.",
                   input_schema={"type": "object", "properties": {"city": {"type": "string"}},
                                 "required": ["city"]})],
    tool_executor=get_weather,
)

# Structured output — response_format → resp.structured (parsed)
req = Request(messages=[Message(role="user", content="Extract name + age")],
              response_format={"type": "object", "properties": {"name": {"type": "string"},
                                                                "age": {"type": "integer"}}})

# Vision — images / PDFs
req = Request(messages=[Message(role="user", content="What's in this image?")],
              media=[MediaPart(mime_type="image/png", data=png_bytes)])

# Web search — one flag, grounded on ANY provider (Anthropic web_search / Gemini Google Search /
# Grok Agent Tools); citations land in resp.sources regardless of which model answers.
req = Request(messages=[Message(role="user", content="Latest news on X?")], web_search=True)
```

## Resilience & observability

```python
def on_usage(resp):                         # per-model token breakdown for monitoring
    for rec in resp.usage_records:
        metrics.incr(f"{rec.provider}:{rec.model}", rec.usage.output_tokens)

router = Router.from_toml("maslul.toml", on_complete=on_usage)
```

Transient errors (`RateLimited`, `Timeout`) retry with exponential backoff; on persistent failure
the request **falls back to the next-higher tier** — which may be a different provider, giving you
cross-provider failover for free. `AuthError` fails fast. Hooks: `on_route` (the `RoutingDecision`),
`on_complete` (the final `Response` with `usage_records`), `on_error` (each failed attempt).

## Configuration

A TOML file (or a plain `dict` — `Router(config={...})`):

```toml
[maslul]
strategy = "route_default"        # route_default | classify | classify_and_answer | verify_cascade
default_level = "hard"            # default-to-capable for the ambiguous middle
min_tokens_to_classify = 40       # CLASSIFY budget guard
request_timeout = 60              # per-call seconds (optional)
max_retries = 2
fallback = true                   # escalate to a higher tier on persistent failure

[maslul.tiers.simple]
provider = "gemini"
model = "gemini-2.5-flash-lite"
[maslul.tiers.medium]
model = "anthropic:claude-haiku-4-5"   # or the provider:model shorthand
[maslul.tiers.hard]
model = "anthropic:claude-sonnet-4-6"

[maslul.classifier]               # required for the classify strategies
model = "anthropic:claude-haiku-4-5"

[maslul.providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"      # secrets by env-var name, never inlined
[maslul.providers.gemini]
vertex_project = "my-gcp-project"      # Vertex AI + Application Default Credentials (no key)
vertex_location = "global"
[maslul.providers.grok]
api_key_env = "XAI_API_KEY"
```

Pointing a capability at a different model or provider is a one-line config change — no code
deploy. Providers can also be injected directly (`Router(config, providers={...})`) for tests or
custom wiring.

## Providers

| Provider | SDK (extra) | Auth |
|---|---|---|
| `anthropic` | `anthropic` | `ANTHROPIC_API_KEY` |
| `gemini` | `google-genai` | Vertex AI + ADC (`vertex_project`), or a Gemini Developer API key |
| `grok` | `xai-sdk` | `XAI_API_KEY` |
| `openai` | `openai` | `OPENAI_API_KEY` |

## Status

Beta (`0.2.x`), fully typed (`py.typed`), async-first. Routing, tool use, structured output,
vision, **web search across all three providers** (`web_search=True`), the four strategies, and
retry/fallback resilience are implemented and exercised against live APIs.

## License

[MIT](LICENSE) © Ilia Tankelevich
