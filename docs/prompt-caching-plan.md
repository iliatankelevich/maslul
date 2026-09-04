# Plan — portable prompt caching (context caching)

**Status:** phases **1–3 implemented** (unreleased, targeting v0.3.0); phase 4 deliberately not built.
Two deviations from the text below, both deliberate — the `Request` field is named **`context_cache`**,
not `cache` (`req.cache` collides with the response cache this document opens by warning about), and
the history breakpoint goes on the **last** turn, not the last *completed* turn (Anthropic's own
multi-turn guidance says the most-recently-appended turn, which caches strictly more). §7's open
questions are now answered — see the notes inline. Remaining: validate in Kippy (§8).
**Author's note:** written after measuring a real workload (see *Why now*). Every number below was
measured or read from a provider doc — nothing here is from memory.

> **Not to be confused with `cache.py`.** maslul already has a **response cache** (exact +
> semantic): *don't call the model at all*. This document is about **prompt caching**: *call the
> model, but pay ~10× less for the part of the prompt it has already seen*. Different layer,
> different failure modes, and they compose. To keep them apart, this document calls the new one
> **context caching**, and the proposed type is `ContextCache` — never just "cache".

---

## 1. Why now

Kippy (the primary consumer) files a family's documents and answers questions about them. Measured
on a real 58-page Hebrew PDF (Bituach Leumi claim, 4.1 MB), against `claude-sonnet-4-6`:

| Call | Input tokens | Cost |
|---|---:|---:|
| Filing (first 5 pages) | 10,672 | ~$0.03 |
| Answering a question (all 58 pages) | **100,067** | **~$0.30** |

Every follow-up question about the same document re-sends the same 100k tokens. Anthropic would
serve that at **0.1×** from its prompt cache — $0.30 → **$0.03** — but maslul cannot ask for it:
[`providers/anthropic.py`](../src/maslul/providers/anthropic.py) `_media_block()` builds the
document block with no `cache_control`, and `MediaPart` has no field to request one.

Two more gaps found while looking:

1. **The only caching maslul supports today is an Anthropic-shaped escape hatch.** Kippy caches its
   persona by hand-building Anthropic content blocks and smuggling them through
   `provider_options["system"]` (see `router.py::_with_guidance`, which explicitly knows about
   "Anthropic structured system: a list of content blocks (often cache_control-marked)"). Gemini
   and Grok get **no caching at all** — they receive `req.system` as plain strings. A portable
   library should not require the caller to know Anthropic's content-block schema.

2. **`Usage.input_tokens` does not mean the same thing on every provider.** All four providers *do*
   map cached tokens into `Usage` (checked — this was my first assumption and it was wrong). But
   they disagree about what the *uncached* number counts, and maslul maps each one straight
   through:

   | provider | maps `input_tokens` from | does that number include the cached tokens? |
   |---|---|---|
   | Anthropic | `usage.input_tokens` | **No** — cached tokens are reported separately |
   | OpenAI | `usage.prompt_tokens` | **Yes** — `cached_tokens` is a subset of it |
   | Grok | `usage.prompt_tokens` | **Yes** |
   | Gemini | `prompt_token_count` | **Yes** |

   So `input_tokens + cache_read_input_tokens` double-counts on three providers and is correct on
   one. Any cost or context-size math built on `Usage` — including Kippy's usage-metrics hook — is
   wrong by the size of the cache hit, in a way that gets *worse* the better caching works. This is
   a bug today, before any of the rest of this plan is built, and it will quietly corrupt exactly
   the dashboard we would use to prove caching works.

Gap (2) is independent of the rest and should ship first. It is also the reason to fix the
convention *before* adding caching, not after.

---

## 2. The provider landscape (the crux)

This is what makes a portable design hard: **Anthropic is the only provider that requires you to
mark what to cache.** The other three cache a matching *prefix* automatically.

| | Anthropic | Gemini (Vertex) | OpenAI | xAI Grok |
|---|---|---|---|---|
| **Mechanism** | Explicit `cache_control` breakpoints on content blocks | **Implicit** (auto) + **explicit** `CachedContent` objects | Implicit (auto) | Implicit (auto) |
| **Caller must act?** | **Yes** — no marker, no cache | No (implicit); yes for explicit | No | No |
| **Granularity** | Up to **4** breakpoints; caches the prefix up to each | Implicit: prefix. Explicit: a named server-side object | Prefix only | Prefix only |
| **Minimum to cache** | 2,048 tok (Sonnet 4.6) / 4,096 (Opus 4.x) / 1,024 (Sonnet 4.5) — *model-dependent* | **explicit** 4,096 (3.x, enforced with a 400); **implicit** documented as the same but measured much higher — ~10,000 on 3.8-flash | 1,024 tok | not documented |
| **TTL** | 5 min (default) or 1 h | Implicit: opaque. Explicit: caller-set `ttl` | 30 min (fixed on current models) | opaque |
| **Write premium** | 1.25× (5 m) / 2× (1 h) | none (implicit) | 1.25× | none documented |
| **Read discount** | **0.1×** | passed through automatically | ~0.1× | "substantially lower" |
| **Affinity knob** | — | — | `prompt_cache_key` | `x-grok-conv-id` header |
| **Usage field** | `cache_creation_input_tokens`, `cache_read_input_tokens` | `usage.cached_content_token_count` | `usage.prompt_tokens_details.cached_tokens` | `usage.prompt_tokens_details.cached_tokens` |

**The one lever that works on all four** is not a marker — it is **prompt layout**: stable content
first, volatile content last. Anthropic breakpoints, Gemini implicit caching, OpenAI's prefix match
and Grok's all key off the same thing: *the longest identical prefix*. Any byte that changes early
invalidates everything after it, on every provider.

### 2.1 maslul's current layout is cache-hostile

`_to_messages()` in **every** provider attaches media to the **last user message**
(`media_at = last_user_index(messages)`). For a document-QA flow that is the worst possible
position: the 100k-token PDF sits *after* the whole conversation, so the moment a second question
is appended the document's position shifts and the prefix differs. The most expensive, most stable
object in the request is parked in the most volatile slot — on all four providers.

This is the single highest-value fix in this plan, and it costs nothing at runtime.

---

## 3. Design

### 3.1 The contract: declare stability, don't command a mechanism

The caller says *what is stable and for how long*. Each provider then does the best it can — an
explicit breakpoint, a server-side cache object, an affinity key, or nothing at all. A caller who
sets a `ContextCache` must never have to know which provider will answer.

```python
# types.py

@dataclass
class ContextCache:
    """Declares which parts of a request are stable enough to reuse across calls.

    This is a HINT, not a command: providers that cache implicitly (Gemini/OpenAI/Grok) honour it
    by ordering and affinity keys; Anthropic honours it with explicit breakpoints. A provider that
    can do nothing with it ignores it — the request still succeeds, it just costs full price.
    Savings are always reported the same way, in ``Usage.cache_read_input_tokens``.
    """

    system: bool = True          # persona + guidance (+ tools: they render before system)
    media: bool = False          # attachments — the expensive one (a 100k-token PDF)
    history: bool = False        # the conversation prefix, for long multi-turn chats
    ttl_seconds: int | None = None   # None = provider default (Anthropic 5 min, OpenAI 30 min)
    key: str | None = None       # affinity key: OpenAI prompt_cache_key / Grok x-grok-conv-id.
                                 # Use a stable per-conversation or per-document id.


@dataclass
class Request:
    ...
    cache: ContextCache | None = None   # None = today's behaviour, byte for byte
```

**Why not per-part flags** (`MediaPart(cache=True)`, `Message(cache=True)`)? Because Anthropic's
breakpoint budget is 4 and its semantics are *prefix up to here*, not *this block* — a per-part
flag invites callers to mark six things and silently lose two. A single declaration lets maslul own
breakpoint placement, which is the part callers get wrong.

### 3.2 The invariant maslul enforces for everyone: stable-first layout

When `req.cache` is set, maslul renders in this order **on every provider**:

```
[ system ] → [ tools ] → [ media ] → [ history ] → [ latest user turn ]
     ↑ stable ───────────────────────────────┘         ↑ volatile
```

Concretely this means media moves from *last* user message to the **first** user message when
`cache.media=True`. That is a behaviour change, which is why it is gated behind `cache.media`
rather than applied unconditionally.

⚠️ **Ordering alone delivers the Gemini/OpenAI/Grok win.** Those three need no other change. Do not
let the Anthropic breakpoint work block shipping the ordering fix.

### 3.3 Per-provider translation

**Anthropic** (`providers/anthropic.py`) — the only one that needs real work.

Render order is `tools → system → messages`, and a breakpoint caches *everything before it*. So:

- `cache.system` → put one breakpoint on the **last system block**. This caches **tools + system**
  together (tools render first) — one breakpoint, two wins.
- `cache.media` → one breakpoint on the **last media block** in the first user message.
- `cache.history` → one breakpoint on the **last content block of the last completed turn**.
- `ttl_seconds >= 3600` → `{"type": "ephemeral", "ttl": "1h"}`, else `{"type": "ephemeral"}`.
- Budget: **max 4** breakpoints. maslul must count and drop the lowest-value one rather than let
  the API 400. Priority when over budget: media > system > history.
- Below the model's minimum (2,048–4,096 tok) the marker is silently ignored by the API — no error,
  just `cache_creation_input_tokens: 0`. maslul should not try to predict this; report it.

**Gemini** (`providers/gemini.py`) — two tiers.

- *Implicit* (default, Gemini 2.5+): nothing to do beyond §3.2 ordering. Free.
- *Explicit* (`client.caches.create(...)`): worth it only when the same large media is queried
  repeatedly and the caller set `ttl_seconds`. Create a `CachedContent` holding the system
  instruction + the media, then pass `cached_content=<name>` in `GenerateContentConfig`. This adds
  **state maslul does not currently have** — a cache handle to store, reuse and delete — so it is
  deliberately the last phase, and opt-in. Keying: hash of (model, system, media bytes); store in a
  bounded in-memory map (same shape as `cache.py`'s LRU), delete on eviction.

**OpenAI** (`providers/openai.py`) — one line.

- `prompt_cache_key = req.cache.key`. Automatic caching does the rest. OpenAI notes the key needs
  ~15 req/min to stay warm, so a per-document key is fine and a per-message key is useless — worth
  a docstring warning.

**Grok** (`providers/grok.py`) — one header.

- `x-grok-conv-id: req.cache.key`. Confirm the `xai_sdk` surface for custom headers before
  committing to this (see §7).

### 3.4 Fix the `Usage` convention (ships first, alone)

Every provider already reports cached tokens (§1, gap 2) — what is missing is a **stated
convention** for what `input_tokens` excludes. Pick **Anthropic's**, because it is the one that maps
to money:

> `Usage.input_tokens` = tokens billed at **full** price.
> `Usage.cache_read_input_tokens` = tokens billed at the **discounted** cache-read price.
> `Usage.cache_creation_input_tokens` = tokens billed at the **write** premium (Anthropic only).
> The three are **disjoint**; the prompt's true size is their sum.

That makes cost a single formula that works for every provider, which is the whole point of the
`Usage` type. Implementation: on OpenAI, Grok and Gemini, **subtract** the cached count from the
provider's total before storing it.

```python
# providers/_common.py
def split_input(total: int, cached: int) -> tuple[int, int]:
    """Providers that report a cached-inclusive prompt total → maslul's disjoint convention.
    Clamped: a provider that ever reports cached > total must not yield negative billable tokens."""
    return max(total - cached, 0), cached
```

Note Grok reads `cached_prompt_text_tokens` (not `prompt_tokens_details.cached_tokens` as the
OpenAI-compatible surface suggests) — keep that; just subtract it.

This is a **behaviour change to a public field**, so it needs a CHANGELOG entry and a note in the
`Usage` docstring. Anyone summing `input_tokens` today is over-counting; after this they are not.

---

## 4. Interactions with maslul's own machinery (the traps)

These are the things a caching PR will break if it is written without reading `router.py`.

1. **The cache is model-scoped; the router picks the model.** A cache written on the `simple` tier
   is cold on `hard`. If `CLASSIFY_AND_ANSWER` escalates, the write premium (1.25×) is paid for
   nothing. **Rule:** only emit cache markers when the model is *pinned* (`complete(req,
   model=...)`) or the strategy cannot escalate. Kippy's document path pins the vision model, so it
   qualifies; the conversation path escalates, so it should cache the system prefix only (cheap to
   re-write) and never the media.

2. **`_with_guidance()` mutates the system prefix.** The CLASSIFY strategies prepend a guidance
   string to `system`. Guidance is constant, so it is cacheable — **but it must sit inside the
   cached span**, i.e. the breakpoint goes after guidance *and* persona, never between them. If a
   future guidance string ever becomes dynamic (a timestamp, a tier name), it silently kills every
   cache hit downstream. Add a test that asserts the rendered guidance is byte-stable.

3. **Tool definitions render before system on Anthropic.** So a caller that changes the tool set
   per-turn (Kippy adds admin tools only for the admin) gets a different prefix per user and shares
   nothing. Worth documenting; not worth fixing in maslul.

4. **The response cache (`cache.py`) sits in front.** On a response-cache hit no provider call
   happens, so no prompt-cache write happens either. That is correct, and the zeroed-usage
   convention already in `cache.py` keeps the accounting honest. No change needed — just don't let
   the two get conflated in naming.

---

## 5. Phasing

Each phase is independently shippable and independently valuable.

| Phase | What | Risk | Value |
|---|---|---|---|
| **1. Accounting** | Make `input_tokens` / `cache_read` / `cache_creation` disjoint on all four providers (§3.4) | low (public-field semantics change → CHANGELOG) | Cost math stops being wrong. Must land **before** the rest, or we cannot measure whether the rest worked |
| **2. Layout** | `ContextCache` type + stable-first ordering + `key` → OpenAI/Grok affinity | low | The full win on Gemini/OpenAI/Grok, with **zero** provider-specific caching code |
| **3. Anthropic** | `cache_control` breakpoints (system, media, history) + the 4-breakpoint budget | medium | The measured 10× on Kippy's document path — the reason this plan exists |
| **4. Gemini explicit** | `client.caches.create()` + cache-handle lifecycle | high (introduces state maslul has never had) | ~~Only if phase 2's implicit caching proves insufficient — **measure first**~~ **Measured 2026-09-04: it is insufficient in the 4,096–10,000 token band. See §7.2.** |

~~Phase 4 may never be needed. Do not build it speculatively: Gemini caches implicitly, and phase 2
may already capture all of it.~~

**Phase 4 is now justified by measurement, not speculation** — the gate this table set has been
met. See outcome 2 in §7 for the numbers and the dead band they describe.

---

## 6. Testing

- **Unit (fakes):** assert breakpoint *placement*, not just presence — e.g. exactly one
  `cache_control` on the last system block; media block carries one when `cache.media=True`; ≤4
  total; none at all when `req.cache is None` (the no-regression guarantee).
- **Unit:** the disjoint-`Usage` convention per provider (§3.4) — feed each provider's raw usage
  shape through and assert `input_tokens + cache_read == provider total`.
- **Integration (live, opt-in):** the only test that proves anything — send the same large request
  **twice** and assert `cache_read_input_tokens > 0` on the second. Run per provider. Anything less
  is testing our own mock.
- **Import isolation:** unchanged — `ContextCache` is a stdlib dataclass in `types.py`, no SDK.

---

## 7. Open questions — **ANSWERED** (checked against the installed SDKs, not from memory)

1. **Grok custom headers → NO. The affinity key is a documented no-op.** `xai_sdk`'s async
   `chat.create()` exposes **no** per-request header parameter, so there is no way to send
   `x-grok-conv-id` (the client-level `metadata` tuple is set once at construction, and maslul
   builds the client once by design). Its `conversation_id` parameter is *not* a cache knob — the
   SDK docstring says it "is added as a span attribute (`gen_ai.conversation.id`)" for
   **OpenTelemetry** tracing. Passing it as a cache key would fake the feature. Grok still gets the
   full §3.2 ordering win. Documented in `providers/grok.py`.
2. **Gemini explicit `CachedContent` → RE-OPENED 2026-09-04. The measurement came back, and
   implicit caching is insufficient for a tool-using agent.** Phase 4 shipped as not-built on the
   reasoning that implicit caching "may already capture all of it". Measured against
   `gemini-3.8-flash` on Vertex (kippy-499107/global), it captures none of it at realistic size:

   | stable prefix | implicit `cache_read` | explicit `CachedContent` |
   |---|---|---|
   | 2,283 tok | 0 | refused — `400: minimum token count to start explicit caching is 4,096` |
   | 4,203 tok | 0 | **4,203 cached** |
   | 4,563 tok | 0 | **4,563 cached** |
   | 5,043 tok | 0 | **5,043 cached** |
   | ~10,800 tok | 8,163 | (not tested — implicit already works) |
   | ~65,000 tok | 61,404 | (not tested) |

   **There is a dead band between the explicit floor (4,096) and roughly 10,000 tokens where
   implicit caching returns nothing and explicit caching returns everything.** That band is exactly
   where an agent with a system prompt plus tool declarations lives: the consumer that motivated
   this plan sits at ~4,420 tokens (persona + guidance + 17 tool schemas) and gets `cache_read=0` on
   three identical back-to-back calls, while an explicit cache at the same size returns ~99% of the
   prompt as a cache read.

   Two things also ruled out while measuring, both worth not re-testing:
   - **Tool declarations do not break implicit caching.** With the same ~13,500-token prefix, both
     with and without 17 tools cached (11,674 and 8,169 read respectively). There is an open
     `vercel/ai` issue claiming otherwise; it does not reproduce here.
   - **The §3.2 ordering is correct and is not the problem.** The consumer already sends
     stable-first / volatile-last, and the miss is purely the size floor.
3. **Small requests → no regression, and now asserted.** Reordering only happens when
   `context_cache.media` is set, and on a single-turn request the first user message *is* the last
   one, so the layout is unchanged. `context_cache=None` is byte-for-byte identical to the old
   behaviour — pinned by `test_*_media_stays_last_without_a_context_cache` on all four providers.
4. **Anthropic minimum per model → confirmed, and deliberately not modelled.** Below the minimum
   (2,048 Sonnet / 4,096 Opus) the API **silently ignores** the marker — no error. maslul hardcodes
   no table; it emits the marker and surfaces the truth as `cache_creation_input_tokens == 0`.
5. **Bonus, resolved while implementing:** the 1-hour TTL **no longer needs a beta header**
   (`{"type": "ephemeral", "ttl": "1h"}` on the plain endpoint), and OpenAI's SDK natively accepts
   `prompt_cache_key` *and* `prompt_cache_retention: "in_memory" | "24h"` — so `ttl_seconds` is
   honoured on OpenAI too, not just Anthropic.

---

## 8. Validation before release

Per the project rule: **integrate in Kippy via a git dependency and validate there before any PyPI
release.** The acceptance test is concrete and already has a fixture — Kippy's 58-page document:

1. Ask a question about it. Expect `cache_creation_input_tokens ≈ 100,000`, cost ≈ $0.375.
2. Ask a second question within the TTL. Expect `cache_read_input_tokens ≈ 100,000`,
   `input_tokens` ≈ the question only, cost ≈ **$0.03**.
3. Confirm Kippy's usage-metrics hook reports the saving rather than a phantom re-spend.

If step 2 does not show a read, the feature does not work — regardless of what the unit tests say.
