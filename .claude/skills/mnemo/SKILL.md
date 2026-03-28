---
name: mnemo
description: >
  Build and extend Mnemo — a drop-in LLM caching proxy written in Rust.
  Use this skill whenever the user wants to work on Mnemo, including:
  writing the Rust proxy core, designing the LoRA fine-tuning loop,
  implementing cost-aware eviction, building staleness detection, designing
  the OpenAI-compatible API surface, setting up the async intelligence layer,
  writing SDK integrations (TypeScript, Python), or planning the federated
  weight-sharing architecture. Also trigger when the user asks about LLM
  caching, semantic cache design, embedding fine-tuning on live traffic,
  cache eviction algorithms, MCP tool caching, or ACP agent-to-agent
  caching in the context of this project.
---

# Mnemo

A self-hosted LLM caching proxy that learns your domain, detects staleness,
and makes cost-aware eviction decisions — getting cheaper and more accurate
the longer it runs.

Supports: plain LLM calls, MCP (Model Context Protocol), ACP
(Agent Communication Protocol). One proxy, every protocol.

---

## What Mnemo is

Any developer points their existing app at Mnemo instead of directly at
OpenAI / Vertex AI / Claude / any LLM. One line change. Nothing else breaks.

Every LLM request passes through the proxy. If a similar question was asked
before — return the cached answer instantly. If not — forward to the real
LLM, cache the response, move on.

Works for:
- Standard chat completion calls (OpenAI / Anthropic / Vertex / Ollama)
- MCP tool calls (Model Context Protocol)
- ACP agent-to-agent messages (Agent Communication Protocol)

---

## What makes Mnemo different

Three things no existing cache combines:

### 1. Task-specific auto-learning embeddings
The cache gets smarter about the specific domain over time. A medical app's
cache learns medical language. A coding app's cache learns code semantics.
Implemented via lightweight LoRA adapter fine-tuning on the app's own live
query-response traffic — silently, in the background, never touching the
hot path.

- Base model: MiniLM-L6 or BGE-small (frozen)
- Adapter: small LoRA layer (~10MB) fine-tuned incrementally
- Training signal: every query-response pair that flows through the proxy
- Guard: elastic weight consolidation to prevent catastrophic forgetting
- Result: after 30 days, the embedding model understands this app's language
  better than any generic embedding ever could

### 2. Learning-based staleness detection
Not TTLs. Not manual versioning. A model that watches behavioral signals and
evicts proactively when cached answers go stale.

Signals watched:
- Users correcting or rejecting cached responses
- Same prompt now getting a different answer from the LLM
- Semantic drift: similar queries clustering differently over time
- External change signals (optional: changelog / news feed integration)
- For MCP: tool schema changes trigger re-evaluation of affected cache entries

This runs entirely async. Nothing stale is ever served because eviction
happens before the next request, not during it.

### 3. Cost-aware eviction algorithm
Eviction priority is based on what a response actually cost to generate —
not just recency or frequency.

```
priority = (token_cost × hit_frequency × recency_score)
           ─────────────────────────────────────────────
              staleness_risk × embedding_uncertainty
```

- `token_cost`: actual $ cost logged when the response was generated
- `hit_frequency`: how often this entry gets served
- `recency_score`: time-decay weight
- `staleness_risk`: score from the staleness detection model
- `embedding_uncertainty`: how confidently the adapter maps this entry

A $2 response is protected. A $0.001 response is sacrificed first.
Standard LRU, LFU, and TinyLFU are all cost-blind. This is not.

---

## What Mnemo is NOT

- Not an AI judge running on every request
- Not a model router
- Not a RAG pipeline
- Not a monitoring/observability tool

The hot path is pure Rust. Overhead target: under 2ms always.

---

## Protocol support

### Plain LLM calls

Standard chat completion. Any provider. OpenAI-compatible surface.

```
App → Mnemo → OpenAI / Vertex / Claude / Ollama / any provider
```

Integration: change baseURL to point at Mnemo. That's it.

```typescript
const client = new OpenAI({
  apiKey: "...",
  baseURL: "http://localhost:8080/v1"  // point at Mnemo, not OpenAI
});
```

---

### MCP — Model Context Protocol

MCP gives models access to tools and external data sources. Mnemo sits
between the app and the MCP server, caching tool calls intelligently.

```
App → Mnemo → MCP server → tool execution
         ↑
    caches cacheable tool calls,
    passes through live/dynamic ones transparently
```

The core challenge: not all tool calls are cacheable. A weather tool
changes every hour. A user profile tool returns the same thing all day.
Mnemo classifies every tool call by cacheability — from explicit config
or learned automatically from traffic patterns.

```
Tool cacheability:
├── Static tools    → always cacheable
│   └── get_user_profile, list_documents, fetch_schema
├── Live tools      → never cacheable
│   └── get_weather, get_stock_price, get_current_time
└── Conditional     → cacheable based on inputs
    └── search_docs(query)  → same query = cacheable
        get_file(id)        → same file id = cacheable
```

**Auto-classification (learned):**
If a tool is not explicitly configured, Mnemo observes it over time.
Same inputs → same outputs consistently → marks cacheable.
Same inputs → different outputs → marks non-cacheable.
No config needed after warm-up period.

**MCP cache key:**
`hash(tool_name + cache_key_fields)` — only the fields that actually
affect the response are included. Not the full request.

**MCP staleness:**
When a tool's schema changes, Mnemo automatically invalidates all cached
entries for that tool. Tool schema versioning is tracked in L1.

**MCP config:**
```yaml
mcp:
  server_url: http://localhost:3000
  tools:
    get_user_profile:
      cacheable: true
    get_weather:
      cacheable: false
    search_docs:
      cacheable: true
      cache_key_fields: [query, index]
    get_file:
      cacheable: true
      cache_key_fields: [file_id]
```

---

### ACP — Agent Communication Protocol

ACP handles agent-to-agent communication. When agents call other agents
with the same task repeatedly — summarising the same document, answering
the same sub-question across multiple runs — Mnemo caches those calls
exactly like LLM calls.

```
Orchestrator agent
       │
       ├── → Sub-agent A: "summarise doc X"   ← Mnemo caches this
       ├── → Sub-agent B: "extract entities"  ← Mnemo caches this
       └── → Sub-agent A: "summarise doc X"   ← HIT — no agent call made
```

Why ACP caching matters more than plain LLM caching: agentic workflows
are highly repetitive. The same sub-tasks execute across multiple runs,
users, and orchestrators. Without caching, every agent call costs tokens.
With Mnemo, repeated agent calls are free.

**ACP cache key:**
`hash(agent_id + task_type + normalized_input)` — agent identity is always
part of the key. Same question to different agents = different cache entries.

**ACP message classification:**
```
├── Task messages    → cacheable (same task + input → same output)
├── Status messages  → never cacheable (live state)
├── Result messages  → stored as cache values, not keys
└── Error messages   → never cached
```

**ACP config:**
```yaml
acp:
  registry_url: http://localhost:4000
  agents:
    summariser-agent:
      cacheable: true
      cache_key_fields: [task, document_id]
    data-extractor-agent:
      cacheable: true
      cache_key_fields: [task, input_hash]
    monitor-agent:
      cacheable: false
```

---

## Architecture

### Layer map

```
Hot path (sync, every request — target < 2ms)
─────────────────────────────────────────────
Request (LLM / MCP / ACP)
  → Protocol detector         ← identifies request type
  → Cacheability check        ← MCP/ACP: is this call cacheable?
  → Normalize + hash          ← protocol-aware key construction
  → L1 exact hash (DashMap)  → HIT: return immediately
  → L2 HNSW vector index     → HIT: return immediately
  → Proxy to upstream         → return response to caller

Cold path (async, after response sent — invisible to caller)
────────────────────────────────────────────────────────────
→ Generate embedding
→ Insert into HNSW index
→ Log token/call cost
→ Feed training signal to LoRA adapter worker
→ Run staleness scorer
→ Update MCP tool classification if needed
→ Write to Redis L3 if cost > threshold
→ Update eviction priority scores
```

### Storage layers

| Layer | Technology | What lives here | Latency |
|-------|-----------|-----------------|---------|
| L1 | DashMap (Rust) | Exact hash → response | ~0.05ms |
| L2 | HNSW index (in-memory) | Pre-computed embeddings → response | ~0.5ms |
| L3 | Redis | Persistent cross-restart cache | ~2ms |

### Key Rust crates

| Crate | Purpose |
|-------|---------|
| `tokio` | Async runtime |
| `axum` | HTTP server |
| `dashmap` | Lock-free concurrent hashmap (L1) |
| `moka` | W-TinyLFU eviction (L1) |
| `hnsw_rs` | HNSW vector index (L2) |
| `deadpool-redis` | Redis connection pool (L3) |
| `sha2` | Prompt / request hashing |
| `serde` / `serde_json` | Serialization |
| `candle-core` | Local embedding inference |
| `tracing` | Structured logging |

### Intelligence layer (async workers)

These never block the hot path. All run as background tokio tasks:

1. **Embedding worker** — generates embeddings for new cache entries,
   inserts into HNSW index
2. **LoRA training worker** — incrementally fine-tunes the embedding
   adapter on accumulated query-response pairs
3. **Staleness worker** — scores cached entries for freshness, evicts
   proactively when staleness_risk exceeds threshold
4. **Cost tracker** — maintains cost log, updates eviction priority
   scores across L1/L2
5. **MCP classifier worker** — observes tool call patterns, auto-classifies
   tools as cacheable/non-cacheable from traffic
6. **ACP pattern worker** — tracks agent call repetition, updates
   cacheability scores per agent/task pair

---

## API surface

### OpenAI-compatible (plain LLM)
```
POST   /v1/chat/completions     → cached LLM call
POST   /v1/completions          → legacy completions
GET    /v1/cache/stats          → hit rate, savings, cost avoided
DELETE /v1/cache                → manual flush
GET    /health                  → liveness probe
```

### MCP endpoints
```
POST   /mcp/tools/call          → cached MCP tool call
GET    /mcp/tools/list          → pass-through (never cached)
POST   /mcp/tools/classify      → manual cacheability override
GET    /mcp/cache/stats         → MCP hit rate + savings
```

### ACP endpoints
```
POST   /acp/tasks               → cached agent task
GET    /acp/agents/list         → agent registry pass-through
GET    /acp/cache/stats         → ACP hit rate + savings
```

---

## Full config reference

```yaml
# mnemo.yaml

upstream:
  provider: openai              # openai | vertexai | anthropic | ollama | custom
  base_url: https://api.openai.com
  api_key: ${OPENAI_API_KEY}

cache:
  l1_max_entries: 10000
  l2_max_entries: 100000
  l3_redis_url: redis://localhost:6379
  l3_min_cost_threshold: 0.01

intelligence:
  embedding_model: bge-small-en
  lora_training_enabled: true
  staleness_detection_enabled: true
  training_batch_size: 32
  training_interval_seconds: 300

mcp:
  server_url: http://localhost:3000
  tools:
    get_user_profile:
      cacheable: true
    get_weather:
      cacheable: false
    search_docs:
      cacheable: true
      cache_key_fields: [query, index]

acp:
  registry_url: http://localhost:4000
  agents:
    summariser-agent:
      cacheable: true
      cache_key_fields: [task, document_id]
    monitor-agent:
      cacheable: false
```

---

## Project structure

```
mnemo/
├── crates/
│   ├── mnemo-proxy/            ← main binary (hot path)
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── server.rs           ← axum HTTP server
│   │   │   ├── proxy.rs            ← forward to upstream
│   │   │   ├── protocol.rs         ← detect LLM / MCP / ACP
│   │   │   ├── normalize.rs        ← protocol-aware normalization
│   │   │   └── config.rs
│   │   └── Cargo.toml
│   │
│   ├── mnemo-cache/            ← cache storage layer
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── l1.rs               ← DashMap exact cache
│   │   │   ├── l2.rs               ← HNSW semantic cache
│   │   │   ├── l3.rs               ← Redis persistence
│   │   │   └── eviction.rs         ← cost-aware eviction algorithm
│   │   └── Cargo.toml
│   │
│   ├── mnemo-intelligence/     ← async intelligence workers
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── embedder.rs         ← embedding generation
│   │   │   ├── lora.rs             ← LoRA adapter fine-tuning loop
│   │   │   ├── staleness.rs        ← staleness detection
│   │   │   └── cost.rs             ← token/call cost tracking
│   │   └── Cargo.toml
│   │
│   ├── mnemo-mcp/              ← MCP protocol handler
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── classifier.rs       ← tool cacheability classifier
│   │   │   ├── key.rs              ← MCP cache key builder
│   │   │   └── schema.rs           ← tool schema change detection
│   │   └── Cargo.toml
│   │
│   ├── mnemo-acp/              ← ACP protocol handler
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── classifier.rs       ← agent call cacheability
│   │   │   └── key.rs              ← ACP cache key builder
│   │   └── Cargo.toml
│   │
│   └── mnemo-sdk/              ← client SDKs
│       ├── ts/
│       └── python/
│
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── k8s/
│       └── helm/
│
├── mnemo.yaml
├── Cargo.toml                  ← workspace
└── README.md
```

---

## Key design rules — never break these

1. **Nothing slow runs on the hot path.** Embedding generation, LoRA
   training, staleness scoring, Redis writes, MCP classification — all
   async, all post-response.

2. **The hot path overhead must stay under 2ms.** Measure it. If a change
   pushes it over, revert or move it async.

3. **Rust binary never changes for intelligence updates.** All learning,
   model updates, and policy changes happen in the intelligence layer.
   The binary is infrastructure — compile once, run forever.

4. **Never share raw responses across tenants.** Federated learning shares
   adapter weight deltas only. This eliminates the semantic cache poisoning
   attack surface entirely.

5. **Cost is a first-class citizen.** Every cached entry — LLM, MCP, or
   ACP — must have a `generation_cost` field. Eviction without cost
   awareness is wrong.

6. **MCP live tools are never cached.** If cacheability is unknown,
   default to non-cacheable. A false negative (miss on a cacheable tool)
   is safe. A false positive (cached result for a live tool) is wrong.

7. **ACP cache keys always include agent_id.** Same question to different
   agents = different cache entries. Never omit agent identity from the key.

---

## Novelty vs existing tools

| Capability | GPTCache | Bifrost | LiteLLM | vCache | Mnemo |
|-----------|---------|---------|---------|--------|-------|
| Plain LLM caching | yes | yes | yes | yes | yes |
| MCP tool caching | no | no | no | no | **yes** |
| ACP agent caching | no | no | no | no | **yes** |
| Sub-2ms overhead | no | partial | no | no | yes |
| Domain-adaptive embeddings | no | no | no | no | **yes** |
| Learning-based staleness | no | no | no | no | **yes** |
| Cost-aware eviction | no | no | no | no | **yes** |
| Auto tool classification | no | no | no | no | **yes** |
| Federated weight sharing | no | no | no | no | **yes** |
| Single binary deploy | no | partial | no | no | yes |
| Provider agnostic | partial | yes | yes | no | yes |

---

## When writing code for Mnemo

- Hot path code lives in `mnemo-proxy`, `mnemo-cache`, `mnemo-mcp`,
  `mnemo-acp`. Keep it minimal. No allocations on the critical path.
  No blocking calls. No `unwrap()` in production — use `thiserror`.

- Intelligence code lives in `mnemo-intelligence`. It can be slower and
  more complex. It runs async and never blocks callers.

- Protocol detection happens first in `mnemo-proxy/src/protocol.rs` before
  any cache lookup. Detected protocol determines normalization strategy
  and cache key construction.

- MCP tool cacheability defaults to `false` when unknown. Never assume
  a tool is cacheable without explicit config or learned classification.

- ACP cache keys always include `agent_id`. Never omit it.

- All inter-crate communication uses async channels (`tokio::sync::mpsc`),
  not shared state where possible.

- The eviction formula in `mnemo-cache/src/eviction.rs` is the single
  source of truth for eviction across all protocols — LLM, MCP, and ACP.

- Embedding model inference runs only in `mnemo-intelligence/src/embedder.rs`
  using `candle-core`. Never import candle in `mnemo-proxy`.

- Config loads once at startup from `mnemo.yaml` and environment variables.
