# Mnemo Architecture

This document explains how Mnemo works internally — every crate, every file, every data flow. It is intended for contributors, auditors, and anyone who wants to understand what happens between a client request arriving and a response leaving.

---

## Table of Contents

- [High-Level Overview](#high-level-overview)
- [Request Lifecycle](#request-lifecycle)
- [Workspace Structure](#workspace-structure)
- [Crate: mnemo-proxy](#crate-mnemo-proxy)
- [Crate: mnemo-cache](#crate-mnemo-cache)
- [Crate: mnemo-intelligence](#crate-mnemo-intelligence)
- [Crate: mnemo-mcp](#crate-mnemo-mcp)
- [Crate: mnemo-acp](#crate-mnemo-acp)
- [Hot Path vs Cold Path](#hot-path-vs-cold-path)
- [Streaming (SSE) Pipeline](#streaming-sse-pipeline)
- [Staleness Detection](#staleness-detection)
- [LoRA Fine-Tuning Loop](#lora-fine-tuning-loop)
- [Cost-Aware Eviction](#cost-aware-eviction)
- [Multi-Provider Auth](#multi-provider-auth)
- [Logging](#logging)
- [Error Handling — Never Cache Errors](#error-handling--never-cache-errors)
- [Deployment](#deployment)
- [Design Rules](#design-rules)

---

## High-Level Overview

Mnemo is a transparent HTTP proxy that sits between your application and an LLM provider. It intercepts every request, checks a three-tier cache, and only forwards to the upstream provider on a cache miss. The key insight is that LLM responses are expensive, slow, and often repetitive — caching them with semantic understanding saves real money and latency.

```
Client Application
        │
        ▼
┌───────────────────────────────────────────────────┐
│                   Mnemo Proxy                      │
│                                                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐       │
│  │ L1 Cache │──▶│ L2 Cache │──▶│ L3 Cache │       │
│  │ (moka)   │   │ (HNSW)   │   │ (Redis)  │       │
│  │ ~0.05ms  │   │ ~0.5ms   │   │ ~2ms     │       │
│  └──────────┘   └──────────┘   └──────────┘       │
│        │              │              │             │
│        └──────────────┴──────────────┘             │
│                    MISS ▼                          │
│              ┌──────────────┐                      │
│              │   Upstream   │──▶ OpenAI / Anthropic│
│              │   Forwarder  │   / VertexAI / Ollama│
│              └──────────────┘                      │
│                                                    │
│  Background Workers:                               │
│  ┌────────────┐ ┌────────────┐ ┌────────────────┐  │
│  │ Embedding  │ │ Staleness  │ │ LoRA Training  │  │
│  │ Worker     │ │ Scanner    │ │ Worker         │  │
│  └────────────┘ └────────────┘ └────────────────┘  │
└───────────────────────────────────────────────────┘
```

Three things make Mnemo different from a dumb HTTP cache:

1. **Semantic matching** — "What is Rust?" and "Tell me about the Rust programming language" hit the same cache entry via vector similarity search.
2. **Learning-based staleness** — No TTLs. Mnemo uses temporal decay, hit patterns, correction signals, and response drift to decide when an entry is stale.
3. **Cost-aware eviction** — When the cache is full, cheap responses are evicted first. A $0.50 GPT-4 response with 100 hits is protected over a $0.001 response with 2 hits.

---

## Request Lifecycle

This is the complete path a request takes through Mnemo, from HTTP arrival to response delivery.

```
1. HTTP Request arrives at axum server
   │
2. Route matched (e.g., /v1/chat/completions)
   │
3. Protocol detection: LLM, MCP, or ACP
   │
4. Request body normalized (strip non-deterministic fields)
   │
5. SHA-256 hash computed → cache key
   │
6. L1 Lookup (DashMap + moka, in-memory exact match)
   │── HIT → Return cached response, set x-mnemo-cache: l1-hit
   │
7. L2 Lookup (extract query text → embed → HNSW search)
   │── HIT (similarity ≥ 0.92) → Return cached response, set x-mnemo-cache: l2-hit
   │
8. L3 Lookup (Redis GET, if configured)
   │── HIT → Promote to L1, queue embedding for L2, return response
   │
9. MISS → Forward to upstream provider
   │
10. Receive upstream response
    │
11. Error check: if response contains error/empty choices/content_filter → return to client, DO NOT cache
    │
12. Estimate generation cost from usage tokens
    │
13. Create CacheEntry with response + cost + metadata
    │
14. Insert into L1 (synchronous, hot path)
    │
15. Queue for L2 (async: send text to embedding worker channel)
    │
16. Insert into L3 if cost ≥ threshold (async Redis SET)
    │
17. Return response to client, set x-mnemo-cache: miss
```

For streaming requests (`stream: true`), steps 9-16 are different — see [Streaming Pipeline](#streaming-sse-pipeline).

---

## Workspace Structure

Mnemo is a Cargo workspace with five crates. Each crate has a single responsibility.

```
mnemo/
├── Cargo.toml              # Workspace root, shared dependency versions
├── mnemo.yaml              # Runtime configuration
├── Dockerfile              # Multi-stage build (~100MB final image)
├── docker-compose.yml      # Mnemo + Redis
│
├── crates/
│   ├── mnemo-proxy/        # HTTP server, routing, upstream forwarding
│   │   ├── src/
│   │   │   ├── main.rs     # Entry point: init, wire, bind
│   │   │   ├── lib.rs      # Module re-exports for integration tests
│   │   │   ├── server.rs   # AppState, route handlers, cache lookup chain
│   │   │   ├── proxy.rs    # Upstream client, auth, streaming, cost estimation
│   │   │   ├── config.rs   # YAML config parsing, provider enum
│   │   │   ├── normalize.rs# Request normalization and hashing
│   │   │   └── protocol.rs # LLM/MCP/ACP detection from URI path
│   │   └── tests/
│   │       └── integration.rs  # 10 end-to-end tests with mock upstream
│   │
│   ├── mnemo-cache/        # Three-tier cache implementations
│   │   └── src/
│   │       ├── lib.rs      # Module exports
│   │       ├── entry.rs    # CacheEntry struct (response + metadata)
│   │       ├── l1.rs       # In-memory exact-match (moka + DashMap)
│   │       ├── l2.rs       # Semantic similarity (HNSW vector index)
│   │       ├── l3.rs       # Persistent storage (Redis)
│   │       └── eviction.rs # Cost-aware priority formula
│   │
│   ├── mnemo-intelligence/ # ML/embedding/training subsystems
│   │   └── src/
│   │       ├── lib.rs      # Re-exports
│   │       ├── embedder.rs # Embedder trait, MockEmbedder, OnnxEmbedder, EmbeddingWorker
│   │       ├── staleness.rs# Staleness risk computation, StalenessWorker
│   │       ├── lora.rs     # LoRA adapter, EWC, training loop
│   │       └── cost.rs     # Cost tracker (placeholder)
│   │
│   ├── mnemo-mcp/          # Model Context Protocol support
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── classifier.rs  # Tool cacheability classification
│   │       ├── key.rs         # MCP-specific cache key building
│   │       └── schema.rs      # Schema change detection
│   │
│   └── mnemo-acp/          # Agent Communication Protocol support
│       └── src/
│           ├── lib.rs
│           ├── classifier.rs  # Message type classification
│           └── key.rs         # ACP-specific cache key building
```

---

## Crate: mnemo-proxy

This is the main crate. It contains both a library (for integration tests) and a binary (the actual server).

### main.rs — Entry Point

The `main()` function wires everything together:

1. Load `Config` from `MNEMO_CONFIG` env var (default: `mnemo.yaml`). Environment variable substitution happens here via `shellexpand`.
2. Initialize `tracing` subscriber using log level from `MNEMO_LOG` > `RUST_LOG` > config file > `"info"`. Format is either `pretty` (human-readable) or `json` (structured), controlled by `logging.format` in config.
3. Create `L1Cache` with configured max entries (default 10,000).
4. Create `L2SemanticCache` wrapped in `Arc` with configured max entries (default 100,000) and embedding dimension 384.
5. Optionally create `L3RedisCache` wrapped in `Arc` if a Redis URL is configured.
6. Create `EmbeddingWorker` with `MockEmbedder`, get the sender/receiver channels.
7. Spawn a background tokio task that reads `EmbedResult` from the receiver channel and inserts embeddings into L2.
8. Create `Upstream` proxy client from config.
9. Build `AppState` with all components.
10. Build axum `Router` via `server::router()`.
11. Bind TCP listener and serve.

### config.rs — Configuration

Parses `mnemo.yaml` into a strongly-typed `Config` struct. Key sections:

- **`UpstreamConfig`**: `provider` (enum: OpenAI, Anthropic, VertexAI, Ollama, Custom), `base_url`, optional `api_key`. The provider determines auth header format and endpoint remapping.
- **`CacheConfig`**: `l1_max_entries`, `l2_max_entries`, optional `l3_redis_url`, `l3_min_cost_threshold` (only cache responses costing above this in L3).
- **`LoggingConfig`**: `level` (error/warn/info/debug/trace), `format` (pretty/json), `log_requests` (per-request debug logging), `log_cache_events` (cache hit/miss info logging). See [Logging](#logging).
- **`IntelligenceConfig`**: `embedding_model`, `lora_training` flag, `staleness_detection` flag, `batch_size`, `training_interval_seconds`.
- **`McpConfig`**: `default_cacheable`, `tool_overrides` (per-tool cacheability), `cache_key_fields`.
- **`AcpConfig`**: `default_cacheable`, `agent_overrides`, `cache_key_fields`.

Environment variables in string values are expanded at load time: `${OPENAI_API_KEY}` becomes the actual key.

### server.rs — Route Handlers and Cache Lookup Chain

This is the largest and most important file. It defines `AppState` and all HTTP handlers.

**AppState** holds:
- `config: Config`
- `l1: L1Cache`
- `l2: Arc<L2SemanticCache>`
- `l3: Option<Arc<L3RedisCache>>`
- `upstream: Upstream`
- `embed_tx: mpsc::Sender<EmbedRequest>` — channel to the embedding worker

**Router endpoints:**

| Method | Path | Handler | Purpose |
|--------|------|---------|---------|
| GET | `/health` | `health` | Returns `{"status":"ok"}` |
| POST | `/v1/chat/completions` | `handle_chat_completions` | Main LLM endpoint, handles stream param |
| POST | `/v1/completions` | `handle_completions` | Legacy completions |
| GET | `/v1/cache/stats` | `cache_stats` | L1/L2/L3 entry counts |
| DELETE | `/v1/cache` | `flush_cache` | Clear all cache layers |
| POST | `/mcp/tools/call` | `handle_mcp_call` | MCP tool call caching |
| GET | `/mcp/tools/list` | `handle_mcp_list` | Pass-through, never cached |
| GET | `/mcp/cache/stats` | `mcp_cache_stats` | MCP cache stats |
| POST | `/acp/tasks` | `handle_acp_task` | ACP task caching |
| GET | `/acp/cache/stats` | `acp_cache_stats` | ACP cache stats |

**`handle_cached_request()`** — The core function. It implements the L1 → L2 → L3 → upstream lookup chain:

```rust
async fn handle_cached_request<F>(
    state: &AppState,
    protocol: Protocol,
    body: &Value,
    upstream_fn: F,
) -> Result<(Value, &'static str), ProxyError>
where
    for<'a> F: FnOnce(&'a Upstream, &'a Value) -> Pin<Box<dyn Future<...> + Send + 'a>>
```

The `for<'a>` higher-ranked trait bound is necessary because the closure borrows from `AppState` and must live long enough for the async call. This was a non-trivial lifetime puzzle to solve.

The function:
1. Normalizes and hashes the request body → `cache_key`.
2. Checks L1 (exact hash match).
3. On L1 miss, extracts query text, generates an embedding synchronously via `MockEmbedder`, and searches L2 HNSW for cosine similarity ≥ 0.92.
4. On L2 miss, checks L3 Redis. On L3 hit, promotes the entry to L1 and queues embedding generation for L2.
5. On total miss, calls `upstream_fn` to forward to the provider.
6. **Error check**: calls `is_error_response()` to detect error objects, empty choices, and `content_filter` finish reasons. If the response is an error, it is returned to the client but **never cached** in any layer.
7. On success, creates a `CacheEntry` with cost metadata, inserts into L1, and async-queues for L2 and optionally L3.
8. Returns the response body and a cache status string (`l1-hit`, `l2-hit`, `l3-hit`, or `miss`).

**`handle_streaming_request()`** — For `stream: true` requests. Same cache lookup, but on miss it returns an SSE stream that buffers chunks as they arrive and caches the assembled response after the stream ends. Uses `CachingStreamAdapter` to wrap the upstream stream.

**`extract_query_text()`** — Pulls the user's actual text from the request body. For chat completions, it takes the last message with `role: "user"`. For legacy completions, it takes the `prompt` field. This text is what gets embedded for L2 semantic matching.

**`get_embedding_sync()`** — Directly calls `MockEmbedder::embed()` on the hot path for L2 lookups. This bypasses the async embedding worker channel because the hot path needs the vector immediately (the worker channel is for background L2 insertion on cache misses).

### proxy.rs — Upstream Forwarding

The `Upstream` struct wraps a `reqwest::Client` and handles all provider-specific logic.

**Key methods:**

- **`apply_auth(request_builder)`** — Adds the correct auth header based on provider:
  - OpenAI / VertexAI: `Authorization: Bearer <key>`
  - Anthropic: `x-api-key: <key>` + `anthropic-version: 2023-06-01`
  - Ollama: no auth
  - Custom: `Authorization: Bearer <key>` if key present

- **`resolve_path(original_path)`** — Remaps endpoints for providers with different API shapes. Anthropic maps `/v1/chat/completions` → `/v1/messages`. Others pass through unchanged.

- **`build_url(path)`** — Joins `base_url` + resolved path.

- **`chat_completions(body)`** — Non-streaming POST to `/v1/chat/completions`. Returns `UpstreamResponse` with the JSON body and estimated cost.

- **`chat_completions_stream(body)`** — Streaming POST. Returns a `CollectingStream` that wraps the raw `reqwest` byte stream.

- **`estimate_cost(body, provider)`** — Provider-aware cost estimation. Reads the `usage` field from a response and estimates dollar cost using provider-specific rates: OpenAI ($0.03/$0.06 per 1K tokens), Anthropic ($0.015/$0.075), Vertex AI ($0.0125/$0.0375), Ollama (flat $0.0001 — local GPU time, no billing). Falls back to $0.001 if no usage data.

**`CollectingStream`** — Implements `tokio_stream::Stream`. As SSE chunks arrive, it buffers the `data:` lines. When the stream ends (`data: [DONE]` or connection close), the `assemble()` method reconstructs a complete JSON response:
- Extracts `model` and `id` from the first chunk.
- Joins all `content` deltas into the full response text.
- Estimates completion tokens from character count if not provided.

**`cached_response_to_sse_bytes()`** — Does the reverse: takes a cached non-streaming JSON response and splits it into SSE frames (~20-char chunks), mimicking the streaming format. This allows cache hits to be served as streams when the client requested `stream: true`.

**`ProxyError`** — Error enum with four variants: `Request` (connection failure), `UpstreamError` (non-2xx status), `StreamInterrupted` (SSE stream cut off), `StreamParse` (invalid SSE data).

### normalize.rs — Request Normalization and Hashing

The point of normalization is to make semantically identical requests produce the same cache key, even if they differ in non-semantic fields.

**LLM normalization** keeps: `model`, `messages`, `prompt`, `tools`, `functions`.
Strips: `stream`, `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `n`, and anything else. This means a request with `temperature: 0.7` and one with `temperature: 0.3` share the same cache key — the cached response is valid for both.

**MCP normalization** keeps: `name`, `arguments`. Everything else is stripped.

**ACP normalization** keeps: `agent_id`, `task`, `input`. The `agent_id` is always included per design rule #7 — two agents asking the same question must get separate cache entries because they may have different permissions or context.

The normalized JSON is serialized deterministically (serde_json with sorted keys) and hashed with SHA-256.

### protocol.rs — Protocol Detection

Simple path-based detection:
- `/v1/*` → `Protocol::Llm`
- `/mcp/*` → `Protocol::Mcp`
- `/acp/*` → `Protocol::Acp`

The protocol determines which normalizer and cache key builder to use.

### tests/integration.rs — End-to-End Tests

Spins up mock upstream servers (axum handlers that return canned responses and count calls via `Arc<AtomicU64>`), starts a full Mnemo proxy against them, and runs 10 tests:

1. **health_endpoint** — `/health` returns 200 with `{"status":"ok"}`.
2. **proxies_to_upstream** — First request reaches the mock upstream (call counter = 1).
3. **second_identical_is_cached** — Same request served from L1 without hitting upstream again (call counter stays at 1).
4. **different_requests_both_hit_upstream** — Different prompts each hit upstream (call counter = 2).
5. **cache_stats** — `/v1/cache/stats` returns valid JSON with `l1_entries`, `l2_entries`, `l3_connected`.
6. **cache_flush** — `DELETE /v1/cache` clears L1, next identical request re-hits upstream.
7. **normalization** — Requests differing only in `stream` and `temperature` share a cache key.
8. **content_preservation** — Cached response is byte-identical to the original upstream response.
9. **error_responses_are_never_cached** — Upstream returns an error object; second identical request still hits upstream (error was not cached).
10. **content_filter_responses_are_never_cached** — Upstream returns `finish_reason: "content_filter"`; second request still hits upstream.

---

## Crate: mnemo-cache

All cache layer implementations and the eviction formula.

### entry.rs — CacheEntry

The fundamental unit stored in all cache layers:

```rust
pub struct CacheEntry {
    pub response: serde_json::Value,       // The actual LLM response
    pub generation_cost: f64,               // Estimated dollar cost
    pub hit_count: u64,                     // Times served from cache
    pub created_at: DateTime<Utc>,          // When first cached
    pub last_hit_at: DateTime<Utc>,         // When last served
    pub staleness_risk: f64,                // 0.0 = fresh, 1.0 = stale
    pub embedding_uncertainty: f64,         // 1.0 = uncertain, 0.0 = confident
}
```

`generation_cost` is a first-class field (design rule #5) — it directly influences eviction priority. A $0.50 response is treated fundamentally differently from a $0.001 response.

`staleness_risk` and `embedding_uncertainty` start at 0.0 and 1.0 respectively and are updated asynchronously by background workers. They feed into the eviction priority formula.

### l1.rs — L1 Exact-Match Cache

Uses [moka](https://github.com/moka-rs/moka) (a concurrent cache with W-TinyLFU eviction) paired with a `DashMap` for hit count tracking.

```rust
pub struct L1Cache {
    cache: moka::sync::Cache<String, Arc<CacheEntry>>,
    hits: DashMap<String, u64>,
}
```

- **get()** — O(1) lookup by hash key. If found, atomically increments hit count and calls `record_hit()` on the entry.
- **insert()** — Stores the entry. moka handles eviction when capacity is exceeded (W-TinyLFU balances frequency and recency).
- **invalidate()** — Removes from both the cache and hit map.
- **flush()** — Clears everything.
- **entry_count()** — Note: moka's count is eventually consistent, not exact. This matters for tests.

### l2.rs — L2 Semantic Cache (HNSW Vector Index)

The most complex cache layer. Uses [hnsw_rs](https://crates.io/crates/hnsw_rs) to build a Hierarchical Navigable Small World graph for fast approximate nearest-neighbor search with cosine distance.

```rust
struct IndexState {
    hnsw: Hnsw<'static, f32, DistCosine>,
    id_to_key: Vec<String>,
    key_to_entry: HashMap<String, (usize, Vec<f32>, f64)>,  // (DataId, embedding, eviction_priority)
    max_entries: usize,
    embedding_dim: usize,
}

pub struct L2SemanticCache {
    state: parking_lot::RwLock<IndexState>,
    similarity_threshold: f64,  // default 0.92
}
```

**HNSW parameters:**
- `M = 16` — Number of bidirectional links per node per layer.
- `max_layer = 16` — Maximum number of layers in the hierarchy.
- `ef_construction = 200` — Size of the dynamic candidate list during index building (higher = better recall, slower build).
- `ef_search = 64` — Size of the dynamic candidate list during search (higher = better recall, slower search).

**How search works:**
1. Client provides a 384-dimensional embedding vector.
2. HNSW searches for the single nearest neighbor using cosine distance.
3. If `1.0 - cosine_distance ≥ 0.92` (the similarity threshold), it's a semantic hit.
4. The returned cache key is used to look up the actual `CacheEntry` from L1.

**Why rebuild on insert:** The HNSW index is rebuilt from scratch on every insert or remove. This sounds expensive but is acceptable because:
- Inserts only happen on cache misses (cold path).
- The index lives entirely in memory.
- Rebuilding 100K vectors takes milliseconds.
- The hot path (search) only needs a read lock.

**Cost-aware eviction in L2:** When the cache is full and a new entry arrives, L2 compares the new entry's eviction priority against the lowest-priority existing entry. If the new entry is more valuable, the lowest-priority entry is evicted; otherwise the insert is rejected. Priority is computed from `CacheEntry` metadata (generation cost, hit frequency, recency, staleness risk) using the same formula as L1 eviction. Background workers can call `update_priority()` to keep scores current without re-inserting embeddings.

**Thread safety:** `L2SemanticCache` uses `parking_lot::RwLock`. Searches acquire a read lock (concurrent, fast). Inserts/removes acquire a write lock (exclusive, infrequent). The `unsafe impl Send + Sync` is sound because all access goes through the RwLock.

### l3.rs — L3 Redis Cache

Optional persistent cache using [deadpool-redis](https://crates.io/crates/deadpool-redis) for connection pooling.

```rust
pub struct L3RedisCache {
    pool: Pool,
    ttl: Duration,  // default 7 days
    prefix: String, // "mnemo:"
}
```

- **get()** — `GET mnemo:<key>`, deserialize JSON to `CacheEntry`.
- **insert()** — `SET mnemo:<key> <json> EX <ttl_seconds>`.
- **flush()** — Uses `SCAN` + `DEL` in batches (never `KEYS *`, which blocks Redis on large datasets).
- **Error handling** — All Redis errors are logged at `warn` level and return `Ok(None)` or `Ok(())`. L3 failures never crash the proxy. This is a deliberate design choice: L3 is an optimization, not a requirement.

### eviction.rs — Cost-Aware Eviction Priority

Used by both L1 (moka's W-TinyLFU) and L2 (explicit lowest-priority eviction) to determine which entries are most worth keeping:

```
priority = (generation_cost × hit_frequency × recency_score)
           ─────────────────────────────────────────────────
                staleness_risk × embedding_uncertainty
```

Where:
- `hit_frequency = hit_count + 1` (avoid zero)
- `recency_score = e^(-age_hours / 24)` — 24-hour half-life, so a 24-hour-old entry has ~37% the recency score of a fresh one
- `staleness_risk` and `embedding_uncertainty` are clamped to minimum 0.01 to avoid division by zero

**What this means in practice:**
- A $0.50 response with 100 hits and low staleness → very high priority (protected).
- A $0.001 response with 2 hits and high staleness → very low priority (evicted first).
- Even a rarely-hit response is protected if it was expensive to generate.

---

## Crate: mnemo-intelligence

The "learning" subsystems that run on the cold path. None of these touch the hot path directly — they improve cache quality over time.

### embedder.rs — Embedding Backends and Worker

**`Embedder` trait** — The abstraction over embedding models:
```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;
    fn backend_name(&self) -> &str;
}
```

**`MockEmbedder`** — The default backend. Generates deterministic 384-dimensional vectors from text using FNV-1a hash seeding a xorshift64 PRNG. Outputs are L2-normalized. This isn't a real embedding model — semantically similar texts won't produce similar vectors. But it is:
- Zero-dependency (no ONNX runtime needed)
- Deterministic (same text → same vector, always)
- Fast (pure arithmetic)
- Sufficient for exact-match caching via L2 key lookup

**`OnnxEmbedder`** (behind `onnx` feature flag) — Real embedding model inference using ONNX Runtime:
- Loads a `.onnx` model file and a HuggingFace `tokenizer.json`.
- Tokenizes input with padding and attention masks.
- Runs inference via `ort::Session`.
- Handles both `[batch, hidden]` and `[batch, seq_len, hidden]` output shapes (mean pooling for the latter).
- L2-normalizes output vectors.

**`EmbeddingWorker`** — Background task that processes embedding requests in batches:
```rust
pub struct EmbeddingWorker {
    embedder: Arc<dyn Embedder>,
    batch_size: usize,  // default 32
}
```

It reads from an `mpsc::Receiver<EmbedRequest>`, batches up to `batch_size` requests (blocking on the first, draining without blocking for the rest), runs embedding via `spawn_blocking` (to avoid starving the async runtime with CPU-bound work), and sends `EmbedResult` to an output channel. The output channel is consumed by a background task in `main.rs` that inserts embeddings into L2.

### staleness.rs — Proactive Staleness Detection

Instead of fixed TTLs, Mnemo computes a continuous staleness risk score for each cache entry.

**`StalenessSignals`** — The inputs:
```rust
pub struct StalenessSignals {
    pub age_hours: f64,              // How old the entry is
    pub hours_since_last_hit: f64,   // How long since someone used it
    pub hit_count: u64,              // Total times served
    pub was_corrected: bool,         // External signal: "this response was wrong"
    pub response_drift_score: f64,   // 0-1: how different the upstream response is now
}
```

**`compute_staleness_risk()`** — The formula:
```
temporal_decay  = 1 - e^(-age_hours / 168)       // ~1 week half-life
hit_decay       = if last_hit > 24h ago:
                    min(hours_since_last_hit / 168, 1.0)
                  else: 0.0
correction      = if was_corrected: 0.8 else: 0.0
drift           = response_drift_score × 0.9

risk = clamp(temporal_decay×0.3 + hit_decay×0.2 + correction + drift, 0, 1)
```

**What each component means:**
- **Temporal decay (weight 0.3)** — Even frequently-hit entries slowly become stale. A 1-week-old entry has ~63% temporal decay.
- **Hit decay (weight 0.2)** — If nobody has used an entry in over 24 hours, it starts getting penalized. Entries that are still being actively used are protected.
- **Correction signal (weight 0.8)** — If a user explicitly flags a response as wrong, the staleness risk jumps massively. This is the strongest signal.
- **Response drift (weight 0.9)** — If Mnemo periodically re-asks the upstream and gets a different response, this signals the cached version is outdated.

**`StalenessWorker`** — Background scanner that periodically iterates all cache keys, computes staleness risk, and emits `StalenessUpdate` events when risk exceeds the eviction threshold. Uses a `SignalProvider` trait to query cache layers for entry metadata.

### lora.rs — Domain-Adaptive LoRA Fine-Tuning

This is the most sophisticated component. The idea: as Mnemo sees real traffic, it can fine-tune the embedding model to produce better vectors for your specific domain. "Better" means semantically similar requests in your domain produce more similar embeddings, improving L2 cache hit rates.

**`LoraAdapter`** — A low-rank adapter applied to base embeddings:
```rust
pub struct LoraAdapter {
    pub rank: usize,               // Rank of decomposition (default 8)
    pub dim: usize,                // Embedding dimension (384)
    pub weight_a: Vec<Vec<f32>>,   // rank × dim (down-projection)
    pub weight_b: Vec<Vec<f32>>,   // dim × rank (up-projection)
    pub version: u64,              // Incremented on each training step
    pub trained_on_samples: usize, // Total samples seen
}
```

**`apply(base_embedding, alpha)`** — Produces the adapted embedding:
```
output = base + alpha × (B @ A @ base)
```
Where `A` projects down to rank dimensions and `B` projects back up. The `alpha` parameter (default 0.1) controls how much the adapter shifts the embedding.

**`EwcState`** (Elastic Weight Consolidation) — Prevents catastrophic forgetting:
```rust
pub struct EwcState {
    pub fisher_diagonal: Vec<f32>,  // Fisher information per weight
    pub prev_weights: Vec<f32>,     // Weights from previous training round
    pub lambda: f32,                // Regularization strength (default 100.0)
}
```

EWC adds a penalty when weights drift from their previous values, weighted by how important each weight was (estimated by the Fisher diagonal). This means the adapter can learn new patterns without destroying what it learned from earlier traffic.

- **`penalty()`**: `(lambda / 2) × Σ F_i × (w_i - w*_i)²`
- **`gradient()`**: `lambda × F_i × (w_i - w*_i)`

**`LoraWorker`** — Background training loop:
1. Accumulates `TrainingPair` structs from live traffic (no lock on the hot path — uses an async channel).
2. Wakes up every `training_interval_seconds` (default 300s).
3. Drains the buffer. Needs `min_samples` (default 32) for the first round, `batch_size` for subsequent rounds.
4. For each pair:
   - Computes a target embedding by perturbing the base embedding using a hash of the response text.
   - Computes the current output via `adapter.apply()`.
   - Computes the residual (target - output).
   - Backpropagates through the linear layers to compute gradients for A and B.
5. Averages gradients over the batch.
6. Adds the EWC penalty gradient from the previous round.
7. Performs gradient descent: `w -= lr × (grad + ewc_grad)`.
8. Estimates the Fisher diagonal as the squared gradient magnitudes.
9. Stores the new weights + Fisher as EWC state for the next round.

### cost.rs — Cost Tracker

Currently a placeholder struct. Will track cumulative costs across cache layers and provide cost savings metrics.

---

## Crate: mnemo-mcp

Handles caching for Model Context Protocol tool calls. MCP tools have varying cacheability — `get_weather` with the same arguments should be cached, but `send_email` must never be.

### classifier.rs — Tool Cacheability

```rust
pub enum Cacheability {
    Always,       // Always cache (e.g., read-only lookups)
    Never,        // Never cache (e.g., side-effecting tools)
    Conditional,  // Cache under certain conditions
    Unknown,      // Default: NOT cached (safe fallback, design rule #6)
}
```

`ToolClassifier` supports two classification methods:
1. **Explicit config** — Set per-tool cacheability in `mnemo.yaml`.
2. **Learned classification** — Track observations (was the response consistent for these arguments?). After 10+ observations: >95% consistency → Always, <30% → Never, else Conditional.

### key.rs — MCP Cache Key Building

Builds SHA-256 keys from tool name + arguments:
- Always includes the tool name.
- If `cache_key_fields` is configured, only hashes the specified argument fields. This is critical for tools where some arguments affect the response (e.g., `document_id`) and others don't (e.g., `request_id`).

### schema.rs — Schema Change Detection

`SchemaTracker` maintains a hash of each tool's schema. When a tool's schema changes (new parameters, different types), it flags all cached entries for that tool as invalid. This prevents serving stale responses after a tool upgrade.

---

## Crate: mnemo-acp

Handles caching for Agent Communication Protocol messages. ACP is agent-to-agent communication where different message types have different caching semantics.

### classifier.rs — Message Type Classification

```rust
pub enum AcpMessageType {
    Task,    // Cacheable — "do this thing" requests
    Status,  // Never cached — ephemeral progress updates
    Result,  // Stored as values — computed outputs
    Error,   // Never cached — transient failures
}
```

Only `Task` messages are cached on the request path. This makes sense: a task with the same input should produce the same result, but status updates and errors are inherently ephemeral.

### key.rs — ACP Cache Key Building

Similar to MCP, but with a critical difference: **`agent_id` is always included in the key** (design rule #7). Two different agents asking the same question must get separate cache entries because they may have different permissions, contexts, or expected output formats.

---

## Hot Path vs Cold Path

This separation is the most important architectural decision in Mnemo.

**Hot Path** (< 2ms budget):
- L1 DashMap lookup
- L2 HNSW search (requires synchronous embedding via `MockEmbedder`)
- L1 insertion on cache miss
- Response delivery to client

**Cold Path** (async, no latency constraint):
- Embedding generation for new entries (via EmbeddingWorker channel)
- L2 HNSW insertion (background task)
- L3 Redis operations (all async)
- Staleness risk computation (periodic background scan)
- LoRA adapter training (periodic background training loop)
- Cost tracking and metrics

The hot path never does network I/O (except for upstream forwarding on cache miss), never waits on a lock for more than microseconds, and never runs model inference beyond the MockEmbedder's hash-based arithmetic.

---

## Streaming (SSE) Pipeline

When a client sends `stream: true`, Mnemo must handle Server-Sent Events:

**On cache miss (streaming from upstream):**
```
Client ←── SSE chunks ←── CachingStreamAdapter ←── CollectingStream ←── reqwest stream ←── Upstream
                                    │
                                    └── On stream end: assemble() → CacheEntry → L1 insert
```

1. `CollectingStream` wraps the raw `reqwest` byte stream.
2. As each chunk arrives, it's forwarded to the client AND buffered internally.
3. When the stream ends (`data: [DONE]`), `assemble()` reconstructs a complete non-streaming JSON response from the buffered chunks.
4. `CachingStreamAdapter` takes the assembled response, creates a `CacheEntry`, and inserts it into L1 + queues for L2/L3.

**On cache hit (replaying from cache):**
```
Client ←── SSE chunks ←── cached_response_to_sse_bytes() ←── CacheEntry.response
```

The cached non-streaming JSON is split into ~20-character chunks and formatted as SSE frames, mimicking the streaming format the client expects.

---

## Staleness Detection

Traditional caches use TTLs: "this entry expires in 1 hour." This is crude — some responses are valid for weeks, others become stale in minutes. Mnemo uses four signals to compute a continuous staleness risk:

```
                    ┌─────────────────┐
                    │ Staleness Risk   │
                    │ (0.0 → 1.0)     │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
  ┌─────▼─────┐       ┌─────▼─────┐       ┌─────▼─────┐
  │ Temporal   │       │ Hit       │       │ External  │
  │ Decay      │       │ Pattern   │       │ Signals   │
  │ (30%)      │       │ (20%)     │       │ (80-90%)  │
  └────────────┘       └───────────┘       └───────────┘
                                                 │
                                          ┌──────┴──────┐
                                    ┌─────▼─────┐ ┌─────▼─────┐
                                    │ Correction│ │ Response  │
                                    │ (80%)     │ │ Drift     │
                                    │           │ │ (90%)     │
                                    └───────────┘ └───────────┘
```

The weights mean correction and drift signals dominate. If a user says "that response was wrong" (correction) or the upstream now returns something different (drift), the entry is rapidly marked stale regardless of age or hit count.

---

## LoRA Fine-Tuning Loop

```
Live Traffic
    │
    ▼
┌──────────────────┐     ┌──────────────────┐
│ Training Buffer  │────▶│ LoRA Worker      │
│ (async channel)  │     │ (every 300s)     │
└──────────────────┘     └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │ For each pair:   │
                         │ target = perturb │
                         │ output = B@A@base│
                         │ residual = t - o │
                         │ backprop A, B    │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │ Average grads    │
                         │ + EWC penalty    │
                         │ from prev round  │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │ Gradient descent │
                         │ w -= lr × grad   │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │ Update Fisher    │
                         │ diagonal (EWC)   │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │ Adapter v(n+1)   │
                         │ ready for next   │
                         │ inference call   │
                         └─────────────────┘
```

The EWC mechanism ensures that training round N+1 doesn't destroy what was learned in round N. The Fisher diagonal estimates which weights are "important" for past data, and the penalty term discourages changing them.

---

## Cost-Aware Eviction

When the cache is full, the eviction priority formula ensures expensive, popular, fresh entries survive. This applies to both L1 (moka's W-TinyLFU, 10,000 entries) and L2 (explicit lowest-priority eviction, 100,000 entries):

```
                    priority (higher = keep longer)
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
   Numerator               Denominator              Result
        │                       │                       │
  cost × hits × recency   staleness × uncertainty   keep/evict
```

**Example scenarios:**

| Entry | Cost | Hits | Age | Staleness | Priority | Outcome |
|-------|------|------|-----|-----------|----------|---------|
| GPT-4 complex query | $0.50 | 100 | 2h | 0.05 | Very High | Protected |
| Simple greeting | $0.001 | 5 | 48h | 0.3 | Very Low | Evicted first |
| Corrected response | $0.10 | 50 | 6h | 0.85 | Low | Evicted soon |
| Fresh expensive | $0.30 | 1 | 0.1h | 0.01 | High | Protected by cost |

---

## Multi-Provider Auth

Mnemo adapts its behavior based on the configured provider:

| Provider | Auth Header | Endpoint Remapping | Notes |
|----------|-------------|-------------------|-------|
| OpenAI | `Authorization: Bearer <key>` | None | Default |
| Anthropic | `x-api-key: <key>` + `anthropic-version: 2023-06-01` | `/v1/chat/completions` → `/v1/messages` | Different API shape |
| VertexAI | `Authorization: Bearer <key>` | None | Same as OpenAI-compatible |
| Ollama | None | None | Local, no auth needed |
| Custom | `Authorization: Bearer <key>` (if key present) | None | Bring your own |

---

## Logging

Mnemo has a configurable logging system controlled by environment variables and/or the config file.

### Log Level Resolution (priority order)

1. `MNEMO_LOG` env var — highest priority, intended for quick debugging
2. `RUST_LOG` env var — standard Rust ecosystem convention
3. `logging.level` in `mnemo.yaml` — config file default
4. `"info"` — hardcoded fallback

### Log Format

Two output formats:

- **`pretty`** (default) — Human-readable, colored output for local development:
  ```
  2026-03-28T10:15:32.123Z  INFO mnemo_proxy: cache hit (L1) key="a1b2c3..."
  ```

- **`json`** — Structured JSON, one object per line. Intended for log aggregators (Datadog, Loki, ELK):
  ```json
  {"timestamp":"2026-03-28T10:15:32.123Z","level":"INFO","target":"mnemo_proxy","fields":{"key":"a1b2c3...","message":"cache hit (L1)"}}
  ```

### Logging Flags

| Flag | Default | Effect |
|------|---------|--------|
| `log_requests` | `false` | Logs every incoming request at `debug` level (model, stream, tool name, message type) |
| `log_cache_events` | `true` | Logs cache hits, misses, and inserts at `info` level with cache key and cost |

When `log_cache_events` is enabled, every request logs one of:
- `cache hit (L1)` / `cache hit (L2 semantic)` / `cache hit (L3 Redis)` with cache key
- `cache miss — forwarding to upstream` with cache key
- `cached new response` with cache key and estimated cost
- `upstream returned error response — not caching` when error detected

### Log Levels in Practice

| Level | What you see |
|-------|-------------|
| `error` | Fatal startup failures, unrecoverable upstream errors |
| `warn` | Non-fatal issues: L3 connection failures, stream errors, error responses from upstream, embedding failures |
| `info` | Startup banner, cache events (if enabled), worker lifecycle |
| `debug` | Per-request details (if enabled), L2 insertion, detailed cache key lookups |
| `trace` | Internal details (embedding vectors, HNSW search internals) |

## Error Handling — Never Cache Errors

Mnemo explicitly detects error responses from upstream providers and **never caches them** in any layer (L1, L2, or L3). This prevents cache poisoning from transient failures.

### What counts as an error response

The `is_error_response()` function checks for:

1. **Error object present** — Response body contains an `"error"` field (OpenAI, Anthropic, most providers).
2. **Anthropic error type** — Response body has `"type": "error"`.
3. **Empty choices** — The `"choices"` array exists but is empty (malformed or failed generation).
4. **Content filter** — Any choice has `"finish_reason": "content_filter"` (provider blocked the response).

### Where error checking happens

- **Non-streaming path**: `handle_cached_request()` checks `is_error_response()` after receiving the upstream response. If detected, the response is returned to the client immediately without touching any cache.
- **Streaming path**: `CachingStreamAdapter` checks `is_error_response()` on the assembled response after the stream ends. If detected, the assembled response is discarded (not cached), though the individual SSE chunks were already forwarded to the client.
- **HTTP-level errors**: `Upstream::forward()` returns `ProxyError::UpstreamError` for 4xx/5xx HTTP status codes, which are converted to `502 BAD_GATEWAY` and never reach the caching layer.

### Error flow diagram

```
Upstream Response
       │
       ▼
┌──────────────────┐
│ HTTP status ≥ 400 │──▶ ProxyError::UpstreamError ──▶ 502 to client (never cached)
└────────┬─────────┘
         │ 200
         ▼
┌──────────────────┐
│ is_error_response│──▶ YES ──▶ Return to client (never cached), log warning
│ (body check)     │
└────────┬─────────┘
         │ NO
         ▼
   Cache in L1/L2/L3
```

## Deployment

### Single Docker Image

```dockerfile
# Build stage: rust:1.83-slim
# - Compiles release binary with all optimizations

# Runtime stage: debian:bookworm-slim (~100MB)
# - Copies binary + default config
# - Non-root user (uid 1000)
# - Healthcheck: curl http://localhost:8080/health
# - Port 8080
```

### docker-compose (Mnemo + Redis)

```
┌─────────────────┐     ┌─────────────────┐
│ mnemo:8080      │────▶│ redis:6379      │
│ (proxy)         │     │ (L3 cache)      │
│                 │     │ persistent vol  │
└────────┬────────┘     └─────────────────┘
         │
    ┌────▼────┐
    │ Upstream│
    │ LLM API │
    └─────────┘
```

Redis is optional. Without it, Mnemo runs with L1 + L2 only (in-memory). Data doesn't survive restarts but the cache warms up quickly from live traffic.

---

## Design Rules

These rules are encoded in the implementation and should be preserved in any future changes:

1. **Hot path < 2ms** — No network I/O, no model inference, no blocking locks on the hot path.
2. **Cost is first-class** (rule #5) — Every `CacheEntry` carries `generation_cost`. Eviction, L3 promotion, and metrics all use it.
3. **Unknown tools are non-cacheable** (rule #6) — MCP tools with unknown cacheability default to `Never`. Safe fallback.
4. **ACP keys include agent_id** (rule #7) — Two agents with the same query get separate cache entries.
5. **L3 failures are non-fatal** — All Redis errors are logged and swallowed. The proxy continues with L1 + L2.
6. **Normalization is protocol-aware** — Each protocol (LLM, MCP, ACP) strips different fields and builds keys differently.
7. **Streaming is transparent** — Clients don't know whether a streamed response came from cache or upstream. The format is identical.
8. **Background workers use channels** — The hot path never blocks waiting for embeddings, training, or staleness computation. It sends work to channels and continues.
9. **Never cache errors** — Error responses (error objects, empty choices, content_filter) are detected and passed through without caching. Prevents cache poisoning from transient failures.
10. **Logging is env-controllable** — `MNEMO_LOG` env var overrides config. JSON format available for production log aggregators. Cache events and request details are independently toggleable.
