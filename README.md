# Mnemo

A self-hosted LLM caching proxy that learns your domain, detects staleness, and makes cost-aware eviction decisions -- getting cheaper and more accurate the longer it runs.

**One line change. Nothing else breaks.**

```typescript
const client = new OpenAI({
  apiKey: "...",
  baseURL: "http://localhost:8080/v1"  // point at Mnemo, not OpenAI
});
```

## What it does

Every LLM request passes through Mnemo. If a similar question was asked before -- return the cached answer instantly. If not -- forward to the real LLM, cache the response, move on.

Works with: OpenAI, Anthropic, Vertex AI, Ollama, any OpenAI-compatible provider.

## What makes it different

**1. Domain-adaptive embeddings** -- The cache gets smarter about your specific domain over time. A medical app's cache learns medical language. A coding app's cache learns code semantics. Implemented via lightweight LoRA adapter fine-tuning on live traffic, silently in the background.

**2. Learning-based staleness detection** -- Not TTLs. Not manual versioning. Mnemo watches behavioral signals (corrections, response drift, query patterns) and evicts proactively when cached answers go stale.

**3. Cost-aware eviction** -- A $2 response gets protected. A $0.001 response gets sacrificed first. The eviction formula factors in actual generation cost, hit frequency, recency, staleness risk, and embedding uncertainty.

## What it is NOT

Not an AI judge on every request. Not a model router. Not a RAG pipeline. The hot path is pure Rust -- under 2ms overhead, always.

## Quick start

### With Docker Compose (recommended)

```bash
# Clone and configure
git clone https://github.com/your-org/mnemo.git
cd mnemo
export OPENAI_API_KEY=sk-...

# Start Mnemo + Redis
docker compose up -d

# Test it
curl http://localhost:8080/health
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}'
```

### From source

```bash
# Build
cargo build --release

# Configure
export OPENAI_API_KEY=sk-...

# Run
./target/release/mnemo
# Listening on 0.0.0.0:8080
```

## Configuration

Edit `mnemo.yaml`:

```yaml
upstream:
  provider: openai           # openai | anthropic | vertexai | ollama | custom
  base_url: https://api.openai.com
  api_key: ${OPENAI_API_KEY}

bind: "0.0.0.0:8080"

cache:
  l1_max_entries: 10000      # In-memory exact match cache
  l2_max_entries: 100000     # In-memory semantic similarity cache
  l3_redis_url: redis://localhost:6379  # Persistent cache (optional)
  l3_min_cost_threshold: 0.01          # Only persist responses above this cost

logging:
  level: info                # error | warn | info | debug | trace
  format: pretty             # pretty | json (structured)
  log_requests: false        # log individual request details at debug level
  log_cache_events: true     # log cache hit/miss/insert events at info level

intelligence:
  embedding_model: bge-small-en
  lora_training_enabled: true
  staleness_detection_enabled: true
  training_batch_size: 32
  training_interval_seconds: 300
```

### Logging

Log level can be controlled three ways (in priority order):

1. `MNEMO_LOG=debug` — env var, takes highest priority
2. `RUST_LOG=mnemo_proxy=debug` — standard Rust env filter
3. `logging.level` in `mnemo.yaml` — config file fallback

Set `format: json` for structured JSON log output (useful for log aggregators like Datadog, Loki, etc).

### Error handling

Mnemo never caches error responses. If the upstream returns an error object, an empty choices array, or a `content_filter` finish reason, the response is passed through to the client but not stored in any cache layer. This prevents poisoning the cache with transient failures.

## Architecture

```
Request → L1 exact hash (DashMap, ~0.05ms)
        → L2 HNSW semantic search (~0.5ms)
        → L3 Redis persistent lookup (~2ms)
        → Upstream LLM (only on full miss)
        → Cache response in L1 + queue for L2/L3
```

### Cache layers

| Layer | Technology | What | Latency |
|-------|-----------|------|---------|
| L1 | DashMap + moka W-TinyLFU | Exact hash match | ~0.05ms |
| L2 | HNSW vector index | Semantic similarity | ~0.5ms |
| L3 | Redis | Persistent cross-restart | ~2ms |

### Intelligence workers (all async, never on hot path)

- **Embedding worker** -- generates embeddings for L2 cache entries
- **LoRA training worker** -- fine-tunes domain-specific embedding adapter with EWC regularization
- **Staleness worker** -- proactively scores and evicts stale entries
- **Cost tracker** -- logs generation costs for eviction decisions

### Eviction formula

```
priority = (token_cost x hit_frequency x recency_score) / (staleness_risk x embedding_uncertainty)
```

Higher priority = more worth keeping. Expensive, frequently-hit, fresh responses are protected. Cheap, stale, uncertain responses are evicted first.

## API

### LLM (OpenAI-compatible)
```
POST   /v1/chat/completions     -- cached LLM call (streaming supported)
POST   /v1/completions          -- legacy completions
GET    /v1/cache/stats           -- hit rate and entry counts
DELETE /v1/cache                 -- manual flush
GET    /health                   -- liveness probe
```

### MCP (Model Context Protocol)
```
POST   /mcp/tools/call           -- cached tool call
GET    /mcp/tools/list           -- pass-through (never cached)
GET    /mcp/cache/stats          -- MCP cache stats
```

### ACP (Agent Communication Protocol)
```
POST   /acp/tasks                -- cached agent task
GET    /acp/cache/stats          -- ACP cache stats
```

## Multi-provider support

| Provider | Auth | Endpoint mapping | Cost estimation |
|----------|------|-----------------|-----------------|
| OpenAI | `Authorization: Bearer` | Standard | $0.03/1K prompt, $0.06/1K completion |
| Anthropic | `x-api-key` + `anthropic-version` | `/v1/chat/completions` -> `/v1/messages` | $0.015/1K prompt, $0.075/1K completion |
| Vertex AI | `Authorization: Bearer` (OAuth2 token) | Standard | $0.0125/1K prompt, $0.0375/1K completion |
| Ollama | None (local) | Standard | Flat $0.0001 (local GPU time) |

### Using with Ollama

```yaml
# mnemo.yaml
upstream:
  provider: ollama
  base_url: http://localhost:11434
```

```bash
# Start Ollama, then Mnemo
ollama serve &
cargo run --release

# Use it
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3","messages":[{"role":"user","content":"Hello"}]}'
```

Ollama uses OpenAI-compatible endpoints, so no special configuration is needed beyond pointing `base_url` at your Ollama instance. No API key required.

## Project structure

```
mnemo/
├── crates/
│   ├── mnemo-proxy/          -- Main binary: axum server, proxy, routing
│   ├── mnemo-cache/          -- L1 (DashMap), L2 (HNSW), L3 (Redis), eviction
│   ├── mnemo-intelligence/   -- Embeddings, LoRA training, staleness detection
│   ├── mnemo-mcp/            -- MCP tool caching + auto-classification
│   └── mnemo-acp/            -- ACP agent task caching
├── Dockerfile
├── docker-compose.yml
└── mnemo.yaml
```

## Development

```bash
# Run tests (70 tests)
cargo test

# Run with debug logging
MNEMO_LOG=debug cargo run

# Build release binary
cargo build --release
```

## License

MIT
