use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{delete, get, post},
    Router,
};
use bytes::Bytes;
use futures_core::Stream;
use serde_json::{json, Value};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

use crate::config::Config;
use crate::normalize::normalize_and_hash;
use crate::protocol::Protocol;
use crate::proxy::{self, CollectingStream, Upstream};
use mnemo_cache::{l1::L1Cache, l2::L2SemanticCache, l3::L3RedisCache, CacheEntry};
use mnemo_intelligence::{EmbedRequest, Embedder};

/// Shared application state.
pub struct AppState {
    pub config: Config,
    pub l1: L1Cache,
    pub l2: Arc<L2SemanticCache>,
    pub l3: Option<Arc<L3RedisCache>>,
    pub upstream: Upstream,
    /// Send text to the embedding worker for L2 indexing.
    pub embed_tx: mpsc::Sender<EmbedRequest>,
    /// The configured embedder (ONNX or Mock) — shared across insert and lookup paths.
    pub embedder: Arc<dyn Embedder + Send + Sync>,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        // Health
        .route("/health", get(health))
        // LLM endpoints
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/cache/stats", get(cache_stats))
        .route("/v1/cache", delete(cache_flush))
        // MCP endpoints
        .route("/mcp/tools/call", post(mcp_tool_call))
        .route("/mcp/tools/list", get(mcp_tools_list))
        .route("/mcp/cache/stats", get(mcp_cache_stats))
        // ACP endpoints
        .route("/acp/tasks", post(acp_task))
        .route("/acp/cache/stats", get(acp_cache_stats))
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

/// Chat completions handler that detects `stream: true` and routes accordingly.
async fn chat_completions(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if state.config.logging.log_requests {
        tracing::debug!(
            model,
            stream = is_stream,
            "incoming chat completions request"
        );
    }

    if is_stream {
        match handle_streaming_request(&state, &body).await {
            Ok(resp) => resp,
            Err((status, payload)) => {
                tracing::warn!(status = status.as_u16(), "chat completions request failed");
                (status, payload).into_response()
            }
        }
    } else {
        match handle_cached_request(&state, Protocol::Llm, &body, |upstream, b| {
            Box::pin(upstream.chat_completions(b))
        })
        .await
        {
            Ok(resp) => resp.into_response(),
            Err((status, payload)) => {
                tracing::warn!(status = status.as_u16(), "chat completions request failed");
                (status, payload).into_response()
            }
        }
    }
}

/// Handle a streaming chat completions request.
///
/// - On cache hit (L1, L2, or L3): reconstruct an SSE stream from the cached response.
/// - On cache miss: proxy the SSE stream from upstream, collect chunks, cache after.
async fn handle_streaming_request(
    state: &AppState,
    body: &Value,
) -> Result<Response, (StatusCode, Json<Value>)> {
    let cache_key = normalize_and_hash(Protocol::Llm, body);

    // L1 exact lookup
    if let Some(entry) = state.l1.get(&cache_key).await {
        tracing::debug!(key = %cache_key, "L1 cache hit (streaming)");
        return Ok(sse_response_from_cached(&entry.response, "l1-hit"));
    }

    // L2 semantic lookup — extract text for embedding
    let query_text = extract_query_text(body);
    if let Some(ref text) = query_text {
        if let Some(embedding) = get_embedding_sync(state, text).await {
            if let Some((l2_key, similarity)) = state.l2.search(&embedding) {
                if let Some(entry) = state.l1.get(&l2_key).await {
                    tracing::debug!(key = %l2_key, similarity, "L2 cache hit (streaming)");
                    return Ok(sse_response_from_cached(&entry.response, "l2-hit"));
                }
            }
        }
    }

    // L3 Redis lookup
    if let Some(ref l3) = state.l3 {
        match l3.get(&cache_key).await {
            Ok(Some(entry)) => {
                tracing::debug!(key = %cache_key, "L3 cache hit (streaming) — promoting to L1");
                let response = entry.response.clone();
                // Promote to L1
                state.l1.insert(cache_key.clone(), entry).await;
                // Queue embedding for L2
                if let Some(ref text) = query_text {
                    let _ = state.embed_tx.try_send(EmbedRequest {
                        cache_key: cache_key.clone(),
                        text: text.clone(),
                    });
                }
                return Ok(sse_response_from_cached(&response, "l3-hit"));
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(error = %e, "L3 lookup failed (non-fatal, streaming)");
            }
        }
    }

    // Cache miss — stream from upstream
    tracing::debug!(key = %cache_key, "cache miss (streaming), forwarding to upstream");

    let collecting_stream = state
        .upstream
        .chat_completions_stream(body)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "upstream streaming error");
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": e.to_string() })),
            )
        })?;

    let adapter = CachingStreamAdapter {
        inner: collecting_stream,
        cache_key,
        query_text,
        state_l1: state.l1.clone(),
        state_l3: state.l3.clone(),
        l3_min_cost: state.config.cache.l3_min_cost_threshold,
        embed_tx: state.embed_tx.clone(),
    };

    let body = Body::from_stream(adapter);
    let response = Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("x-mnemo-cache", "miss")
        .body(body)
        .unwrap_or_else(|_| {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::empty())
                .expect("fallback response")
        });

    Ok(response)
}

fn sse_response_from_cached(response: &Value, cache_header: &str) -> Response {
    let sse_chunks = proxy::cached_response_to_sse_bytes(response);
    let stream = tokio_stream::iter(sse_chunks.into_iter().map(Ok::<Bytes, std::io::Error>));
    let body = Body::from_stream(stream);
    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("x-mnemo-cache", cache_header)
        .body(body)
        .unwrap_or_else(|_| {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::empty())
                .expect("fallback response")
        })
}

/// Stream adapter that pipes SSE bytes through while collecting for post-stream caching.
struct CachingStreamAdapter {
    inner: CollectingStream,
    cache_key: String,
    query_text: Option<String>,
    state_l1: L1Cache,
    state_l3: Option<Arc<L3RedisCache>>,
    l3_min_cost: f64,
    embed_tx: mpsc::Sender<EmbedRequest>,
}

impl Stream for CachingStreamAdapter {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let inner = Pin::new(&mut self.inner);
        match inner.poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => Poll::Ready(Some(Ok(bytes))),
            Poll::Ready(Some(Err(e))) => {
                tracing::warn!(error = %e, "SSE stream error, partial response will not be cached");
                Poll::Ready(Some(Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    e.to_string(),
                ))))
            }
            Poll::Ready(None) => {
                // Stream finished. Assemble and cache — but only if not an error.
                if let Some(assembled) = self.inner.assemble() {
                    if is_error_response(&assembled.assembled) {
                        tracing::warn!(key = %self.cache_key, "streaming response was an error — not caching");
                    } else {
                        let l1 = self.state_l1.clone();
                        let l3 = self.state_l3.clone();
                        let key = self.cache_key.clone();
                        let entry = CacheEntry::new(assembled.assembled, assembled.generation_cost);
                        let l3_min_cost = self.l3_min_cost;

                        // Queue embedding generation for L2
                        if let Some(ref text) = self.query_text {
                            let _ = self.embed_tx.try_send(EmbedRequest {
                                cache_key: key.clone(),
                                text: text.clone(),
                            });
                        }

                        tokio::spawn(async move {
                            // Store in L1
                            l1.insert(key.clone(), entry.clone()).await;
                            // Store in L3 if cost exceeds threshold
                            if entry.generation_cost >= l3_min_cost {
                                if let Some(ref l3) = l3 {
                                    if let Err(e) = l3.insert(&key, &entry).await {
                                        tracing::warn!(error = %e, "L3 insert failed (non-fatal, streaming)");
                                    }
                                }
                            }
                            tracing::debug!("cached assembled streaming response");
                        });
                    }
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Result<Response, (StatusCode, Json<Value>)> {
    if state.config.logging.log_requests {
        tracing::debug!("incoming legacy completions request");
    }
    handle_cached_request(&state, Protocol::Llm, &body, |upstream, b| {
        Box::pin(upstream.completions(b))
    })
    .await
}

/// Core cache-or-forward logic with L1 → L2 → L3 → upstream lookup chain.
/// Returns a Response (not Json) so we can set the x-mnemo-cache header.
async fn handle_cached_request<F>(
    state: &AppState,
    protocol: Protocol,
    body: &Value,
    forward_fn: F,
) -> Result<Response, (StatusCode, Json<Value>)>
where
    for<'a> F: FnOnce(
        &'a Upstream,
        &'a Value,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<crate::proxy::UpstreamResponse, crate::proxy::ProxyError>,
                > + Send
                + 'a,
        >,
    >,
{
    let cache_key = normalize_and_hash(protocol, body);
    let log_cache = state.config.logging.log_cache_events;

    // ── L1 exact lookup ──
    if let Some(entry) = state.l1.get(&cache_key).await {
        if log_cache {
            tracing::info!(key = %cache_key, "cache hit (L1)");
        }
        tracing::debug!(key = %cache_key, "L1 cache hit");
        return Ok(json_response_with_cache_header(
            entry.response.clone(),
            "l1-hit",
        ));
    }

    // ── L2 semantic lookup ──
    let query_text = extract_query_text(body);
    if let Some(ref text) = query_text {
        if let Some(embedding) = get_embedding_sync(state, text).await {
            if let Some((l2_key, similarity)) = state.l2.search(&embedding) {
                // Found a semantically similar entry — look it up in L1
                if let Some(entry) = state.l1.get(&l2_key).await {
                    if log_cache {
                        tracing::info!(key = %l2_key, similarity = format!("{:.4}", similarity), "cache hit (L2 semantic)");
                    }
                    tracing::debug!(key = %l2_key, similarity, "L2 cache hit");
                    return Ok(json_response_with_cache_header(
                        entry.response.clone(),
                        "l2-hit",
                    ));
                }
            }
        }
    }

    // ── L3 Redis lookup ──
    if let Some(ref l3) = state.l3 {
        match l3.get(&cache_key).await {
            Ok(Some(entry)) => {
                if log_cache {
                    tracing::info!(key = %cache_key, "cache hit (L3 Redis) — promoting to L1");
                }
                tracing::debug!(key = %cache_key, "L3 cache hit — promoting to L1");
                let response = entry.response.clone();
                // Promote to L1
                state.l1.insert(cache_key.clone(), entry).await;
                // Queue embedding for L2
                if let Some(ref text) = query_text {
                    let _ = state.embed_tx.try_send(EmbedRequest {
                        cache_key: cache_key.clone(),
                        text: text.clone(),
                    });
                }
                return Ok(json_response_with_cache_header(response, "l3-hit"));
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(error = %e, "L3 lookup failed (non-fatal)");
            }
        }
    }

    // ── Cache miss — forward to upstream ──
    if log_cache {
        tracing::info!(key = %cache_key, "cache miss — forwarding to upstream");
    }
    tracing::debug!(key = %cache_key, "full cache miss, forwarding to upstream");

    let upstream_resp = forward_fn(&state.upstream, body).await.map_err(|e| {
        tracing::error!(error = %e, "upstream request failed");
        (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": e.to_string() })),
        )
    })?;

    // ── Never cache error responses ──
    if is_error_response(&upstream_resp.body) {
        tracing::warn!(
            key = %cache_key,
            "upstream returned error response — not caching"
        );
        return Ok(json_response_with_cache_header(upstream_resp.body, "miss"));
    }

    // Store in L1
    let entry = CacheEntry::new(upstream_resp.body.clone(), upstream_resp.generation_cost);
    state.l1.insert(cache_key.clone(), entry.clone()).await;

    // Queue embedding generation for L2
    if let Some(ref text) = query_text {
        let _ = state.embed_tx.try_send(EmbedRequest {
            cache_key: cache_key.clone(),
            text: text.clone(),
        });
    }

    // Store in L3 if cost exceeds threshold (async, non-blocking)
    if upstream_resp.generation_cost >= state.config.cache.l3_min_cost_threshold {
        if let Some(ref l3) = state.l3 {
            let l3 = Arc::clone(l3);
            let l3_key = cache_key.clone();
            let l3_entry = entry;
            tokio::spawn(async move {
                if let Err(e) = l3.insert(&l3_key, &l3_entry).await {
                    tracing::warn!(error = %e, "L3 insert failed (non-fatal)");
                }
            });
        }
    }

    if log_cache {
        tracing::info!(
            key = %cache_key,
            cost = format!("${:.6}", upstream_resp.generation_cost),
            "cached new response"
        );
    }

    Ok(json_response_with_cache_header(upstream_resp.body, "miss"))
}

/// Build a JSON response with the x-mnemo-cache header.
fn json_response_with_cache_header(body: Value, cache_status: &str) -> Response {
    Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .header("x-mnemo-cache", cache_status)
        .body(Body::from(serde_json::to_string(&body).unwrap_or_default()))
        .unwrap_or_else(|_| Json(body).into_response())
}

/// Check if an upstream response is an error that should NOT be cached.
/// Detects: OpenAI/Anthropic error objects, empty choices, and missing content.
fn is_error_response(body: &Value) -> bool {
    // Explicit error field (OpenAI, Anthropic, and most providers)
    if body.get("error").is_some() {
        return true;
    }
    // Anthropic-style "type": "error"
    if body.get("type").and_then(|v| v.as_str()) == Some("error") {
        return true;
    }
    // Empty choices array (malformed or failed generation)
    if let Some(choices) = body.get("choices").and_then(|c| c.as_array()) {
        if choices.is_empty() {
            return true;
        }
        // Check for content_filter or error finish reasons
        for choice in choices {
            if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                if reason == "content_filter" {
                    return true;
                }
            }
        }
    }
    false
}

/// Extract the user's query text from the request body for embedding.
fn extract_query_text(body: &Value) -> Option<String> {
    // For chat completions: use the last user message
    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages.iter().rev() {
            if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
                if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                    return Some(content.to_string());
                }
            }
        }
    }
    // For legacy completions: use the prompt
    if let Some(prompt) = body.get("prompt").and_then(|p| p.as_str()) {
        return Some(prompt.to_string());
    }
    // For ACP: use the task + input
    if let Some(task) = body.get("task").and_then(|t| t.as_str()) {
        let input = body.get("input").and_then(|i| i.as_str()).unwrap_or("");
        return Some(format!("{} {}", task, input));
    }
    None
}

/// Get an embedding synchronously using the configured embedder.
/// Returns None if the embedder fails.
async fn get_embedding_sync(state: &AppState, text: &str) -> Option<Vec<f32>> {
    match state.embedder.embed(text) {
        Ok(embedding) => Some(embedding),
        Err(e) => {
            tracing::warn!(error = %e, "embedding failed for L2 lookup");
            None
        }
    }
}

async fn cache_stats(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "l1_entries": state.l1.entry_count(),
        "l2_entries": state.l2.len(),
        "l3_connected": state.l3.is_some(),
    }))
}

async fn cache_flush(State(state): State<Arc<AppState>>) -> Json<Value> {
    state.l1.flush().await;
    state.l2.flush();
    // L3 flush
    if let Some(ref l3) = state.l3 {
        if let Err(e) = l3.flush().await {
            tracing::warn!(error = %e, "L3 flush failed");
        }
    }
    Json(json!({ "status": "flushed" }))
}

// --- MCP ---

async fn mcp_tool_call(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Result<Response, (StatusCode, Json<Value>)> {
    let tool_name = body.get("name").and_then(|v| v.as_str()).unwrap_or("");
    if state.config.logging.log_requests {
        tracing::debug!(tool = tool_name, "incoming MCP tool call");
    }

    // Design rule #6: unknown tools default to non-cacheable.
    // If no mcp config exists at all, ALL tool calls bypass the cache.
    let mcp_config = match state.config.mcp {
        Some(ref cfg) => cfg,
        None => {
            let resp = state
                .upstream
                .forward_to("/mcp/tools/call", &body)
                .await
                .map_err(|e| {
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(json!({ "error": e.to_string() })),
                    )
                })?;
            return Ok(json_response_with_cache_header(resp.body, "bypass"));
        }
    };

    if let Some(tool_config) = mcp_config.tools.get(tool_name) {
        if !tool_config.cacheable {
            let resp = state
                .upstream
                .forward_to("/mcp/tools/call", &body)
                .await
                .map_err(|e| {
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(json!({ "error": e.to_string() })),
                    )
                })?;
            return Ok(json_response_with_cache_header(resp.body, "bypass"));
        }

        // Use cache_key_fields for MCP cache key if configured
        if !tool_config.cache_key_fields.is_empty() {
            let arguments = body.get("arguments").cloned().unwrap_or(json!({}));
            let cache_key =
                mnemo_mcp::key::build_mcp_key(tool_name, &arguments, &tool_config.cache_key_fields);
            return handle_mcp_with_key(state, &body, &cache_key).await;
        }
    } else {
        // Unknown tool — bypass
        let resp = state
            .upstream
            .forward_to("/mcp/tools/call", &body)
            .await
            .map_err(|e| {
                (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({ "error": e.to_string() })),
                )
            })?;
        return Ok(json_response_with_cache_header(resp.body, "bypass"));
    }

    handle_cached_request(&state, Protocol::Mcp, &body, |upstream, b| {
        Box::pin(upstream.forward_to("/mcp/tools/call", b))
    })
    .await
}

/// Handle an MCP tool call with a pre-computed cache key from cache_key_fields.
async fn handle_mcp_with_key(
    state: Arc<AppState>,
    body: &Value,
    cache_key: &str,
) -> Result<Response, (StatusCode, Json<Value>)> {
    let log_cache = state.config.logging.log_cache_events;

    // L1 lookup
    if let Some(entry) = state.l1.get(cache_key).await {
        if log_cache {
            tracing::info!(key = %cache_key, "MCP cache hit (L1)");
        }
        return Ok(json_response_with_cache_header(
            entry.response.clone(),
            "l1-hit",
        ));
    }

    // Cache miss — forward
    if log_cache {
        tracing::info!(key = %cache_key, "MCP cache miss — forwarding");
    }

    let resp = state
        .upstream
        .forward_to("/mcp/tools/call", body)
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": e.to_string() })),
            )
        })?;

    if is_error_response(&resp.body) {
        return Ok(json_response_with_cache_header(resp.body, "miss"));
    }

    let entry = CacheEntry::new(resp.body.clone(), resp.generation_cost);
    state.l1.insert(cache_key.to_string(), entry).await;

    Ok(json_response_with_cache_header(resp.body, "miss"))
}

async fn mcp_tools_list(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    // MCP tools/list is a GET — never cached, just pass through.
    let resp = state
        .upstream
        .forward_get("/mcp/tools/list")
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": e.to_string() })),
            )
        })?;
    Ok(Json(resp.body))
}

async fn mcp_cache_stats(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "l1_entries": state.l1.entry_count(),
        "l2_entries": state.l2.len(),
    }))
}

// --- ACP ---

async fn acp_task(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Result<Response, (StatusCode, Json<Value>)> {
    let msg_type = mnemo_acp::classifier::AcpMessageType::from_body(&body);
    if state.config.logging.log_requests {
        tracing::debug!(msg_type = ?msg_type, "incoming ACP task");
    }
    if !msg_type.is_cacheable() {
        let resp = state
            .upstream
            .forward_to("/acp/tasks", &body)
            .await
            .map_err(|e| {
                (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({ "error": e.to_string() })),
                )
            })?;
        return Ok(json_response_with_cache_header(resp.body, "bypass"));
    }

    // Check for per-agent cache_key_fields configuration
    let agent_id = body.get("agent_id").and_then(|v| v.as_str()).unwrap_or("");
    if let Some(ref acp_config) = state.config.acp {
        if let Some(agent_config) = acp_config.agents.get(agent_id) {
            if !agent_config.cache_key_fields.is_empty() {
                let task = body.get("task").and_then(|v| v.as_str()).unwrap_or("");
                let input = body.get("input").cloned().unwrap_or(json!({}));
                let cache_key = mnemo_acp::key::build_acp_key(
                    agent_id,
                    task,
                    &input,
                    &agent_config.cache_key_fields,
                );
                return handle_acp_with_key(&state, &body, &cache_key).await;
            }
        }
    }

    handle_cached_request(&state, Protocol::Acp, &body, |upstream, b| {
        Box::pin(upstream.forward_to("/acp/tasks", b))
    })
    .await
}

/// Handle an ACP task with a pre-computed cache key from cache_key_fields.
async fn handle_acp_with_key(
    state: &AppState,
    body: &Value,
    cache_key: &str,
) -> Result<Response, (StatusCode, Json<Value>)> {
    let log_cache = state.config.logging.log_cache_events;

    // L1 lookup
    if let Some(entry) = state.l1.get(cache_key).await {
        if log_cache {
            tracing::info!(key = %cache_key, "ACP cache hit (L1)");
        }
        return Ok(json_response_with_cache_header(
            entry.response.clone(),
            "l1-hit",
        ));
    }

    // Cache miss — forward
    if log_cache {
        tracing::info!(key = %cache_key, "ACP cache miss — forwarding");
    }

    let resp = state
        .upstream
        .forward_to("/acp/tasks", body)
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": e.to_string() })),
            )
        })?;

    if is_error_response(&resp.body) {
        return Ok(json_response_with_cache_header(resp.body, "miss"));
    }

    let entry = CacheEntry::new(resp.body.clone(), resp.generation_cost);
    state.l1.insert(cache_key.to_string(), entry).await;

    Ok(json_response_with_cache_header(resp.body, "miss"))
}

async fn acp_cache_stats(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "l1_entries": state.l1.entry_count(),
        "l2_entries": state.l2.len(),
    }))
}
