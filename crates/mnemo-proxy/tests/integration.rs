//! End-to-end integration tests for the Mnemo proxy.
//!
//! These tests spin up:
//! 1. A mock upstream LLM server (fake OpenAI)
//! 2. The actual Mnemo proxy pointed at the mock
//! 3. A reqwest client to exercise the proxy
//!
//! No real LLM, no Redis — everything is in-memory and self-contained.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::{routing::post, Json, Router};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use mnemo_cache::l1::L1Cache;
use mnemo_cache::l2::L2SemanticCache;
use mnemo_intelligence::{EmbedRequest, EmbedResult, EmbeddingWorker, MockEmbedder, EMBEDDING_DIM};
use mnemo_proxy::config::{
    CacheConfig, Config, IntelligenceConfig, LoggingConfig, Provider, UpstreamConfig,
};
use mnemo_proxy::proxy::Upstream;
use mnemo_proxy::server::{self, AppState};

/// Start a mock upstream that counts calls and returns a canned response.
async fn start_mock_upstream() -> (String, Arc<AtomicU64>) {
    let call_count = Arc::new(AtomicU64::new(0));
    let counter = call_count.clone();

    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(body): Json<Value>| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                let model = body
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("gpt-4");
                Json(json!({
                    "id": "chatcmpl-test123",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "The capital of France is Paris."
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 8,
                        "total_tokens": 18
                    }
                }))
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });

    (format!("http://{}", addr), call_count)
}

/// Create a fully wired Mnemo proxy pointing at the given upstream URL.
/// Returns the proxy's base URL.
async fn start_mnemo_proxy(upstream_url: &str) -> String {
    let config = Config {
        upstream: UpstreamConfig {
            provider: Provider::OpenAI,
            base_url: upstream_url.to_string(),
            api_key: Some("test-key".to_string()),
        },
        cache: CacheConfig::default(),
        intelligence: IntelligenceConfig::default(),
        logging: LoggingConfig::default(),
        mcp: None,
        acp: None,
        bind: "0.0.0.0:0".to_string(), // unused — we bind manually
    };

    let l1 = L1Cache::new(config.cache.l1_max_entries);
    let l2 = Arc::new(L2SemanticCache::with_defaults(1000, EMBEDDING_DIM));

    let (embed_tx, embed_rx) = mpsc::channel::<EmbedRequest>(1024);
    let (result_tx, mut result_rx) = mpsc::channel::<EmbedResult>(1024);

    let embedder = Arc::new(MockEmbedder::new());
    let worker = EmbeddingWorker::new(embedder);
    let _handle = worker.spawn(embed_rx, result_tx);

    // Background L2 inserter
    let l2_bg = Arc::clone(&l2);
    tokio::spawn(async move {
        while let Some(result) = result_rx.recv().await {
            l2_bg.insert(result.cache_key, result.embedding, 1.0);
        }
    });

    let upstream = Upstream::new(config.upstream.clone());

    let state = Arc::new(AppState {
        config,
        l1,
        l2,
        l3: None,
        upstream,
        embed_tx,
        embedder: Arc::new(MockEmbedder::new()),
    });

    let app = server::router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });

    format!("http://{}", addr)
}

/// Start a mock upstream that returns an error response body (with 200 status
/// but an `error` field, simulating provider-level errors like rate limits).
async fn start_error_upstream() -> (String, Arc<AtomicU64>) {
    let call_count = Arc::new(AtomicU64::new(0));
    let counter = call_count.clone();

    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(_body): Json<Value>| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Json(json!({
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                }))
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });

    (format!("http://{}", addr), call_count)
}

fn chat_request(content: &str) -> Value {
    json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": content}]
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let (upstream_url, _) = start_mock_upstream().await;
    let proxy_url = start_mnemo_proxy(&upstream_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/health", proxy_url))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn chat_completions_proxies_to_upstream() {
    let (upstream_url, call_count) = start_mock_upstream().await;
    let proxy_url = start_mnemo_proxy(&upstream_url).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&chat_request("What is the capital of France?"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();

    // Verify the response contains the expected content
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.contains("Paris"));

    // Upstream was called exactly once
    assert_eq!(call_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn second_identical_request_is_served_from_cache() {
    let (upstream_url, call_count) = start_mock_upstream().await;
    let proxy_url = start_mnemo_proxy(&upstream_url).await;

    let client = reqwest::Client::new();
    let req_body = chat_request("What is the capital of France?");

    // First request — cache miss, hits upstream
    let resp1 = client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp1.status(), 200);
    assert_eq!(call_count.load(Ordering::SeqCst), 1);

    // Second identical request — cache hit, should NOT hit upstream
    let resp2 = client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp2.status(), 200);

    // Upstream should still only have been called once
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "second request should be served from L1 cache"
    );

    // Both responses should have identical content
    let body1: Value = resp1.json().await.unwrap();
    let body2: Value = resp2.json().await.unwrap();
    assert_eq!(
        body1["choices"][0]["message"]["content"],
        body2["choices"][0]["message"]["content"]
    );
}

#[tokio::test]
async fn different_requests_both_hit_upstream() {
    let (upstream_url, call_count) = start_mock_upstream().await;
    let proxy_url = start_mnemo_proxy(&upstream_url).await;

    let client = reqwest::Client::new();

    client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&chat_request("What is 2+2?"))
        .send()
        .await
        .unwrap();

    client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&chat_request("What is 3+3?"))
        .send()
        .await
        .unwrap();

    assert_eq!(
        call_count.load(Ordering::SeqCst),
        2,
        "different prompts should both hit upstream"
    );
}

#[tokio::test]
async fn cache_stats_endpoint_returns_valid_json() {
    let (upstream_url, _) = start_mock_upstream().await;
    let proxy_url = start_mnemo_proxy(&upstream_url).await;

    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/v1/cache/stats", proxy_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let stats: Value = resp.json().await.unwrap();

    // Verify the stats response has the expected fields
    assert!(stats.get("l1_entries").is_some());
    assert!(stats.get("l2_entries").is_some());
    assert!(stats.get("l3_connected").is_some());
    assert_eq!(stats["l3_connected"], false);
}

#[tokio::test]
async fn cache_flush_clears_entries() {
    let (upstream_url, call_count) = start_mock_upstream().await;
    let proxy_url = start_mnemo_proxy(&upstream_url).await;

    let client = reqwest::Client::new();
    let req_body = chat_request("hello");

    // Populate cache
    client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(call_count.load(Ordering::SeqCst), 1);

    // Flush
    let resp = client
        .delete(format!("{}/v1/cache", proxy_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "flushed");

    // Same request after flush should hit upstream again
    client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        2,
        "after flush, should hit upstream again"
    );
}

#[tokio::test]
async fn normalization_strips_stream_and_temperature() {
    let (upstream_url, call_count) = start_mock_upstream().await;
    let proxy_url = start_mnemo_proxy(&upstream_url).await;

    let client = reqwest::Client::new();

    // Request with stream:false and temperature:0.7
    let req1 = json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "test"}],
        "stream": false,
        "temperature": 0.7
    });
    client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&req1)
        .send()
        .await
        .unwrap();
    assert_eq!(call_count.load(Ordering::SeqCst), 1);

    // Same semantic request with different stream/temperature values
    let req2 = json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "test"}],
        "stream": false,
        "temperature": 0.0,
        "max_tokens": 100
    });
    client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&req2)
        .send()
        .await
        .unwrap();

    // Should be a cache hit — normalization strips stream/temperature/max_tokens
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "normalized requests should share cache key"
    );
}

#[tokio::test]
async fn cached_response_preserves_content() {
    let (upstream_url, call_count) = start_mock_upstream().await;
    let proxy_url = start_mnemo_proxy(&upstream_url).await;

    let client = reqwest::Client::new();
    let req = chat_request("cost test");

    // First request — from upstream
    let resp1: Value = client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&req)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    // Second request — from cache
    let resp2: Value = client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&req)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    // Content should be identical
    assert_eq!(
        resp1["choices"][0]["message"]["content"],
        resp2["choices"][0]["message"]["content"]
    );
    // Model should be preserved
    assert_eq!(resp1["model"], resp2["model"]);
    // Upstream called only once
    assert_eq!(call_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn error_responses_are_never_cached() {
    let (upstream_url, call_count) = start_error_upstream().await;
    let proxy_url = start_mnemo_proxy(&upstream_url).await;

    let client = reqwest::Client::new();
    let req = chat_request("trigger an error");

    // First request — upstream returns error response
    let resp1: Value = client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&req)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        resp1.get("error").is_some(),
        "response should contain error"
    );
    assert_eq!(call_count.load(Ordering::SeqCst), 1);

    // Second identical request — should NOT be served from cache, must hit upstream again
    let resp2: Value = client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&req)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(resp2.get("error").is_some());

    // Upstream must have been called twice — error was not cached
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        2,
        "error responses must not be cached — upstream should be called every time"
    );
}

#[tokio::test]
async fn content_filter_responses_are_never_cached() {
    let call_count = Arc::new(AtomicU64::new(0));
    let counter = call_count.clone();

    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(_body): Json<Value>| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Json(json!({
                    "id": "chatcmpl-filtered",
                    "object": "chat.completion",
                    "model": "gpt-4",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": ""
                        },
                        "finish_reason": "content_filter"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 0,
                        "total_tokens": 10
                    }
                }))
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let upstream_url = format!("http://{}", addr);
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });

    let proxy_url = start_mnemo_proxy(&upstream_url).await;
    let client = reqwest::Client::new();
    let req = chat_request("blocked content");

    // First request
    client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&req)
        .send()
        .await
        .unwrap();
    assert_eq!(call_count.load(Ordering::SeqCst), 1);

    // Second identical request — must not be cached
    client
        .post(format!("{}/v1/chat/completions", proxy_url))
        .json(&req)
        .send()
        .await
        .unwrap();

    assert_eq!(
        call_count.load(Ordering::SeqCst),
        2,
        "content_filter responses must not be cached"
    );
}
