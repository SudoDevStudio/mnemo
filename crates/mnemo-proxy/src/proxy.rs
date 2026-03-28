use crate::config::{Provider, UpstreamConfig};
use bytes::Bytes;
use futures_core::Stream;
use reqwest::Client;
use serde_json::Value;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Forward a request to the upstream LLM provider.
pub struct Upstream {
    client: Client,
    config: UpstreamConfig,
}

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("upstream request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("upstream returned non-success status {status}: {body}")]
    UpstreamError { status: u16, body: String },
    #[error("stream interrupted: {0}")]
    StreamInterrupted(String),
    #[error("failed to parse streamed response: {0}")]
    StreamParse(String),
}

/// Response from the upstream provider, with cost metadata.
pub struct UpstreamResponse {
    pub body: Value,
    pub generation_cost: f64,
}

/// Result of collecting a complete SSE stream: the assembled full response and raw SSE chunks.
pub struct StreamedResponse {
    /// The fully assembled non-streaming response (suitable for caching).
    pub assembled: Value,
    /// Estimated dollar cost of the generation.
    pub generation_cost: f64,
}

/// A wrapper that collects SSE bytes as they flow through, then allows
/// retrieval of the assembled response after the stream ends.
pub struct CollectingStream {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    /// Accumulated raw SSE text for post-stream assembly.
    buffer: Vec<String>,
    /// Set to true if an error occurred mid-stream.
    errored: bool,
    /// Set to true when the stream has finished.
    finished: bool,
    /// Model from the first chunk (used when assembling).
    model: Option<String>,
    /// ID from the first chunk.
    id: Option<String>,
    /// Provider for cost estimation.
    provider: Provider,
}

impl CollectingStream {
    fn new(
        inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
        provider: Provider,
    ) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
            errored: false,
            finished: false,
            model: None,
            id: None,
            provider,
        }
    }

    /// After the stream is finished, assemble the full response from buffered chunks.
    /// Returns `None` if the stream errored or is not yet finished.
    pub fn assemble(&self) -> Option<StreamedResponse> {
        if self.errored || !self.finished {
            return None;
        }

        let mut content_parts: Vec<String> = Vec::new();
        let mut role = "assistant".to_string();
        let mut finish_reason: Option<String> = None;
        let mut usage: Option<Value> = None;
        let mut prompt_tokens: u64 = 0;
        let mut completion_tokens: u64 = 0;

        for raw_line in &self.buffer {
            // Each buffer entry is one SSE data payload (already stripped of "data: " prefix).
            if raw_line == "[DONE]" {
                continue;
            }
            let chunk: Value = match serde_json::from_str(raw_line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Extract content delta
            if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
                for choice in choices {
                    if let Some(delta) = choice.get("delta") {
                        if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
                            content_parts.push(c.to_string());
                        }
                        if let Some(r) = delta.get("role").and_then(|v| v.as_str()) {
                            role = r.to_string();
                        }
                    }
                    if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                        finish_reason = Some(fr.to_string());
                    }
                }
            }

            // Some providers include usage in the final chunk
            if let Some(u) = chunk.get("usage") {
                usage = Some(u.clone());
                prompt_tokens = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                completion_tokens = u
                    .get("completion_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
            }
        }

        let full_content = content_parts.join("");

        // Estimate completion tokens from content length if usage was not provided
        if usage.is_none() {
            // Rough heuristic: ~4 chars per token
            completion_tokens = (full_content.len() as u64 / 4).max(1);
        }

        let assembled = serde_json::json!({
            "id": self.id.as_deref().unwrap_or("chatcmpl-cached"),
            "object": "chat.completion",
            "model": self.model.as_deref().unwrap_or("unknown"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": role,
                    "content": full_content,
                },
                "finish_reason": finish_reason.as_deref().unwrap_or("stop"),
            }],
            "usage": usage.unwrap_or_else(|| serde_json::json!({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            })),
        });

        let generation_cost = estimate_cost(&assembled, &self.provider);

        Some(StreamedResponse {
            assembled,
            generation_cost,
        })
    }
}

impl Stream for CollectingStream {
    type Item = Result<Bytes, ProxyError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                // Parse SSE lines from this chunk and buffer the data payloads
                let text = String::from_utf8_lossy(&bytes);
                for line in text.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        let data = data.trim();
                        if !data.is_empty() {
                            // Extract model/id from first parseable chunk
                            if self.model.is_none() && data != "[DONE]" {
                                if let Ok(v) = serde_json::from_str::<Value>(data) {
                                    if let Some(m) = v.get("model").and_then(|v| v.as_str()) {
                                        self.model = Some(m.to_string());
                                    }
                                    if let Some(id) = v.get("id").and_then(|v| v.as_str()) {
                                        self.id = Some(id.to_string());
                                    }
                                }
                            }
                            self.buffer.push(data.to_string());
                        }
                    }
                }
                Poll::Ready(Some(Ok(bytes)))
            }
            Poll::Ready(Some(Err(e))) => {
                self.errored = true;
                Poll::Ready(Some(Err(ProxyError::StreamInterrupted(e.to_string()))))
            }
            Poll::Ready(None) => {
                self.finished = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Upstream {
    pub fn new(config: UpstreamConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client");

        Self { client, config }
    }

    /// Apply provider-specific authentication headers to a request.
    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.config.provider {
            Provider::Anthropic => {
                let req = req.header("anthropic-version", "2023-06-01");
                if let Some(ref api_key) = self.config.api_key {
                    req.header("x-api-key", api_key.as_str())
                } else {
                    req
                }
            }
            Provider::Ollama => {
                // Ollama runs locally; no auth needed.
                req
            }
            Provider::OpenAI | Provider::VertexAI | Provider::Custom => {
                if let Some(ref api_key) = self.config.api_key {
                    req.header("Authorization", format!("Bearer {}", api_key))
                } else {
                    req
                }
            }
        }
    }

    /// Map an OpenAI-compatible path to the provider's actual endpoint path.
    fn resolve_path(&self, path: &str) -> String {
        match self.config.provider {
            Provider::Anthropic => {
                // Translate OpenAI-style chat completions to Anthropic messages endpoint.
                if path == "/v1/chat/completions" {
                    "/v1/messages".to_string()
                } else {
                    path.to_string()
                }
            }
            _ => path.to_string(),
        }
    }

    /// Build the full URL for a given path, accounting for provider path mapping.
    fn build_url(&self, path: &str) -> String {
        let resolved = self.resolve_path(path);
        format!("{}{}", self.config.base_url.trim_end_matches('/'), resolved)
    }

    /// Forward a chat completions request to the upstream provider (non-streaming).
    pub async fn chat_completions(&self, body: &Value) -> Result<UpstreamResponse, ProxyError> {
        let url = self.build_url("/v1/chat/completions");
        self.forward(&url, body).await
    }

    /// Forward a chat completions request and return an SSE byte stream.
    /// The body MUST already contain `"stream": true`.
    pub async fn chat_completions_stream(
        &self,
        body: &Value,
    ) -> Result<CollectingStream, ProxyError> {
        let url = self.build_url("/v1/chat/completions");
        self.forward_stream(&url, body).await
    }

    /// Forward a legacy completions request.
    pub async fn completions(&self, body: &Value) -> Result<UpstreamResponse, ProxyError> {
        let url = self.build_url("/v1/completions");
        self.forward(&url, body).await
    }

    /// Generic forward to a URL (POST).
    pub async fn forward_to(
        &self,
        path: &str,
        body: &Value,
    ) -> Result<UpstreamResponse, ProxyError> {
        let url = self.build_url(path);
        self.forward(&url, body).await
    }

    /// Forward a GET request to the upstream provider.
    pub async fn forward_get(&self, path: &str) -> Result<UpstreamResponse, ProxyError> {
        let url = self.build_url(path);
        let req = self.client.get(&url);
        let req = self.apply_auth(req);

        let resp = req.send().await?;
        let status = resp.status().as_u16();
        let resp_body: Value = resp.json().await?;

        if status >= 400 {
            return Err(ProxyError::UpstreamError {
                status,
                body: resp_body.to_string(),
            });
        }

        Ok(UpstreamResponse {
            body: resp_body,
            generation_cost: 0.0,
        })
    }

    async fn forward(&self, url: &str, body: &Value) -> Result<UpstreamResponse, ProxyError> {
        let req = self.client.post(url).json(body);
        let req = self.apply_auth(req);

        let resp = req.send().await?;
        let status = resp.status().as_u16();
        let resp_body: Value = resp.json().await?;

        if status >= 400 {
            return Err(ProxyError::UpstreamError {
                status,
                body: resp_body.to_string(),
            });
        }

        let generation_cost = estimate_cost(&resp_body, &self.config.provider);

        Ok(UpstreamResponse {
            body: resp_body,
            generation_cost,
        })
    }

    /// Send a request and return the raw SSE byte stream wrapped in a `CollectingStream`.
    async fn forward_stream(
        &self,
        url: &str,
        body: &Value,
    ) -> Result<CollectingStream, ProxyError> {
        let req = self.client.post(url).json(body);
        let req = self.apply_auth(req);

        let resp = req.send().await?;
        let status = resp.status().as_u16();

        if status >= 400 {
            // For error responses, read the body as text
            let body_text = resp.text().await.unwrap_or_default();
            return Err(ProxyError::UpstreamError {
                status,
                body: body_text,
            });
        }

        let byte_stream = resp.bytes_stream();
        Ok(CollectingStream::new(
            Box::pin(byte_stream),
            self.config.provider.clone(),
        ))
    }
}

/// Estimate the dollar cost from the usage field in the response.
///
/// Provider-aware: Ollama runs locally so cost is always 0.0.
/// For cloud providers, cost is estimated from token counts and model-class rates.
pub fn estimate_cost(response: &Value, provider: &Provider) -> f64 {
    match provider {
        Provider::Ollama => {
            // Ollama runs locally — no API billing. Use a small non-zero value
            // so entries still participate in cost-aware eviction (a response
            // that took local GPU time is still worth caching).
            0.0001
        }
        _ => {
            if let Some(usage) = response.get("usage") {
                let prompt_tokens = usage
                    .get("prompt_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let completion_tokens = usage
                    .get("completion_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                // Cost rates per token by provider.
                let (prompt_rate, completion_rate) = match provider {
                    Provider::OpenAI => (0.00003, 0.00006),       // GPT-4 class
                    Provider::Anthropic => (0.000015, 0.000075),  // Claude Sonnet class
                    Provider::VertexAI => (0.0000125, 0.0000375), // Gemini Pro class
                    Provider::Custom => (0.00003, 0.00006),       // Default to GPT-4
                    Provider::Ollama => unreachable!(),
                };

                let prompt_cost = prompt_tokens as f64 * prompt_rate;
                let completion_cost = completion_tokens as f64 * completion_rate;

                prompt_cost + completion_cost
            } else {
                0.001 // Default minimum cost when usage data unavailable
            }
        }
    }
}

/// Reconstruct an SSE byte stream from a cached non-streaming response.
/// Produces a series of SSE `data:` lines mimicking the OpenAI streaming format,
/// ending with `data: [DONE]`.
pub fn cached_response_to_sse_bytes(cached: &Value) -> Vec<Bytes> {
    let mut chunks = Vec::new();

    let id = cached
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("chatcmpl-cached");
    let model = cached
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    // Extract the full content from the cached response
    let content = cached
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|msg| msg.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");

    let role = cached
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|msg| msg.get("role"))
        .and_then(|r| r.as_str())
        .unwrap_or("assistant");

    // First chunk: role
    let role_chunk = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "role": role, "content": "" },
            "finish_reason": null,
        }],
    });
    chunks.push(Bytes::from(format!(
        "data: {}\n\n",
        serde_json::to_string(&role_chunk).unwrap_or_default()
    )));

    // Content chunks: split into ~20-char segments to simulate streaming
    if !content.is_empty() {
        let chunk_size = 20;
        let chars: Vec<char> = content.chars().collect();
        for segment in chars.chunks(chunk_size) {
            let segment_str: String = segment.iter().collect();
            let content_chunk = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": { "content": segment_str },
                    "finish_reason": null,
                }],
            });
            chunks.push(Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&content_chunk).unwrap_or_default()
            )));
        }
    }

    // Final chunk: finish_reason
    let done_chunk = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    });
    chunks.push(Bytes::from(format!(
        "data: {}\n\n",
        serde_json::to_string(&done_chunk).unwrap_or_default()
    )));

    // Terminal marker
    chunks.push(Bytes::from("data: [DONE]\n\n"));

    chunks
}
