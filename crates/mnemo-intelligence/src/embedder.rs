//! Embedding worker for the L2 HNSW semantic cache.
//!
//! This runs on the **cold path** as an async background worker.
//! It never blocks the hot path.
//!
//! Two backends are provided:
//! - `MockEmbedder`: deterministic hash-based embeddings (default, zero dependencies).
//! - `OnnxEmbedder`: real ONNX Runtime inference via `ort` + HuggingFace `tokenizers`
//!   (behind the `onnx` feature flag).
//!
//! The [`EmbeddingWorker`] wraps any backend implementing [`Embedder`] and runs a
//! channel-driven background loop consuming [`EmbedRequest`]s and producing
//! [`EmbedResult`]s.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

/// Dimension of the embedding vectors (BGE-small-en / all-MiniLM-L6-v2).
pub const EMBEDDING_DIM: usize = 384;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during embedding operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("failed to load model: {0}")]
    ModelLoad(String),

    #[error("failed to load tokenizer: {0}")]
    TokenizerLoad(String),

    #[error("tokenization failed: {0}")]
    Tokenization(String),

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("unexpected output shape: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("worker channel closed")]
    ChannelClosed,
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, EmbedError>;

// ---------------------------------------------------------------------------
// Request / result types
// ---------------------------------------------------------------------------

/// A request to embed a cache entry's text.
#[derive(Debug, Clone)]
pub struct EmbedRequest {
    pub cache_key: String,
    pub text: String,
}

/// The result of an embedding operation.
#[derive(Debug, Clone)]
pub struct EmbedResult {
    pub cache_key: String,
    pub embedding: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Embedder trait
// ---------------------------------------------------------------------------

/// Trait for embedding backends.
///
/// Implementations must be `Send + Sync` so they can be shared across
/// async tasks behind an `Arc`.
pub trait Embedder: Send + Sync {
    /// Generate a single embedding vector for the given text.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for a batch of texts.
    ///
    /// The default implementation calls [`embed`](Embedder::embed) in a loop.
    /// Backends that support native batching should override this.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Human-readable name of the backend (for logging).
    fn backend_name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Mock embedder (default — always available)
// ---------------------------------------------------------------------------

/// A deterministic mock embedder that produces consistent, hash-based
/// embeddings. Useful for development, testing, and CI where no ONNX model
/// is available.
///
/// The output is deterministic: the same input text always produces the same
/// embedding vector. Vectors are L2-normalized so cosine similarity works.
pub struct MockEmbedder {
    dim: usize,
}

impl MockEmbedder {
    /// Create a new mock embedder with the standard embedding dimension.
    pub fn new() -> Self {
        Self { dim: EMBEDDING_DIM }
    }

    /// Create a mock embedder with a custom dimension.
    pub fn with_dim(dim: usize) -> Self {
        Self { dim }
    }
}

impl Default for MockEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Use a simple hash-based approach to generate deterministic
        // pseudo-random embeddings. We seed a basic PRNG with a hash
        // of the input text so the same text always maps to the same
        // vector.
        let hash = deterministic_hash(text);
        let mut state = hash;
        let mut vec = Vec::with_capacity(self.dim);
        for _ in 0..self.dim {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to [-1, 1]
            let val = (state as f32) / (u64::MAX as f32) * 2.0 - 1.0;
            vec.push(val);
        }
        // L2-normalize
        let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        Ok(vec)
    }

    fn backend_name(&self) -> &str {
        "mock (deterministic hash)"
    }
}

/// FNV-1a hash for deterministic seeding. Not cryptographic, just consistent.
fn deterministic_hash(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// ONNX embedder (behind `onnx` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "onnx")]
mod onnx_backend {
    use super::*;
    use parking_lot::Mutex;

    /// Real embedding backend using ONNX Runtime for model inference
    /// and HuggingFace `tokenizers` for text preprocessing.
    ///
    /// The ONNX session requires `&mut self` for `run()`, so we wrap it
    /// in a `Mutex` to satisfy the `Embedder: Send + Sync` requirement.
    pub struct OnnxEmbedder {
        session: Mutex<ort::session::Session>,
        tokenizer: tokenizers::Tokenizer,
        dim: usize,
    }

    impl OnnxEmbedder {
        /// Load an ONNX embedding model and its tokenizer.
        ///
        /// # Arguments
        /// * `model_path` — path to the `.onnx` model file
        /// * `tokenizer_path` — path to the `tokenizer.json` file
        /// * `dim` — expected embedding dimension (e.g. 384)
        pub fn new(model_path: &str, tokenizer_path: &str, dim: usize) -> Result<Self> {
            let mut builder = ort::session::Session::builder()
                .map_err(|e| EmbedError::ModelLoad(e.to_string()))?;
            builder = builder
                .with_intra_threads(1)
                .map_err(|e| EmbedError::ModelLoad(e.to_string()))?;
            let session = builder
                .commit_from_file(model_path)
                .map_err(|e| EmbedError::ModelLoad(e.to_string()))?;

            let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
                .map_err(|e| EmbedError::TokenizerLoad(e.to_string()))?;

            Ok(Self {
                session: Mutex::new(session),
                tokenizer,
                dim,
            })
        }

        /// Tokenize a batch of texts and run inference.
        fn run_inference(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            // Tokenize
            let encodings = self
                .tokenizer
                .encode_batch(texts.to_vec(), true)
                .map_err(|e| EmbedError::Tokenization(e.to_string()))?;

            let batch_size = encodings.len();
            let seq_len = encodings
                .iter()
                .map(|e| e.get_ids().len())
                .max()
                .unwrap_or(0);

            // Build input tensors (input_ids, attention_mask, token_type_ids)
            let mut input_ids: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
            let mut attention_mask: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
            let mut token_type_ids: Vec<i64> = Vec::with_capacity(batch_size * seq_len);

            for encoding in &encodings {
                let ids = encoding.get_ids();
                let mask = encoding.get_attention_mask();
                let type_ids = encoding.get_type_ids();

                for i in 0..seq_len {
                    if i < ids.len() {
                        input_ids.push(ids[i] as i64);
                        attention_mask.push(mask[i] as i64);
                        token_type_ids.push(type_ids[i] as i64);
                    } else {
                        input_ids.push(0);
                        attention_mask.push(0);
                        token_type_ids.push(0);
                    }
                }
            }

            let input_ids_tensor =
                ort::value::Tensor::from_array((vec![batch_size, seq_len], input_ids))
                    .map_err(|e| EmbedError::Inference(format!("input_ids tensor: {e}")))?;

            let attention_mask_tensor =
                ort::value::Tensor::from_array((vec![batch_size, seq_len], attention_mask))
                    .map_err(|e| EmbedError::Inference(format!("attention_mask tensor: {e}")))?;

            let token_type_ids_tensor =
                ort::value::Tensor::from_array((vec![batch_size, seq_len], token_type_ids))
                    .map_err(|e| EmbedError::Inference(format!("token_type_ids tensor: {e}")))?;

            // Build inputs using ort::inputs! macro (returns Vec, not Result)
            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ];

            // Session::run requires &mut self
            let mut session = self.session.lock();
            let outputs = session
                .run(inputs)
                .map_err(|e| EmbedError::Inference(e.to_string()))?;

            // Extract the last_hidden_state or sentence_embedding output.
            // Most ONNX sentence-transformer models output shape
            // [batch, seq_len, hidden] or [batch, hidden].
            let output = outputs
                .get("last_hidden_state")
                .or_else(|| outputs.get("sentence_embedding"))
                .ok_or_else(|| EmbedError::Inference("no recognized output tensor found".into()))?;

            let (tensor_shape, tensor_data) = output
                .try_extract_tensor::<f32>()
                .map_err(|e| EmbedError::Inference(format!("extracting tensor: {e}")))?;

            let rank = tensor_shape.len();

            // Handle [batch, hidden] — direct sentence embeddings
            if rank == 2 {
                let hidden = tensor_shape[1] as usize;
                if hidden != self.dim {
                    return Err(EmbedError::ShapeMismatch {
                        expected: self.dim,
                        got: hidden,
                    });
                }
                let results = tensor_data
                    .chunks(hidden)
                    .take(batch_size)
                    .map(l2_normalize)
                    .collect();
                return Ok(results);
            }

            // Handle [batch, seq_len, hidden] — mean pooling needed
            if rank == 3 {
                let hidden = tensor_shape[2] as usize;
                if hidden != self.dim {
                    return Err(EmbedError::ShapeMismatch {
                        expected: self.dim,
                        got: hidden,
                    });
                }

                let mut results = Vec::with_capacity(batch_size);
                for (b, encoding) in encodings.iter().enumerate().take(batch_size) {
                    let attn_mask = encoding.get_attention_mask();
                    let mask_len = attn_mask.iter().sum::<u32>() as f32;
                    let mut pooled = vec![0.0f32; hidden];
                    for s in 0..seq_len {
                        let attn = if s < attn_mask.len() {
                            attn_mask[s] as f32
                        } else {
                            0.0
                        };
                        if attn > 0.0 {
                            let offset = (b * seq_len + s) * hidden;
                            for (d, p) in pooled.iter_mut().enumerate() {
                                *p += tensor_data[offset + d];
                            }
                        }
                    }
                    if mask_len > 0.0 {
                        for p in pooled.iter_mut() {
                            *p /= mask_len;
                        }
                    }
                    results.push(l2_normalize(&pooled));
                }
                return Ok(results);
            }

            Err(EmbedError::Inference(format!(
                "unexpected tensor rank: {}",
                rank
            )))
        }
    }

    impl Embedder for OnnxEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            let mut results = self.run_inference(&[text])?;
            results
                .pop()
                .ok_or_else(|| EmbedError::Inference("empty result".into()))
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            if texts.is_empty() {
                return Ok(Vec::new());
            }
            self.run_inference(texts)
        }

        fn backend_name(&self) -> &str {
            "onnx (ort)"
        }
    }
}

#[cfg(feature = "onnx")]
pub use onnx_backend::OnnxEmbedder;

// ---------------------------------------------------------------------------
// L2 normalization helper
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

// ---------------------------------------------------------------------------
// EmbeddingWorker — channel-driven async background worker
// ---------------------------------------------------------------------------

/// The embedding worker consumes [`EmbedRequest`]s from an MPSC channel,
/// generates embeddings using the configured [`Embedder`] backend, and
/// sends [`EmbedResult`]s to an output channel.
///
/// This is designed for the **cold path** — it runs as a background
/// `tokio::task` and never blocks the hot proxy path.
pub struct EmbeddingWorker {
    embedder: Arc<dyn Embedder>,
    /// Maximum number of requests to batch together before running inference.
    batch_size: usize,
}

impl EmbeddingWorker {
    /// Create a new worker with the given embedder backend.
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            embedder,
            batch_size: 32,
        }
    }

    /// Set the maximum batch size for inference. Defaults to 32.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Create a worker using the default [`MockEmbedder`].
    pub fn mock() -> Self {
        Self::new(Arc::new(MockEmbedder::new()))
    }

    /// Create a worker using a real ONNX model (requires `onnx` feature).
    #[cfg(feature = "onnx")]
    pub fn onnx(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let embedder = OnnxEmbedder::new(model_path, tokenizer_path, EMBEDDING_DIM)?;
        Ok(Self::new(Arc::new(embedder)))
    }

    /// Generate a single embedding (convenience method, not the worker loop).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embedder.embed(text)
    }

    /// Generate a batch of embeddings (convenience method, not the worker loop).
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embedder.embed_batch(texts)
    }

    /// Spawn the background worker loop.
    ///
    /// The worker reads from `rx`, generates embeddings, and sends results
    /// to `tx`. It runs until the input channel is closed, then exits
    /// gracefully.
    ///
    /// The worker accumulates up to `batch_size` requests before running
    /// a single batched inference call, amortizing overhead.
    pub fn spawn(
        self,
        mut rx: mpsc::Receiver<EmbedRequest>,
        tx: mpsc::Sender<EmbedResult>,
    ) -> JoinHandle<()> {
        let embedder = self.embedder.clone();
        let batch_size = self.batch_size;

        tokio::spawn(async move {
            tracing::info!(
                backend = embedder.backend_name(),
                batch_size,
                "embedding worker started"
            );

            let mut batch_keys: Vec<String> = Vec::with_capacity(batch_size);
            let mut batch_texts: Vec<String> = Vec::with_capacity(batch_size);

            loop {
                batch_keys.clear();
                batch_texts.clear();

                // Wait for the first request (blocking).
                let first = match rx.recv().await {
                    Some(req) => req,
                    None => {
                        tracing::info!("embedding worker: input channel closed, shutting down");
                        break;
                    }
                };

                batch_keys.push(first.cache_key);
                batch_texts.push(first.text);

                // Drain up to (batch_size - 1) more requests without waiting.
                while batch_keys.len() < batch_size {
                    match rx.try_recv() {
                        Ok(req) => {
                            batch_keys.push(req.cache_key);
                            batch_texts.push(req.text);
                        }
                        Err(_) => break,
                    }
                }

                let n = batch_keys.len();
                tracing::debug!(count = n, "embedding worker: processing batch");

                // Run inference (synchronous, but on a background task so it
                // doesn't block the Tokio runtime's I/O reactor).
                // Use spawn_blocking for the CPU-bound inference work to avoid
                // starving the async runtime.
                let embedder_clone = embedder.clone();
                let owned_texts: Vec<String> = batch_texts.clone();

                let result = tokio::task::spawn_blocking(move || {
                    let refs: Vec<&str> = owned_texts.iter().map(|s| s.as_str()).collect();
                    embedder_clone.embed_batch(&refs)
                })
                .await;

                let embeddings = match result {
                    Ok(Ok(embeddings)) => embeddings,
                    Ok(Err(e)) => {
                        tracing::error!(error = %e, "embedding inference failed");
                        continue;
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "embedding task panicked");
                        continue;
                    }
                };

                if embeddings.len() != n {
                    tracing::error!(
                        expected = n,
                        got = embeddings.len(),
                        "embedding count mismatch"
                    );
                    continue;
                }

                // Send results to the output channel.
                for (key, embedding) in batch_keys.drain(..).zip(embeddings) {
                    let result = EmbedResult {
                        cache_key: key,
                        embedding,
                    };
                    if tx.send(result).await.is_err() {
                        tracing::warn!("embedding worker: output channel closed, shutting down");
                        return;
                    }
                }
            }

            tracing::info!("embedding worker stopped");
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_embedder_produces_correct_dimension() {
        let embedder = MockEmbedder::new();
        let vec = embedder.embed("hello world").unwrap();
        assert_eq!(vec.len(), EMBEDDING_DIM);
    }

    #[test]
    fn mock_embedder_is_deterministic() {
        let embedder = MockEmbedder::new();
        let v1 = embedder.embed("test input").unwrap();
        let v2 = embedder.embed("test input").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn mock_embedder_different_inputs_differ() {
        let embedder = MockEmbedder::new();
        let v1 = embedder.embed("hello").unwrap();
        let v2 = embedder.embed("world").unwrap();
        assert_ne!(v1, v2);
    }

    #[test]
    fn mock_embedder_output_is_normalized() {
        let embedder = MockEmbedder::new();
        let vec = embedder.embed("normalize me").unwrap();
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}");
    }

    #[test]
    fn mock_embedder_batch() {
        let embedder = MockEmbedder::new();
        let results = embedder.embed_batch(&["a", "b", "c"]).unwrap();
        assert_eq!(results.len(), 3);
        for v in &results {
            assert_eq!(v.len(), EMBEDDING_DIM);
        }
    }

    #[tokio::test]
    async fn worker_processes_requests() {
        let (req_tx, req_rx) = mpsc::channel::<EmbedRequest>(16);
        let (res_tx, mut res_rx) = mpsc::channel::<EmbedResult>(16);

        let worker = EmbeddingWorker::mock();
        let handle = worker.spawn(req_rx, res_tx);

        // Send a request
        req_tx
            .send(EmbedRequest {
                cache_key: "key1".into(),
                text: "hello world".into(),
            })
            .await
            .unwrap();

        // Receive the result
        let result = res_rx.recv().await.unwrap();
        assert_eq!(result.cache_key, "key1");
        assert_eq!(result.embedding.len(), EMBEDDING_DIM);

        // Drop the sender to signal shutdown
        drop(req_tx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn worker_handles_multiple_requests() {
        let (req_tx, req_rx) = mpsc::channel::<EmbedRequest>(64);
        let (res_tx, mut res_rx) = mpsc::channel::<EmbedResult>(64);

        let worker = EmbeddingWorker::mock().with_batch_size(4);
        let handle = worker.spawn(req_rx, res_tx);

        let n = 10;
        for i in 0..n {
            req_tx
                .send(EmbedRequest {
                    cache_key: format!("key{i}"),
                    text: format!("text {i}"),
                })
                .await
                .unwrap();
        }
        drop(req_tx);

        let mut results = Vec::new();
        while let Some(r) = res_rx.recv().await {
            results.push(r);
        }
        assert_eq!(results.len(), n);

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn worker_deterministic_through_channel() {
        let embedder = MockEmbedder::new();
        let expected = embedder.embed("consistency check").unwrap();

        let (req_tx, req_rx) = mpsc::channel::<EmbedRequest>(4);
        let (res_tx, mut res_rx) = mpsc::channel::<EmbedResult>(4);

        let worker = EmbeddingWorker::mock();
        let handle = worker.spawn(req_rx, res_tx);

        req_tx
            .send(EmbedRequest {
                cache_key: "ck".into(),
                text: "consistency check".into(),
            })
            .await
            .unwrap();

        let result = res_rx.recv().await.unwrap();
        assert_eq!(result.embedding, expected);

        drop(req_tx);
        handle.await.unwrap();
    }
}
