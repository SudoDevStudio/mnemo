pub mod cost;
pub mod embedder;
pub mod lora;
pub mod staleness;

// Re-export key embedding types for convenience.
pub use embedder::{
    EmbedError, EmbedRequest, EmbedResult, Embedder, EmbeddingWorker, MockEmbedder, EMBEDDING_DIM,
};

// Re-export staleness detection types.
pub use staleness::{
    compute_staleness_risk, SignalProvider, StalenessSignals, StalenessUpdate, StalenessWorker,
};

#[cfg(feature = "onnx")]
pub use embedder::OnnxEmbedder;
