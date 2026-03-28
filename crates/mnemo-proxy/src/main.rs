use std::path::PathBuf;
use std::sync::Arc;

use mnemo_cache::eviction::eviction_priority;
use mnemo_cache::l1::L1Cache;
use mnemo_cache::l2::L2SemanticCache;
use mnemo_cache::l3::L3RedisCache;
use mnemo_intelligence::{EmbedRequest, EmbedResult, EmbeddingWorker, MockEmbedder, EMBEDDING_DIM};
use mnemo_proxy::config::{Config, LogFormat};
use mnemo_proxy::proxy::Upstream;
use mnemo_proxy::server::AppState;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    // Load config first (before tracing init so we can use logging config)
    let config_path = std::env::var("MNEMO_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("mnemo.yaml"));

    let config = Config::load(&config_path).unwrap_or_else(|e| {
        eprintln!(
            "FATAL: failed to load config '{}': {}",
            config_path.display(),
            e
        );
        std::process::exit(1);
    });

    // Initialize tracing from: MNEMO_LOG > RUST_LOG > config file > "info"
    let env_filter = std::env::var("MNEMO_LOG")
        .or_else(|_| std::env::var("RUST_LOG"))
        .unwrap_or_else(|_| config.logging.level.clone());

    let filter = EnvFilter::try_new(&env_filter).unwrap_or_else(|_| EnvFilter::new("info"));

    match config.logging.format {
        LogFormat::Json => {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .json()
                .init();
        }
        LogFormat::Pretty => {
            tracing_subscriber::fmt().with_env_filter(filter).init();
        }
    }

    let bind_addr = config.bind.clone();

    tracing::info!(
        provider = ?config.upstream.provider,
        upstream = %config.upstream.base_url,
        bind = %bind_addr,
        l1_max = config.cache.l1_max_entries,
        l2_max = config.cache.l2_max_entries,
        log_level = %env_filter,
        log_format = ?config.logging.format,
        log_requests = config.logging.log_requests,
        log_cache_events = config.logging.log_cache_events,
        "starting mnemo proxy"
    );

    // ── Initialize cache layers ──

    // L1: exact-match (DashMap + moka)
    let l1 = L1Cache::new(config.cache.l1_max_entries);

    // L2: semantic cache (HNSW)
    let l2 = Arc::new(L2SemanticCache::with_defaults(
        config.cache.l2_max_entries as usize,
        EMBEDDING_DIM,
    ));

    // L3: Redis persistence (optional)
    let l3 = if let Some(ref redis_url) = config.cache.l3_redis_url {
        match L3RedisCache::new(redis_url) {
            Ok(cache) => {
                tracing::info!(url = %redis_url, "L3 Redis cache connected");
                Some(Arc::new(cache))
            }
            Err(e) => {
                tracing::warn!(error = %e, "L3 Redis cache failed to connect — running without L3");
                None
            }
        }
    } else {
        tracing::info!("L3 Redis cache not configured — running with L1+L2 only");
        None
    };

    // ── Initialize embedding worker ──

    let (embed_request_tx, embed_request_rx) = tokio::sync::mpsc::channel::<EmbedRequest>(1024);
    let (embed_result_tx, embed_result_rx) = tokio::sync::mpsc::channel::<EmbedResult>(1024);

    // Use real ONNX embedder when the feature is enabled and model files exist,
    // otherwise fall back to the deterministic mock embedder.
    #[cfg(feature = "onnx")]
    let (embedder, backend_name): (Arc<dyn mnemo_intelligence::Embedder>, &str) = {
        let model_dir =
            std::env::var("MNEMO_MODEL_DIR").unwrap_or_else(|_| "models/bge-small-en".to_string());
        let model_path = format!("{}/model.onnx", model_dir);
        let tokenizer_path = format!("{}/tokenizer.json", model_dir);

        if std::path::Path::new(&model_path).exists()
            && std::path::Path::new(&tokenizer_path).exists()
        {
            match mnemo_intelligence::OnnxEmbedder::new(&model_path, &tokenizer_path, EMBEDDING_DIM)
            {
                Ok(e) => {
                    tracing::info!(model = %model_path, "loaded ONNX embedding model");
                    (Arc::new(e), "onnx")
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to load ONNX model, falling back to mock embedder");
                    (Arc::new(MockEmbedder::new()), "mock (deterministic hash)")
                }
            }
        } else {
            tracing::info!(
                model_path = %model_path,
                "ONNX model files not found, using mock embedder (set MNEMO_MODEL_DIR to override)"
            );
            (Arc::new(MockEmbedder::new()), "mock (deterministic hash)")
        }
    };

    #[cfg(not(feature = "onnx"))]
    let (embedder, backend_name): (Arc<dyn mnemo_intelligence::Embedder>, &str) =
        (Arc::new(MockEmbedder::new()), "mock (deterministic hash)");

    let worker = EmbeddingWorker::new(embedder.clone())
        .with_batch_size(config.intelligence.training_batch_size);
    let _worker_handle = worker.spawn(embed_request_rx, embed_result_tx);
    tracing::info!(
        backend = backend_name,
        dim = EMBEDDING_DIM,
        "embedding worker started"
    );

    // ── Background task: consume embedding results and insert into L2 ──
    let l2_for_bg = Arc::clone(&l2);
    let l1_for_bg = l1.clone();
    tokio::spawn(async move {
        let mut rx = embed_result_rx;
        while let Some(result) = rx.recv().await {
            // Compute eviction priority from the L1 cache entry metadata.
            // Use peek() to avoid perturbing hit stats from background work.
            let priority = if let Some(entry) = l1_for_bg.peek(&result.cache_key).await {
                eviction_priority(&entry)
            } else {
                1.0
            };
            l2_for_bg.insert(result.cache_key.clone(), result.embedding, priority);
            tracing::debug!(key = %result.cache_key, priority, "inserted embedding into L2");
        }
        tracing::info!("embedding result consumer stopped");
    });

    // ── Start staleness worker (if enabled) ──
    if config.intelligence.staleness_detection_enabled {
        let (staleness_tx, mut staleness_rx) =
            tokio::sync::mpsc::channel::<mnemo_intelligence::StalenessUpdate>(256);

        let l1_for_staleness = l1.clone();
        let l2_for_staleness = Arc::clone(&l2);

        // The staleness worker needs a SignalProvider. We implement a simple one
        // that reads from L1.
        let signal_l1 = l1.clone();
        let provider = L1SignalProvider { l1: signal_l1 };

        let staleness_worker =
            mnemo_intelligence::StalenessWorker::new(config.intelligence.training_interval_seconds);
        let _staleness_handle = staleness_worker.spawn(staleness_tx, provider, 0.7);
        tracing::info!(
            interval_secs = config.intelligence.training_interval_seconds,
            "staleness detection worker started"
        );

        // Background task: consume staleness updates and apply to L1/L2
        tokio::spawn(async move {
            while let Some(update) = staleness_rx.recv().await {
                // Use peek() to read without perturbing hit stats
                if let Some(entry) = l1_for_staleness.peek(&update.cache_key).await {
                    let mut updated = (*entry).clone();
                    updated.staleness_risk = update.staleness_risk;
                    l1_for_staleness
                        .insert(update.cache_key.clone(), updated.clone())
                        .await;
                    // Update L2 eviction priority
                    l2_for_staleness
                        .update_priority(&update.cache_key, eviction_priority(&updated));
                }
                tracing::debug!(
                    key = %update.cache_key,
                    risk = update.staleness_risk,
                    "applied staleness update"
                );
            }
        });
    } else {
        tracing::info!("staleness detection disabled");
    }

    // ── Start LoRA training worker (if enabled) ──
    if config.intelligence.lora_training_enabled {
        let lora_config = mnemo_intelligence::lora::LoraConfig {
            batch_size: config.intelligence.training_batch_size,
            training_interval_secs: config.intelligence.training_interval_seconds,
            ..Default::default()
        };
        let lora_worker = Arc::new(mnemo_intelligence::lora::LoraWorker::new(lora_config));
        let _lora_handle = Arc::clone(&lora_worker).spawn();
        tracing::info!(
            batch_size = config.intelligence.training_batch_size,
            interval_secs = config.intelligence.training_interval_seconds,
            "LoRA training worker started"
        );
    } else {
        tracing::info!("LoRA training disabled");
    }

    // ── Initialize upstream proxy ──
    let upstream = Upstream::new(config.upstream.clone());

    let state = Arc::new(AppState {
        config,
        l1,
        l2,
        l3,
        upstream,
        embed_tx: embed_request_tx,
        embedder,
    });

    let app = mnemo_proxy::server::router(state);

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .unwrap_or_else(|e| {
            tracing::error!(error = %e, addr = %bind_addr, "failed to bind");
            std::process::exit(1);
        });

    tracing::info!(addr = %bind_addr, "mnemo proxy listening");

    axum::serve(listener, app).await.unwrap_or_else(|e| {
        tracing::error!(error = %e, "server error");
        std::process::exit(1);
    });
}

/// Simple SignalProvider that reads cache entry metadata from L1.
struct L1SignalProvider {
    l1: L1Cache,
}

impl mnemo_intelligence::SignalProvider for L1SignalProvider {
    fn signals_for(&self, cache_key: &str) -> Option<mnemo_intelligence::StalenessSignals> {
        // peek() is async (moka future::Cache), but SignalProvider is sync.
        // block_in_place + block_on is safe here because we're in a multi-threaded
        // tokio runtime and this runs on the cold-path staleness scanner.
        let entry = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.l1.peek(cache_key))
        })?;

        let now = chrono::Utc::now();
        let age_hours = (now - entry.created_at).num_seconds() as f64 / 3600.0;
        let hours_since_last_hit = (now - entry.last_hit_at).num_seconds() as f64 / 3600.0;

        Some(mnemo_intelligence::StalenessSignals {
            age_hours,
            hours_since_last_hit,
            hit_count: entry.hit_count,
            was_corrected: false,
            response_drift_score: 0.0,
        })
    }

    fn all_keys(&self) -> Vec<String> {
        self.l1.tracked_keys()
    }
}
