//! L2 Semantic Cache — HNSW-backed approximate nearest-neighbor lookup.
//!
//! Given a query embedding, finds the most similar cached entry using cosine
//! similarity. Returns a cache hit when similarity exceeds a configurable
//! threshold (default 0.92).
//!
//! Thread-safe: all public methods take `&self` and synchronize internally
//! via `parking_lot::RwLock`.

use std::collections::HashMap;

use hnsw_rs::anndists::dist::distances::DistCosine;
use hnsw_rs::hnsw::{Hnsw, Neighbour};
use parking_lot::RwLock;
use tracing::{debug, warn};

/// Default cosine similarity threshold for cache hits.
const DEFAULT_SIMILARITY_THRESHOLD: f64 = 0.92;

/// HNSW max connections per node (M parameter). 16 is a good balance between
/// recall and memory for embedding dimensions 384-1536.
const HNSW_MAX_NB_CONNECTION: usize = 16;

/// HNSW max layers.
const HNSW_MAX_LAYER: usize = 16;

/// HNSW ef_construction — higher values give better recall at build time.
const HNSW_EF_CONSTRUCTION: usize = 200;

/// HNSW ef_search — higher values give better recall at query time.
const HNSW_EF_SEARCH: usize = 64;

/// Internal state guarded by a RwLock. We rebuild the HNSW index from scratch
/// on mutations (insert/remove) because `hnsw_rs` does not support point
/// deletion. The rebuild cost is acceptable because:
///   - Inserts are infrequent relative to searches (cache miss path only).
///   - The index is in-memory and rebuilds are fast for typical cache sizes.
struct IndexState {
    /// Maps DataId (usize) -> cache_key.
    id_to_key: Vec<String>,
    /// Maps cache_key -> (DataId, embedding, eviction_priority).
    key_to_entry: HashMap<String, (usize, Vec<f32>, f64)>,
    /// The HNSW index. `'static` lifetime because we own all data (Vec<f32>).
    hnsw: Hnsw<'static, f32, DistCosine>,
    /// Maximum number of entries before eviction kicks in.
    max_entries: usize,
    /// Embedding dimensionality — all vectors must match this.
    embedding_dim: usize,
}

impl IndexState {
    fn new(max_entries: usize, embedding_dim: usize) -> Self {
        let hnsw = Hnsw::new(
            HNSW_MAX_NB_CONNECTION,
            max_entries.max(1),
            HNSW_MAX_LAYER,
            HNSW_EF_CONSTRUCTION,
            DistCosine,
        );
        IndexState {
            id_to_key: Vec::new(),
            key_to_entry: HashMap::new(),
            hnsw,
            max_entries,
            embedding_dim,
        }
    }

    /// Rebuild the HNSW index from the current `key_to_entry` map.
    fn rebuild(&mut self) {
        let capacity = self.key_to_entry.len().max(1);
        let new_hnsw = Hnsw::new(
            HNSW_MAX_NB_CONNECTION,
            capacity,
            HNSW_MAX_LAYER,
            HNSW_EF_CONSTRUCTION,
            DistCosine,
        );

        let mut new_id_to_key = Vec::with_capacity(capacity);
        let mut new_key_to_entry = HashMap::with_capacity(capacity);

        // Re-insert all entries with fresh sequential DataIds.
        for (key, (_old_id, embedding, priority)) in self.key_to_entry.drain() {
            let new_id = new_id_to_key.len();
            new_hnsw.insert((&embedding, new_id));
            new_id_to_key.push(key.clone());
            new_key_to_entry.insert(key, (new_id, embedding, priority));
        }

        self.hnsw = new_hnsw;
        self.id_to_key = new_id_to_key;
        self.key_to_entry = new_key_to_entry;
    }

    /// Find the key with the lowest eviction priority.
    fn lowest_priority_key(&self) -> Option<(String, f64)> {
        self.key_to_entry
            .iter()
            .min_by(|a, b| {
                a.1 .2
                    .partial_cmp(&b.1 .2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(key, (_, _, priority))| (key.clone(), *priority))
    }
}

/// L2 Semantic Cache backed by an in-memory HNSW index.
///
/// Thread-safe: designed for concurrent read-heavy workloads with infrequent
/// writes. Reads acquire a read lock; inserts and removes acquire a write lock
/// and rebuild the index.
pub struct L2SemanticCache {
    state: RwLock<IndexState>,
    similarity_threshold: f64,
}

impl L2SemanticCache {
    /// Create a new L2 semantic cache.
    ///
    /// # Arguments
    /// * `max_entries` — Maximum number of cached embeddings.
    /// * `similarity_threshold` — Cosine similarity threshold for a cache hit (0.0-1.0).
    ///   Use `0.0` for "always match nearest", or the default `0.92` for high precision.
    /// * `embedding_dim` — Dimensionality of embedding vectors (e.g., 384 for MiniLM,
    ///   1536 for OpenAI text-embedding-ada-002).
    pub fn new(max_entries: usize, similarity_threshold: f64, embedding_dim: usize) -> Self {
        debug!(
            max_entries,
            similarity_threshold, embedding_dim, "creating L2 semantic cache"
        );
        L2SemanticCache {
            state: RwLock::new(IndexState::new(max_entries, embedding_dim)),
            similarity_threshold,
        }
    }

    /// Create a new L2 semantic cache with the default similarity threshold (0.92).
    pub fn with_defaults(max_entries: usize, embedding_dim: usize) -> Self {
        Self::new(max_entries, DEFAULT_SIMILARITY_THRESHOLD, embedding_dim)
    }

    /// Search for the nearest cached entry to the given embedding.
    ///
    /// Returns `Some((cache_key, similarity))` if the nearest neighbor's cosine
    /// similarity exceeds the configured threshold, `None` otherwise.
    ///
    /// This is the hot-path operation — it only acquires a read lock.
    pub fn search(&self, embedding: &[f32]) -> Option<(String, f32)> {
        let state = self.state.read();

        if state.key_to_entry.is_empty() {
            return None;
        }

        if embedding.len() != state.embedding_dim {
            warn!(
                expected = state.embedding_dim,
                got = embedding.len(),
                "embedding dimension mismatch in L2 search"
            );
            return None;
        }

        // Search for 1 nearest neighbor. ef_search controls search quality.
        let neighbours: Vec<Neighbour> = state.hnsw.search(embedding, 1, HNSW_EF_SEARCH);

        let nearest = neighbours.first()?;

        // hnsw_rs DistCosine returns cosine *distance* = 1.0 - cosine_similarity
        let similarity = 1.0 - nearest.distance;

        if (similarity as f64) < self.similarity_threshold {
            debug!(
                similarity,
                threshold = self.similarity_threshold,
                "L2 search below threshold"
            );
            return None;
        }

        let data_id = nearest.d_id;
        let cache_key = state.id_to_key.get(data_id)?;

        Some((cache_key.clone(), similarity))
    }

    /// Insert a (cache_key, embedding) pair into the index with an eviction priority.
    ///
    /// If the cache key already exists, its embedding and priority are updated.
    /// If the cache is full, the lowest-priority entry is evicted to make room —
    /// unless the new entry's priority is lower than all existing entries, in which
    /// case the insert is rejected.
    pub fn insert(&self, cache_key: String, embedding: Vec<f32>, priority: f64) {
        let mut state = self.state.write();

        if embedding.len() != state.embedding_dim {
            warn!(
                expected = state.embedding_dim,
                got = embedding.len(),
                key = %cache_key,
                "embedding dimension mismatch in L2 insert, skipping"
            );
            return;
        }

        let is_update = state.key_to_entry.contains_key(&cache_key);

        if !is_update && state.key_to_entry.len() >= state.max_entries {
            // Cache full — evict the lowest-priority entry if the new one is worth more.
            if let Some((victim_key, victim_priority)) = state.lowest_priority_key() {
                if priority > victim_priority {
                    debug!(
                        evicted_key = %victim_key,
                        evicted_priority = victim_priority,
                        new_key = %cache_key,
                        new_priority = priority,
                        "L2 evicting lowest-priority entry"
                    );
                    state.key_to_entry.remove(&victim_key);
                } else {
                    debug!(
                        max_entries = state.max_entries,
                        key = %cache_key,
                        priority,
                        "L2 cache full, new entry priority too low — rejecting"
                    );
                    return;
                }
            } else {
                return;
            }
        }

        if is_update {
            // Remove old entry; rebuild will reassign IDs.
            state.key_to_entry.remove(&cache_key);
        }

        // Assign a temporary DataId — will be fixed on rebuild.
        let data_id = state.id_to_key.len();
        state
            .key_to_entry
            .insert(cache_key.clone(), (data_id, embedding, priority));

        // Rebuild the entire index to maintain consistency.
        state.rebuild();

        debug!(
            key = %cache_key,
            total = state.key_to_entry.len(),
            priority,
            "L2 cache insert"
        );
    }

    /// Update the eviction priority for an existing entry.
    ///
    /// Called by background workers (staleness, cost tracker) to keep priorities
    /// current without re-inserting embeddings.
    pub fn update_priority(&self, cache_key: &str, priority: f64) {
        let mut state = self.state.write();
        if let Some(entry) = state.key_to_entry.get_mut(cache_key) {
            entry.2 = priority;
        }
    }

    /// Remove a cache key from the index.
    ///
    /// Returns `true` if the key was found and removed, `false` otherwise.
    pub fn remove(&self, cache_key: &str) -> bool {
        let mut state = self.state.write();

        if state.key_to_entry.remove(cache_key).is_none() {
            return false;
        }

        // Rebuild the index without the removed entry.
        state.rebuild();

        debug!(
            key = %cache_key,
            total = state.key_to_entry.len(),
            "L2 cache remove"
        );
        true
    }

    /// Flush the entire L2 cache, removing all entries and rebuilding the index.
    pub fn flush(&self) {
        let mut state = self.state.write();
        state.key_to_entry.clear();
        state.rebuild();
        debug!("L2 cache flushed");
    }

    /// Returns the number of entries in the cache.
    pub fn len(&self) -> usize {
        self.state.read().key_to_entry.len()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the configured similarity threshold.
    pub fn similarity_threshold(&self) -> f64 {
        self.similarity_threshold
    }
}

// SAFETY: RwLock<IndexState> handles synchronization.
// Hnsw<'static, f32, DistCosine> is Send+Sync because it owns all data.
unsafe impl Send for L2SemanticCache {}
unsafe impl Sync for L2SemanticCache {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a normalized vector. Cosine similarity of identical
    /// normalized vectors is 1.0, and DistCosine distance is 0.0.
    fn normalized_vec(values: &[f32]) -> Vec<f32> {
        let norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm == 0.0 {
            return values.to_vec();
        }
        values.iter().map(|v| v / norm).collect()
    }

    #[test]
    fn test_empty_cache_returns_none() {
        let cache = L2SemanticCache::new(100, 0.92, 4);
        let query = normalized_vec(&[1.0, 0.0, 0.0, 0.0]);
        assert!(cache.search(&query).is_none());
    }

    #[test]
    fn test_exact_match_returns_hit() {
        let cache = L2SemanticCache::new(100, 0.90, 4);
        let embedding = normalized_vec(&[1.0, 2.0, 3.0, 4.0]);
        cache.insert("key1".to_string(), embedding.clone(), 1.0);

        let result = cache.search(&embedding);
        assert!(result.is_some(), "exact match should be a hit");
        let (key, similarity) = result.as_ref().unwrap();
        assert_eq!(key, "key1");
        assert!(
            *similarity > 0.99,
            "exact match similarity should be ~1.0, got {}",
            similarity
        );
    }

    #[test]
    fn test_similar_vector_returns_hit() {
        let cache = L2SemanticCache::new(100, 0.90, 4);
        let v1 = normalized_vec(&[1.0, 2.0, 3.0, 4.0]);
        // Slightly perturbed version of v1 — should still be very similar.
        let v2 = normalized_vec(&[1.01, 2.01, 3.01, 4.01]);

        cache.insert("key1".to_string(), v1, 1.0);
        let result = cache.search(&v2);
        assert!(result.is_some(), "similar vector should be a hit");
        let (key, similarity) = result.as_ref().unwrap();
        assert_eq!(key, "key1");
        assert!(*similarity > 0.95);
    }

    #[test]
    fn test_dissimilar_vector_returns_none() {
        let cache = L2SemanticCache::new(100, 0.92, 4);
        let v1 = normalized_vec(&[1.0, 0.0, 0.0, 0.0]);
        let v2 = normalized_vec(&[0.0, 0.0, 0.0, 1.0]);

        cache.insert("key1".to_string(), v1, 1.0);
        let result = cache.search(&v2);
        // Orthogonal vectors: cosine similarity ~ 0.0, well below threshold.
        assert!(result.is_none(), "orthogonal vectors should not match");
    }

    #[test]
    fn test_nearest_neighbor_among_multiple() {
        let cache = L2SemanticCache::new(100, 0.80, 4);
        let v1 = normalized_vec(&[1.0, 0.0, 0.0, 0.0]);
        let v2 = normalized_vec(&[0.0, 1.0, 0.0, 0.0]);
        let v3 = normalized_vec(&[0.0, 0.0, 1.0, 0.0]);

        cache.insert("x-axis".to_string(), v1, 1.0);
        cache.insert("y-axis".to_string(), v2, 1.0);
        cache.insert("z-axis".to_string(), v3, 1.0);

        // Query close to x-axis.
        let query = normalized_vec(&[0.95, 0.05, 0.0, 0.0]);
        let result = cache.search(&query);
        assert!(result.is_some());
        assert_eq!(result.as_ref().unwrap().0, "x-axis");
    }

    #[test]
    fn test_insert_updates_existing_key() {
        let cache = L2SemanticCache::new(100, 0.80, 4);
        let v1 = normalized_vec(&[1.0, 0.0, 0.0, 0.0]);
        let v2 = normalized_vec(&[0.0, 1.0, 0.0, 0.0]);

        cache.insert("key1".to_string(), v1, 1.0);
        assert_eq!(cache.len(), 1);

        // Update key1 to point in the y direction.
        cache.insert("key1".to_string(), v2, 1.0);
        assert_eq!(cache.len(), 1);

        // Search near y-axis should find key1.
        let query = normalized_vec(&[0.05, 0.95, 0.0, 0.0]);
        let result = cache.search(&query);
        assert!(result.is_some());
        assert_eq!(result.as_ref().unwrap().0, "key1");
    }

    #[test]
    fn test_remove_existing_key() {
        let cache = L2SemanticCache::new(100, 0.90, 4);
        let v1 = normalized_vec(&[1.0, 2.0, 3.0, 4.0]);

        cache.insert("key1".to_string(), v1.clone(), 1.0);
        assert_eq!(cache.len(), 1);

        let removed = cache.remove("key1");
        assert!(removed);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        // Search should return nothing.
        assert!(cache.search(&v1).is_none());
    }

    #[test]
    fn test_remove_nonexistent_key_returns_false() {
        let cache = L2SemanticCache::new(100, 0.92, 4);
        assert!(!cache.remove("nope"));
    }

    #[test]
    fn test_dimension_mismatch_search_returns_none() {
        let cache = L2SemanticCache::new(100, 0.92, 4);
        let v1 = normalized_vec(&[1.0, 2.0, 3.0, 4.0]);
        cache.insert("key1".to_string(), v1, 1.0);

        // Wrong dimension query.
        let bad_query = normalized_vec(&[1.0, 2.0, 3.0]);
        assert!(cache.search(&bad_query).is_none());
    }

    #[test]
    fn test_dimension_mismatch_insert_is_rejected() {
        let cache = L2SemanticCache::new(100, 0.92, 4);
        let wrong_dim = vec![1.0, 2.0, 3.0]; // dim 3, expected 4
        cache.insert("key1".to_string(), wrong_dim, 1.0);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_full_cache_evicts_lowest_priority() {
        let cache = L2SemanticCache::new(2, 0.90, 4);
        cache.insert(
            "cheap".to_string(),
            normalized_vec(&[1.0, 0.0, 0.0, 0.0]),
            1.0,
        );
        cache.insert(
            "medium".to_string(),
            normalized_vec(&[0.0, 1.0, 0.0, 0.0]),
            5.0,
        );
        assert_eq!(cache.len(), 2);

        // Insert higher-priority entry — should evict "cheap" (priority 1.0).
        cache.insert(
            "expensive".to_string(),
            normalized_vec(&[0.0, 0.0, 1.0, 0.0]),
            10.0,
        );
        assert_eq!(cache.len(), 2);

        // "cheap" should be gone, "medium" and "expensive" remain.
        let query_cheap = normalized_vec(&[1.0, 0.0, 0.0, 0.0]);
        assert!(
            cache.search(&query_cheap).is_none(),
            "evicted entry should not be found"
        );

        let query_medium = normalized_vec(&[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(cache.search(&query_medium).unwrap().0, "medium");

        let query_expensive = normalized_vec(&[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(cache.search(&query_expensive).unwrap().0, "expensive");
    }

    #[test]
    fn test_full_cache_rejects_low_priority_insert() {
        let cache = L2SemanticCache::new(2, 0.90, 4);
        cache.insert("a".to_string(), normalized_vec(&[1.0, 0.0, 0.0, 0.0]), 5.0);
        cache.insert("b".to_string(), normalized_vec(&[0.0, 1.0, 0.0, 0.0]), 10.0);
        assert_eq!(cache.len(), 2);

        // Insert with lower priority than both — should be rejected.
        cache.insert("c".to_string(), normalized_vec(&[0.0, 0.0, 1.0, 0.0]), 1.0);
        assert_eq!(cache.len(), 2);

        // Both original entries should still be present.
        let query_a = normalized_vec(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(cache.search(&query_a).unwrap().0, "a");
    }

    #[test]
    fn test_full_cache_allows_update_of_existing_key() {
        let cache = L2SemanticCache::new(2, 0.90, 4);
        cache.insert("a".to_string(), normalized_vec(&[1.0, 0.0, 0.0, 0.0]), 1.0);
        cache.insert("b".to_string(), normalized_vec(&[0.0, 1.0, 0.0, 0.0]), 1.0);

        // Updating existing key should work even when full.
        cache.insert("a".to_string(), normalized_vec(&[0.0, 0.0, 1.0, 0.0]), 1.0);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_update_priority() {
        let cache = L2SemanticCache::new(2, 0.90, 4);
        cache.insert(
            "cheap".to_string(),
            normalized_vec(&[1.0, 0.0, 0.0, 0.0]),
            1.0,
        );
        cache.insert(
            "medium".to_string(),
            normalized_vec(&[0.0, 1.0, 0.0, 0.0]),
            5.0,
        );

        // Boost "cheap" priority above "medium".
        cache.update_priority("cheap", 20.0);

        // Now inserting a new entry should evict "medium" (priority 5.0), not "cheap" (priority 20.0).
        cache.insert(
            "new".to_string(),
            normalized_vec(&[0.0, 0.0, 1.0, 0.0]),
            10.0,
        );
        assert_eq!(cache.len(), 2);

        let query_cheap = normalized_vec(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(cache.search(&query_cheap).unwrap().0, "cheap");

        let query_medium = normalized_vec(&[0.0, 1.0, 0.0, 0.0]);
        assert!(
            cache.search(&query_medium).is_none(),
            "medium should have been evicted"
        );
    }

    #[test]
    fn test_with_defaults() {
        let cache = L2SemanticCache::with_defaults(100, 384);
        assert!((cache.similarity_threshold() - 0.92).abs() < f64::EPSILON);
    }

    #[test]
    fn test_len_and_is_empty() {
        let cache = L2SemanticCache::new(100, 0.92, 4);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.insert("k".to_string(), normalized_vec(&[1.0, 0.0, 0.0, 0.0]), 1.0);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_thread_safety_concurrent_reads() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(L2SemanticCache::new(1000, 0.90, 4));

        // Insert some data first.
        for i in 0..10 {
            let v = normalized_vec(&[i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]);
            cache.insert(format!("key{}", i), v, 1.0);
        }

        // Spawn multiple reader threads.
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let cache = Arc::clone(&cache);
                thread::spawn(move || {
                    let query =
                        normalized_vec(&[i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]);
                    for _ in 0..100 {
                        let _ = cache.search(&query);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("reader thread panicked");
        }
    }

    #[test]
    fn test_remove_and_search_consistency() {
        let cache = L2SemanticCache::new(100, 0.80, 4);
        let v1 = normalized_vec(&[1.0, 0.0, 0.0, 0.0]);
        let v2 = normalized_vec(&[0.0, 1.0, 0.0, 0.0]);

        cache.insert("first".to_string(), v1, 1.0);
        cache.insert("second".to_string(), v2, 1.0);

        // Remove first, search should only find second.
        cache.remove("first");

        let query = normalized_vec(&[0.0, 1.0, 0.0, 0.0]);
        let result = cache.search(&query);
        assert!(result.is_some());
        assert_eq!(result.as_ref().unwrap().0, "second");
    }
}
