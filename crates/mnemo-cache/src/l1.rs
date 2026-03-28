use dashmap::DashMap;
use moka::future::Cache;
use std::sync::Arc;

use crate::CacheEntry;

/// L1 exact-match cache layer.
/// Uses DashMap for O(1) concurrent lookups and moka for W-TinyLFU eviction.
#[derive(Clone)]
pub struct L1Cache {
    /// The main cache with W-TinyLFU eviction policy.
    store: Cache<String, Arc<CacheEntry>>,
    /// Concurrent map for fast exact lookups + mutable hit tracking.
    hits: Arc<DashMap<String, u64>>,
}

impl L1Cache {
    pub fn new(max_entries: u64) -> Self {
        let hits: Arc<DashMap<String, u64>> = Arc::new(DashMap::new());
        let hits_for_listener = Arc::clone(&hits);

        let store = Cache::builder()
            .max_capacity(max_entries)
            .eviction_listener(move |key: Arc<String>, _value, _cause| {
                hits_for_listener.remove(key.as_ref());
            })
            .build();

        Self { store, hits }
    }

    /// Look up an exact cache key. Returns the entry if found.
    /// Also records the hit on the CacheEntry (updates hit_count and last_hit_at)
    /// so that eviction priority stays accurate.
    pub async fn get(&self, key: &str) -> Option<Arc<CacheEntry>> {
        if let Some(entry) = self.store.get(key).await {
            // Track hit count in the DashMap
            self.hits
                .entry(key.to_string())
                .and_modify(|c| *c += 1)
                .or_insert(1);
            // Update the CacheEntry's own hit metadata so eviction priority
            // reflects actual usage. Clone-mutate-reinsert is cheap because
            // moka just updates the value for the same key.
            let mut updated = (*entry).clone();
            updated.record_hit();
            let updated = Arc::new(updated);
            self.store
                .insert(key.to_string(), Arc::clone(&updated))
                .await;
            Some(updated)
        } else {
            None
        }
    }

    /// Read-only lookup that does NOT record a hit or update metadata.
    /// Use this from background workers (staleness, eviction) that need to
    /// inspect entries without perturbing cache statistics.
    pub async fn peek(&self, key: &str) -> Option<Arc<CacheEntry>> {
        self.store.get(key).await
    }

    /// Insert a new cache entry.
    pub async fn insert(&self, key: String, entry: CacheEntry) {
        self.store.insert(key, Arc::new(entry)).await;
    }

    /// Remove a cache entry.
    pub async fn invalidate(&self, key: &str) {
        self.store.invalidate(key).await;
        self.hits.remove(key);
    }

    /// Flush the entire cache.
    pub async fn flush(&self) {
        self.store.invalidate_all();
        self.hits.clear();
    }

    /// Current number of entries.
    pub fn entry_count(&self) -> u64 {
        self.store.entry_count()
    }

    /// Get hit count for a specific key.
    pub fn hit_count(&self, key: &str) -> u64 {
        self.hits.get(key).map(|v| *v).unwrap_or(0)
    }

    /// Return all keys that have been tracked (hit at least once).
    pub fn tracked_keys(&self) -> Vec<String> {
        self.hits.iter().map(|entry| entry.key().clone()).collect()
    }
}
