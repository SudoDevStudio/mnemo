use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A cached response entry. Every entry — LLM, MCP, or ACP — carries a
/// generation_cost field (design rule #5: cost is a first-class citizen).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The cached response body.
    pub response: serde_json::Value,
    /// Actual dollar cost when this response was generated.
    pub generation_cost: f64,
    /// Number of times this entry has been served.
    pub hit_count: u64,
    /// When this entry was created.
    pub created_at: DateTime<Utc>,
    /// When this entry was last served.
    pub last_hit_at: DateTime<Utc>,
    /// Staleness risk score (0.0 = fresh, 1.0 = stale). Updated async.
    pub staleness_risk: f64,
    /// Embedding uncertainty score. Updated async.
    pub embedding_uncertainty: f64,
}

impl CacheEntry {
    pub fn new(response: serde_json::Value, generation_cost: f64) -> Self {
        let now = Utc::now();
        Self {
            response,
            generation_cost,
            hit_count: 0,
            created_at: now,
            last_hit_at: now,
            staleness_risk: 0.0,
            embedding_uncertainty: 1.0,
        }
    }

    /// Record a cache hit.
    pub fn record_hit(&mut self) {
        self.hit_count += 1;
        self.last_hit_at = Utc::now();
    }
}
