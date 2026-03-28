use chrono::Utc;

use crate::CacheEntry;

/// Compute the eviction priority for a cache entry.
///
/// ```text
/// priority = (token_cost × hit_frequency × recency_score)
///            ─────────────────────────────────────────────
///               staleness_risk × embedding_uncertainty
/// ```
///
/// Higher priority = more worth keeping. Evict lowest priority first.
pub fn eviction_priority(entry: &CacheEntry) -> f64 {
    let hit_frequency = (entry.hit_count as f64) + 1.0; // +1 to avoid zero

    // Recency score: exponential decay. Half-life = 24 hours.
    let age_hours = (Utc::now() - entry.last_hit_at).num_seconds().max(0) as f64 / 3600.0;
    let recency_score = (-age_hours / 24.0_f64).exp();

    let numerator = entry.generation_cost * hit_frequency * recency_score;

    // Clamp denominators to avoid division by zero.
    let staleness = entry.staleness_risk.max(0.01);
    let uncertainty = entry.embedding_uncertainty.max(0.01);

    numerator / (staleness * uncertainty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_expensive_entries_have_higher_priority() {
        let cheap = CacheEntry::new(json!({}), 0.001);
        let expensive = CacheEntry::new(json!({}), 1.0);

        assert!(eviction_priority(&expensive) > eviction_priority(&cheap));
    }

    #[test]
    fn test_stale_entries_have_lower_priority() {
        let mut fresh = CacheEntry::new(json!({}), 0.5);
        fresh.staleness_risk = 0.1;

        let mut stale = CacheEntry::new(json!({}), 0.5);
        stale.staleness_risk = 0.9;

        assert!(eviction_priority(&fresh) > eviction_priority(&stale));
    }
}
