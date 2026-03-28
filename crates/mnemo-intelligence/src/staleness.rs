//! Staleness detection worker for proactive cache eviction.
//!
//! Runs as a background tokio task on the **cold path**. Periodically scans cached
//! entries, scores them for freshness, and emits [`StalenessUpdate`] messages when
//! an entry's staleness risk exceeds a threshold.
//!
//! # Signals
//!
//! 1. **Temporal decay** — entries get staler over time (power-law decay, ~1 week half-life).
//! 2. **Hit pattern anomaly** — an entry that stopped receiving hits is suspicious.
//! 3. **Correction signals** — external feedback that a cached response was wrong.
//! 4. **Response drift** — the upstream now returns a different answer for the same prompt.
//! 5. **Semantic drift** — similar queries clustering differently (future, stubbed).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Staleness signals & risk computation
// ---------------------------------------------------------------------------

/// Observable signals for a single cache entry used to compute staleness risk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StalenessSignals {
    /// How old the cached entry is, in hours.
    pub age_hours: f64,
    /// Hours since the entry was last hit by a request.
    pub hours_since_last_hit: f64,
    /// Total number of cache hits the entry has received.
    pub hit_count: u64,
    /// Whether external correction feedback has been recorded.
    pub was_corrected: bool,
    /// Response drift score in `[0.0, 1.0]`: 0 = identical, 1 = completely different.
    pub response_drift_score: f64,
}

/// Compute the overall staleness risk from signals.
///
/// Returns a value clamped to `[0.0, 1.0]`. Higher means staler.
///
/// # Formula
///
/// ```text
/// temporal_decay  = 1.0 - exp(-age_hours / 168.0)          // ~1 week half-life
/// hit_decay       = if hours_since_last_hit > 24 { min(hours_since_last_hit / 168.0, 1.0) } else { 0.0 }
/// correction_wt   = if was_corrected { 0.8 } else { 0.0 }
/// drift_wt        = response_drift_score * 0.9
/// risk            = clamp(temporal_decay * 0.3 + hit_decay * 0.2 + correction_wt + drift_wt, 0.0, 1.0)
/// ```
pub fn compute_staleness_risk(signals: &StalenessSignals) -> f64 {
    let temporal_decay = 1.0 - (-signals.age_hours / 168.0_f64).exp();

    let hit_decay = if signals.hours_since_last_hit > 24.0 {
        (signals.hours_since_last_hit / 168.0).min(1.0)
    } else {
        0.0
    };

    let correction_weight = if signals.was_corrected { 0.8 } else { 0.0 };

    let drift_weight = signals.response_drift_score * 0.9;

    let raw = temporal_decay * 0.3 + hit_decay * 0.2 + correction_weight + drift_weight;
    raw.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Update message
// ---------------------------------------------------------------------------

/// Message emitted by the staleness worker when a cache entry's risk changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StalenessUpdate {
    /// The cache key whose staleness risk was recomputed.
    pub cache_key: String,
    /// The new overall staleness risk in `[0.0, 1.0]`.
    pub staleness_risk: f64,
}

// ---------------------------------------------------------------------------
// Per-entry bookkeeping stored inside the worker
// ---------------------------------------------------------------------------

/// Internal record the worker maintains for each cache key it knows about.
#[derive(Debug, Clone)]
struct EntryRecord {
    was_corrected: bool,
    drift_score: f64,
}

impl Default for EntryRecord {
    fn default() -> Self {
        Self {
            was_corrected: false,
            drift_score: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// StalenessWorker
// ---------------------------------------------------------------------------

/// Background worker that periodically recomputes staleness risk for cached entries.
///
/// The worker itself does **not** own the cache — it receives signals via
/// [`record_correction`](Self::record_correction) and [`record_drift`](Self::record_drift),
/// and emits [`StalenessUpdate`] messages on the channel provided to [`spawn`](Self::spawn).
///
/// Cache-side metadata (age, hit count, etc.) must be supplied through the
/// [`SignalProvider`] trait so the worker can pull current stats during each scan.
pub struct StalenessWorker {
    scan_interval: Duration,
    records: Arc<RwLock<HashMap<String, EntryRecord>>>,
}

/// Trait that the cache layer implements so the staleness worker can pull
/// per-entry metadata during scans without owning the cache itself.
///
/// A default no-op implementation is provided for testing.
pub trait SignalProvider: Send + Sync + 'static {
    /// Return the current signals for the given cache key, or `None` if the
    /// entry no longer exists.
    fn signals_for(&self, cache_key: &str) -> Option<StalenessSignals>;

    /// Return all cache keys that should be scanned.
    fn all_keys(&self) -> Vec<String>;
}

impl StalenessWorker {
    /// Create a new staleness worker.
    ///
    /// `scan_interval_secs` controls how often the background loop wakes up
    /// to re-score entries.
    pub fn new(scan_interval_secs: u64) -> Self {
        Self {
            scan_interval: Duration::from_secs(scan_interval_secs),
            records: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Spawn the background scanning loop.
    ///
    /// The worker will periodically pull signals from the `provider`, merge
    /// them with its own bookkeeping (corrections, drift observations), and
    /// send a [`StalenessUpdate`] for every entry whose risk exceeds
    /// `eviction_threshold`.
    pub fn spawn<P: SignalProvider>(
        self,
        update_tx: mpsc::Sender<StalenessUpdate>,
        provider: P,
        eviction_threshold: f64,
    ) -> JoinHandle<()> {
        let records = Arc::clone(&self.records);
        let interval = self.scan_interval;

        tokio::spawn(async move {
            info!(
                interval_secs = interval.as_secs(),
                threshold = eviction_threshold,
                "staleness worker started"
            );

            let mut ticker = tokio::time::interval(interval);
            // The first tick completes immediately — skip it so we don't scan
            // on an empty cache right at startup.
            ticker.tick().await;

            loop {
                ticker.tick().await;
                Self::scan_once(&records, &update_tx, &provider, eviction_threshold).await;
            }
        })
    }

    /// Record that external feedback marked a cached response as incorrect.
    pub fn record_correction(&self, cache_key: &str) {
        let mut map = self.records.write();
        map.entry(cache_key.to_owned()).or_default().was_corrected = true;
        debug!(cache_key, "recorded correction signal");
    }

    /// Record a response-drift observation for a cache key.
    ///
    /// `drift_score` should be in `[0.0, 1.0]`. Values outside that range
    /// are clamped.
    pub fn record_drift(&self, cache_key: &str, drift_score: f64) {
        let clamped = drift_score.clamp(0.0, 1.0);
        let mut map = self.records.write();
        map.entry(cache_key.to_owned()).or_default().drift_score = clamped;
        debug!(
            cache_key,
            drift_score = clamped,
            "recorded drift observation"
        );
    }

    /// Returns a snapshot of the internal records (useful for diagnostics).
    pub fn record_count(&self) -> usize {
        self.records.read().len()
    }

    // -- internal -----------------------------------------------------------

    async fn scan_once<P: SignalProvider>(
        records: &Arc<RwLock<HashMap<String, EntryRecord>>>,
        update_tx: &mpsc::Sender<StalenessUpdate>,
        provider: &P,
        eviction_threshold: f64,
    ) {
        let keys = provider.all_keys();
        let num_keys = keys.len();
        let mut eviction_candidates = 0u64;

        for key in &keys {
            // Pull base signals from the cache layer.
            let Some(mut signals) = provider.signals_for(key) else {
                // Entry was removed between all_keys() and now — skip.
                continue;
            };

            // Overlay worker-side bookkeeping (corrections, drift).
            {
                let map = records.read();
                if let Some(rec) = map.get(key.as_str()) {
                    if rec.was_corrected {
                        signals.was_corrected = true;
                    }
                    // Take the max of provider-reported drift and locally recorded drift.
                    if rec.drift_score > signals.response_drift_score {
                        signals.response_drift_score = rec.drift_score;
                    }
                }
            }

            let risk = compute_staleness_risk(&signals);

            if risk >= eviction_threshold {
                eviction_candidates += 1;
                let update = StalenessUpdate {
                    cache_key: key.clone(),
                    staleness_risk: risk,
                };
                if let Err(e) = update_tx.send(update).await {
                    warn!(
                        error = %e,
                        "staleness update channel closed; stopping scan"
                    );
                    return;
                }
            }
        }

        debug!(
            scanned = num_keys,
            eviction_candidates, "staleness scan complete"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- compute_staleness_risk unit tests ----------------------------------

    #[test]
    fn fresh_entry_has_low_risk() {
        let signals = StalenessSignals {
            age_hours: 1.0,
            hours_since_last_hit: 0.5,
            hit_count: 10,
            was_corrected: false,
            response_drift_score: 0.0,
        };
        let risk = compute_staleness_risk(&signals);
        assert!(
            risk < 0.05,
            "fresh entry risk should be very low, got {risk}"
        );
    }

    #[test]
    fn week_old_entry_moderate_risk() {
        let signals = StalenessSignals {
            age_hours: 168.0,
            hours_since_last_hit: 2.0,
            hit_count: 50,
            was_corrected: false,
            response_drift_score: 0.0,
        };
        let risk = compute_staleness_risk(&signals);
        // temporal_decay = 1 - exp(-1) ≈ 0.632, * 0.3 ≈ 0.19
        assert!(
            (0.15..0.25).contains(&risk),
            "week-old entry risk should be moderate, got {risk}"
        );
    }

    #[test]
    fn corrected_entry_high_risk() {
        let signals = StalenessSignals {
            age_hours: 1.0,
            hours_since_last_hit: 0.5,
            hit_count: 10,
            was_corrected: true,
            response_drift_score: 0.0,
        };
        let risk = compute_staleness_risk(&signals);
        assert!(
            risk >= 0.8,
            "corrected entry should have high risk, got {risk}"
        );
    }

    #[test]
    fn full_drift_caps_at_one() {
        let signals = StalenessSignals {
            age_hours: 500.0,
            hours_since_last_hit: 500.0,
            hit_count: 0,
            was_corrected: true,
            response_drift_score: 1.0,
        };
        let risk = compute_staleness_risk(&signals);
        assert!(
            (risk - 1.0).abs() < f64::EPSILON,
            "max signals should clamp to 1.0, got {risk}"
        );
    }

    #[test]
    fn zero_signals_zero_risk() {
        let signals = StalenessSignals {
            age_hours: 0.0,
            hours_since_last_hit: 0.0,
            hit_count: 0,
            was_corrected: false,
            response_drift_score: 0.0,
        };
        let risk = compute_staleness_risk(&signals);
        assert!(
            risk.abs() < f64::EPSILON,
            "all-zero signals should give 0.0 risk, got {risk}"
        );
    }

    #[test]
    fn hit_decay_kicks_in_after_24h() {
        let base = StalenessSignals {
            age_hours: 0.0,
            hours_since_last_hit: 23.0,
            hit_count: 5,
            was_corrected: false,
            response_drift_score: 0.0,
        };
        let above = StalenessSignals {
            hours_since_last_hit: 48.0,
            ..base.clone()
        };
        let risk_below = compute_staleness_risk(&base);
        let risk_above = compute_staleness_risk(&above);
        assert!(
            risk_above > risk_below,
            "hit decay should increase risk after 24h: below={risk_below}, above={risk_above}"
        );
    }

    #[test]
    fn drift_score_proportional() {
        let low = StalenessSignals {
            age_hours: 0.0,
            hours_since_last_hit: 0.0,
            hit_count: 10,
            was_corrected: false,
            response_drift_score: 0.1,
        };
        let high = StalenessSignals {
            response_drift_score: 0.9,
            ..low.clone()
        };
        let r_low = compute_staleness_risk(&low);
        let r_high = compute_staleness_risk(&high);
        assert!(
            r_high > r_low,
            "higher drift should give higher risk: low={r_low}, high={r_high}"
        );
    }

    // -- worker integration tests -------------------------------------------

    /// Minimal signal provider for tests.
    struct TestProvider {
        entries: HashMap<String, StalenessSignals>,
    }

    impl SignalProvider for TestProvider {
        fn signals_for(&self, cache_key: &str) -> Option<StalenessSignals> {
            self.entries.get(cache_key).cloned()
        }

        fn all_keys(&self) -> Vec<String> {
            self.entries.keys().cloned().collect()
        }
    }

    #[tokio::test]
    async fn worker_emits_update_for_stale_entry() {
        let mut entries = HashMap::new();
        entries.insert(
            "stale-key".to_owned(),
            StalenessSignals {
                age_hours: 500.0,
                hours_since_last_hit: 200.0,
                hit_count: 1,
                was_corrected: false,
                response_drift_score: 0.0,
            },
        );
        let provider = TestProvider { entries };

        let (tx, mut rx) = mpsc::channel::<StalenessUpdate>(16);
        let worker = StalenessWorker::new(1);

        // Use scan_once directly so the test is deterministic.
        StalenessWorker::scan_once(&worker.records, &tx, &provider, 0.3).await;

        let update = rx.try_recv().expect("should have received an update");
        assert_eq!(update.cache_key, "stale-key");
        assert!(update.staleness_risk >= 0.3);
    }

    #[tokio::test]
    async fn worker_skips_fresh_entry() {
        let mut entries = HashMap::new();
        entries.insert(
            "fresh-key".to_owned(),
            StalenessSignals {
                age_hours: 1.0,
                hours_since_last_hit: 0.5,
                hit_count: 100,
                was_corrected: false,
                response_drift_score: 0.0,
            },
        );
        let provider = TestProvider { entries };

        let (tx, mut rx) = mpsc::channel::<StalenessUpdate>(16);
        let worker = StalenessWorker::new(1);

        StalenessWorker::scan_once(&worker.records, &tx, &provider, 0.3).await;

        assert!(
            rx.try_recv().is_err(),
            "fresh entry should not trigger an update"
        );
    }

    #[tokio::test]
    async fn correction_signal_overlays_provider() {
        let mut entries = HashMap::new();
        entries.insert(
            "corrected-key".to_owned(),
            StalenessSignals {
                age_hours: 1.0,
                hours_since_last_hit: 0.5,
                hit_count: 100,
                was_corrected: false, // provider says no correction
                response_drift_score: 0.0,
            },
        );
        let provider = TestProvider { entries };

        let (tx, mut rx) = mpsc::channel::<StalenessUpdate>(16);
        let worker = StalenessWorker::new(1);

        // Record a correction through the worker API.
        worker.record_correction("corrected-key");

        StalenessWorker::scan_once(&worker.records, &tx, &provider, 0.3).await;

        let update = rx
            .try_recv()
            .expect("corrected entry should trigger update");
        assert_eq!(update.cache_key, "corrected-key");
        assert!(
            update.staleness_risk >= 0.8,
            "corrected entry risk should be >= 0.8, got {}",
            update.staleness_risk
        );
    }

    #[tokio::test]
    async fn drift_signal_overlays_provider() {
        let mut entries = HashMap::new();
        entries.insert(
            "drifted-key".to_owned(),
            StalenessSignals {
                age_hours: 1.0,
                hours_since_last_hit: 0.5,
                hit_count: 100,
                was_corrected: false,
                response_drift_score: 0.0, // provider sees no drift
            },
        );
        let provider = TestProvider { entries };

        let (tx, mut rx) = mpsc::channel::<StalenessUpdate>(16);
        let worker = StalenessWorker::new(1);

        // Record high drift through the worker API.
        worker.record_drift("drifted-key", 0.95);

        StalenessWorker::scan_once(&worker.records, &tx, &provider, 0.3).await;

        let update = rx.try_recv().expect("drifted entry should trigger update");
        assert_eq!(update.cache_key, "drifted-key");
        assert!(
            update.staleness_risk > 0.8,
            "high-drift entry risk should be > 0.8, got {}",
            update.staleness_risk
        );
    }

    #[test]
    fn record_drift_clamps_values() {
        let worker = StalenessWorker::new(60);
        worker.record_drift("key", 2.5);
        {
            let map = worker.records.read();
            let rec = map.get("key").expect("record should exist");
            assert!(
                (rec.drift_score - 1.0).abs() < f64::EPSILON,
                "drift should be clamped to 1.0"
            );
        }

        worker.record_drift("key", -0.5);
        {
            let map = worker.records.read();
            let rec = map.get("key").expect("record should exist");
            assert!(
                rec.drift_score.abs() < f64::EPSILON,
                "drift should be clamped to 0.0"
            );
        }
    }
}
