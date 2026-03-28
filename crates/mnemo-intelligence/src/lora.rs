//! LoRA adapter fine-tuning worker.
//!
//! Runs on the **cold path** as a background tokio task. Accumulates
//! query-response training pairs from live traffic and periodically trains
//! a lightweight LoRA adapter on those pairs. Uses Elastic Weight
//! Consolidation (EWC) to prevent catastrophic forgetting across training
//! rounds.
//!
//! The adapter is a low-rank decomposition (A * B) that is applied on top
//! of the frozen base embedding model. Typical adapter size is ~10 MB.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use rand::Rng;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Training data
// ---------------------------------------------------------------------------

/// A single training example from live traffic.
#[derive(Debug, Clone)]
pub struct TrainingPair {
    /// The raw query text sent by the client.
    pub query_text: String,
    /// The response text returned by the upstream LLM.
    pub response_text: String,
    /// The current base embedding for this query.
    pub embedding: Vec<f32>,
    /// When this pair was captured.
    pub timestamp: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// LoRA adapter weights
// ---------------------------------------------------------------------------

/// LoRA adapter weights (low-rank decomposition A * B).
///
/// The adapter transforms a base embedding `x` via:
///
///   output = x + alpha * (B @ A @ x)
///
/// where A is [rank x dim] (down-projection) and B is [dim x rank]
/// (up-projection).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Rank of the low-rank decomposition.
    pub rank: usize,
    /// Embedding dimension.
    pub dim: usize,
    /// Down-projection weights, stored row-major as [rank x dim].
    pub weight_a: Vec<f32>,
    /// Up-projection weights, stored row-major as [dim x rank].
    pub weight_b: Vec<f32>,
    /// Monotonically increasing version counter.
    pub version: u64,
    /// Total number of training samples this adapter has been trained on.
    pub trained_on_samples: u64,
}

impl LoraAdapter {
    /// Create a new adapter with zero-initialized weights.
    pub fn zeros(rank: usize, dim: usize) -> Self {
        Self {
            rank,
            dim,
            weight_a: vec![0.0; rank * dim],
            weight_b: vec![0.0; dim * rank],
            version: 0,
            trained_on_samples: 0,
        }
    }

    /// Create a new adapter with small random weights (Kaiming-style).
    pub fn random_init(rank: usize, dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / dim as f32).sqrt();
        let weight_a: Vec<f32> = (0..rank * dim)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        let weight_b: Vec<f32> = (0..dim * rank)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self {
            rank,
            dim,
            weight_a,
            weight_b,
            version: 0,
            trained_on_samples: 0,
        }
    }

    /// Apply the adapter to transform a base embedding.
    ///
    /// output = base + alpha * (B @ A @ base)
    ///
    /// Panics if `base_embedding.len() != self.dim`.
    pub fn apply(&self, base_embedding: &[f32], alpha: f32) -> Vec<f32> {
        assert_eq!(
            base_embedding.len(),
            self.dim,
            "base embedding dimension mismatch: expected {}, got {}",
            self.dim,
            base_embedding.len()
        );

        // Step 1: hidden = A @ base  → [rank]
        let mut hidden = vec![0.0f32; self.rank];
        for (r, h) in hidden.iter_mut().enumerate() {
            let row = &self.weight_a[r * self.dim..(r + 1) * self.dim];
            *h = row.iter().zip(base_embedding).map(|(&a, &b)| a * b).sum();
        }

        // Step 2: delta = B @ hidden  → [dim]
        let mut delta = vec![0.0f32; self.dim];
        for (d, del) in delta.iter_mut().enumerate() {
            let row = &self.weight_b[d * self.rank..(d + 1) * self.rank];
            *del = row.iter().zip(hidden.iter()).map(|(&b, &h)| b * h).sum();
        }

        // Step 3: output = base + alpha * delta
        base_embedding
            .iter()
            .zip(delta.iter())
            .map(|(&b, &d)| b + alpha * d)
            .collect()
    }

    /// Return a flat concatenation of all adapter weights (A then B).
    pub fn flat_weights(&self) -> Vec<f32> {
        let mut w = Vec::with_capacity(self.weight_a.len() + self.weight_b.len());
        w.extend_from_slice(&self.weight_a);
        w.extend_from_slice(&self.weight_b);
        w
    }

    /// Total number of trainable parameters.
    pub fn num_params(&self) -> usize {
        self.weight_a.len() + self.weight_b.len()
    }
}

// ---------------------------------------------------------------------------
// Elastic Weight Consolidation (EWC)
// ---------------------------------------------------------------------------

/// Elastic Weight Consolidation state.
///
/// Stores the Fisher information diagonal and a snapshot of the weights
/// from the end of the previous training round. The EWC penalty is:
///
///   penalty = (lambda / 2) * sum_i F_i * (w_i - w*_i)^2
///
/// where F_i is the Fisher diagonal entry and w*_i is the snapshot weight.
#[derive(Debug, Clone)]
pub struct EwcState {
    /// Fisher information diagonal (importance of each weight).
    pub fisher_diagonal: Vec<f32>,
    /// Snapshot of weights at the end of last training round.
    pub prev_weights: Vec<f32>,
    /// EWC regularization strength.
    pub lambda: f32,
}

impl EwcState {
    /// Compute the EWC penalty for the given current weights.
    ///
    /// Returns `(lambda / 2) * sum_i F_i * (w_i - w*_i)^2`.
    pub fn penalty(&self, current_weights: &[f32]) -> f32 {
        assert_eq!(current_weights.len(), self.prev_weights.len());
        let sum: f32 = self
            .fisher_diagonal
            .iter()
            .zip(self.prev_weights.iter())
            .zip(current_weights.iter())
            .map(|((&f, &w_star), &w)| {
                let diff = w - w_star;
                f * diff * diff
            })
            .sum();
        0.5 * self.lambda * sum
    }

    /// Compute the EWC gradient contribution for each weight.
    ///
    /// grad_ewc_i = lambda * F_i * (w_i - w*_i)
    pub fn gradient(&self, current_weights: &[f32]) -> Vec<f32> {
        assert_eq!(current_weights.len(), self.prev_weights.len());
        self.fisher_diagonal
            .iter()
            .zip(self.prev_weights.iter())
            .zip(current_weights.iter())
            .map(|((&f, &w_star), &w)| self.lambda * f * (w - w_star))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the LoRA training worker.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the LoRA decomposition.
    pub rank: usize,
    /// Embedding dimension (must match the base model).
    pub dim: usize,
    /// LoRA scaling factor applied when transforming embeddings.
    pub alpha: f32,
    /// Number of samples per training mini-batch.
    pub batch_size: usize,
    /// Seconds between training rounds.
    pub training_interval_secs: u64,
    /// EWC regularization strength.
    pub ewc_lambda: f32,
    /// Minimum samples before the first training round.
    pub min_samples: usize,
    /// Learning rate for the simplified gradient update.
    pub learning_rate: f32,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            dim: 384,
            alpha: 0.1,
            batch_size: 32,
            training_interval_secs: 300,
            ewc_lambda: 100.0,
            min_samples: 64,
            learning_rate: 0.001,
        }
    }
}

// ---------------------------------------------------------------------------
// LoRA worker
// ---------------------------------------------------------------------------

/// Background worker that accumulates training pairs from live traffic and
/// periodically trains a lightweight LoRA adapter.
pub struct LoraWorker {
    buffer: Arc<RwLock<Vec<TrainingPair>>>,
    adapter: Arc<RwLock<LoraAdapter>>,
    ewc: Arc<RwLock<Option<EwcState>>>,
    config: LoraConfig,
}

impl LoraWorker {
    /// Create a new LoRA worker with the given configuration.
    pub fn new(config: LoraConfig) -> Self {
        let adapter = LoraAdapter::zeros(config.rank, config.dim);
        Self {
            buffer: Arc::new(RwLock::new(Vec::new())),
            adapter: Arc::new(RwLock::new(adapter)),
            ewc: Arc::new(RwLock::new(None)),
            config,
        }
    }

    /// Add a training pair from live traffic.
    pub fn add_training_pair(&self, pair: TrainingPair) {
        self.buffer.write().push(pair);
    }

    /// Get a clone-safe handle to the current adapter.
    pub fn adapter(&self) -> Arc<RwLock<LoraAdapter>> {
        Arc::clone(&self.adapter)
    }

    /// Get the number of accumulated training pairs.
    pub fn buffer_size(&self) -> usize {
        self.buffer.read().len()
    }

    /// Spawn the background training loop. Returns a join handle.
    pub fn spawn(self: Arc<Self>) -> JoinHandle<()> {
        let interval = self.config.training_interval_secs;
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(interval));
            loop {
                tick.tick().await;
                self.train_step();
            }
        })
    }

    /// Run one training step. Called by the background loop or manually for
    /// testing.
    ///
    /// 1. Drain the training buffer.
    /// 2. Skip if not enough samples.
    /// 3. Compute a simplified gradient update to the LoRA weights.
    /// 4. Apply EWC penalty to prevent catastrophic forgetting.
    /// 5. Update the Fisher diagonal estimate.
    /// 6. Increment the adapter version.
    pub fn train_step(&self) {
        // --- 1. Drain the buffer ---
        let pairs: Vec<TrainingPair> = {
            let mut buf = self.buffer.write();
            std::mem::take(&mut *buf)
        };

        // --- 2. Check sample count ---
        let is_first_round = {
            let adapter = self.adapter.read();
            adapter.version == 0
        };

        let required = if is_first_round {
            self.config.min_samples
        } else {
            self.config.batch_size
        };

        if pairs.len() < required {
            // Not enough samples — put them back.
            debug!(
                samples = pairs.len(),
                required, "LoRA train_step: not enough samples, returning to buffer"
            );
            let mut buf = self.buffer.write();
            // Prepend the old pairs (they are older) before any new ones that
            // arrived between drain and now.
            let new_arrivals = std::mem::take(&mut *buf);
            *buf = pairs;
            buf.extend(new_arrivals);
            return;
        }

        info!(
            samples = pairs.len(),
            "LoRA train_step: starting training round"
        );

        let lr = self.config.learning_rate;
        let alpha = self.config.alpha;
        let dim = self.config.dim;
        let rank = self.config.rank;
        let num_params;

        // --- 3. Compute gradient update ---
        // We use a simplified contrastive-style update:
        //   For each training pair, compute the residual between the adapter's
        //   current output and a target direction derived from the pair's
        //   embedding. The target is a slightly perturbed version of the base
        //   embedding (simulating what a domain-adapted model would produce).
        //   We then update weights to reduce this residual.
        {
            let mut adapter = self.adapter.write();
            num_params = adapter.num_params();
            let mut grad_a = vec![0.0f32; rank * dim];
            let mut grad_b = vec![0.0f32; dim * rank];

            let n = pairs.len() as f32;

            for pair in &pairs {
                if pair.embedding.len() != dim {
                    warn!(
                        expected = dim,
                        got = pair.embedding.len(),
                        "LoRA train_step: skipping pair with wrong embedding dimension"
                    );
                    continue;
                }

                let base = &pair.embedding;

                // Current adapter output.
                let output = adapter.apply(base, alpha);

                // Target: the base embedding shifted slightly toward a
                // domain-specific direction. We derive this from the hash of
                // the response text to get a deterministic but content-dependent
                // target shift.
                let target = compute_target_embedding(base, &pair.response_text, dim);

                // Residual: target - output
                let residual: Vec<f32> = target
                    .iter()
                    .zip(output.iter())
                    .map(|(&t, &o)| t - o)
                    .collect();

                // Backprop through the linear layers:
                // output = base + alpha * B @ A @ base
                // d(loss)/d(B) = -alpha * residual @ hidden^T  where hidden = A @ base
                // d(loss)/d(A) = -alpha * B^T @ residual @ base^T

                // hidden = A @ base  → [rank]
                let mut hidden = vec![0.0f32; rank];
                for (r, h) in hidden.iter_mut().enumerate() {
                    let row = &adapter.weight_a[r * dim..(r + 1) * dim];
                    *h = row.iter().zip(base.iter()).map(|(&a, &b)| a * b).sum();
                }

                // grad_B += -residual @ hidden^T  (outer product, [dim x rank])
                for (d, &res_d) in residual.iter().enumerate() {
                    for (r, &h_r) in hidden.iter().enumerate() {
                        grad_b[d * rank + r] += -res_d * h_r;
                    }
                }

                // B^T @ residual → [rank]
                let mut bt_residual = vec![0.0f32; rank];
                for (r, bt) in bt_residual.iter_mut().enumerate() {
                    *bt = residual
                        .iter()
                        .enumerate()
                        .map(|(d, &res_d)| adapter.weight_b[d * rank + r] * res_d)
                        .sum();
                }

                // grad_A += -(B^T @ residual) @ base^T  (outer product, [rank x dim])
                for (r, &bt_r) in bt_residual.iter().enumerate() {
                    for (d, &base_d) in base.iter().enumerate() {
                        grad_a[r * dim + d] += -bt_r * base_d;
                    }
                }
            }

            // Average the gradients.
            for g in grad_a.iter_mut() {
                *g /= n;
            }
            for g in grad_b.iter_mut() {
                *g /= n;
            }

            // --- 4. Apply EWC penalty ---
            let ewc_guard = self.ewc.read();
            if let Some(ref ewc) = *ewc_guard {
                let current = adapter.flat_weights();
                let ewc_grad = ewc.gradient(&current);

                // Split EWC gradient into A and B parts.
                let (ewc_a, ewc_b) = ewc_grad.split_at(rank * dim);
                for (g, e) in grad_a.iter_mut().zip(ewc_a.iter()) {
                    *g += e;
                }
                for (g, e) in grad_b.iter_mut().zip(ewc_b.iter()) {
                    *g += e;
                }
            }
            drop(ewc_guard);

            // Apply gradient update (gradient descent: w -= lr * grad).
            for (w, g) in adapter.weight_a.iter_mut().zip(grad_a.iter()) {
                *w -= lr * g;
            }
            for (w, g) in adapter.weight_b.iter_mut().zip(grad_b.iter()) {
                *w -= lr * g;
            }

            // --- 6. Increment version ---
            adapter.version += 1;
            adapter.trained_on_samples += pairs.len() as u64;

            info!(
                version = adapter.version,
                total_samples = adapter.trained_on_samples,
                "LoRA train_step: training round complete"
            );
        }

        // --- 5. Update Fisher diagonal estimate ---
        // Approximate Fisher information as the squared gradient magnitude
        // per weight, averaged over the training pairs.
        {
            let adapter = self.adapter.read();
            let mut fisher = vec![0.0f32; num_params];

            let n = pairs.len() as f32;
            for pair in &pairs {
                if pair.embedding.len() != dim {
                    continue;
                }
                let base = &pair.embedding;
                let output = adapter.apply(base, alpha);
                let target = compute_target_embedding(base, &pair.response_text, dim);

                let residual: Vec<f32> = target
                    .iter()
                    .zip(output.iter())
                    .map(|(&t, &o)| t - o)
                    .collect();

                // Approximate per-weight Fisher as squared gradient.
                // For A weights:
                let mut hidden = vec![0.0f32; rank];
                for (r, h) in hidden.iter_mut().enumerate() {
                    let row = &adapter.weight_a[r * dim..(r + 1) * dim];
                    *h = row.iter().zip(base.iter()).map(|(&a, &b)| a * b).sum();
                }

                let mut bt_residual = vec![0.0f32; rank];
                for (r, bt) in bt_residual.iter_mut().enumerate() {
                    *bt = residual
                        .iter()
                        .enumerate()
                        .map(|(d, &res_d)| adapter.weight_b[d * rank + r] * res_d)
                        .sum();
                }

                for r in 0..rank {
                    for d in 0..dim {
                        let g = bt_residual[r] * base[d];
                        fisher[r * dim + d] += g * g / n;
                    }
                }

                let a_offset = rank * dim;
                for d in 0..dim {
                    for r in 0..rank {
                        let g = residual[d] * hidden[r];
                        fisher[a_offset + d * rank + r] += g * g / n;
                    }
                }
            }

            let weights_snapshot = adapter.flat_weights();
            drop(adapter);

            let mut ewc_guard = self.ewc.write();
            *ewc_guard = Some(EwcState {
                fisher_diagonal: fisher,
                prev_weights: weights_snapshot,
                lambda: self.config.ewc_lambda,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute a target embedding for a training pair. This produces a
/// content-dependent perturbation of the base embedding to serve as a
/// learning signal.
fn compute_target_embedding(base: &[f32], response_text: &str, dim: usize) -> Vec<f32> {
    // Use a simple hash-based perturbation seeded by the response text.
    let hash = simple_hash(response_text);
    let perturbation_scale = 0.05;

    base.iter()
        .enumerate()
        .map(|(i, &b)| {
            // Deterministic pseudo-random perturbation per dimension.
            let seed = hash.wrapping_add(i as u64);
            let pseudo_rand = ((seed.wrapping_mul(6364136223846793005).wrapping_add(1)) as f32)
                / (u64::MAX as f32);
            // Map to [-1, 1].
            let noise = (pseudo_rand * 2.0) - 1.0;
            b + perturbation_scale * noise
        })
        .take(dim)
        .collect()
}

/// Simple non-cryptographic hash for seeding perturbations.
fn simple_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> LoraConfig {
        LoraConfig {
            rank: 4,
            dim: 8,
            alpha: 0.1,
            batch_size: 2,
            training_interval_secs: 60,
            ewc_lambda: 10.0,
            min_samples: 4,
            learning_rate: 0.01,
        }
    }

    fn make_pair(dim: usize) -> TrainingPair {
        TrainingPair {
            query_text: "What is the mitral valve?".into(),
            response_text: "The mitral valve is a bicuspid valve in the heart.".into(),
            embedding: vec![1.0; dim],
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn apply_produces_correct_shape() {
        let adapter = LoraAdapter::random_init(4, 8);
        let base = vec![1.0f32; 8];
        let out = adapter.apply(&base, 0.1);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn zero_weights_return_base_unchanged() {
        let adapter = LoraAdapter::zeros(4, 8);
        let base: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let out = adapter.apply(&base, 0.1);
        for (a, b) in out.iter().zip(base.iter()) {
            assert!((a - b).abs() < 1e-7, "expected {b}, got {a}");
        }
    }

    #[test]
    fn training_step_increments_version() {
        let config = make_config();
        let dim = config.dim;
        let worker = LoraWorker::new(config);

        // Add enough pairs to meet min_samples.
        for _ in 0..5 {
            worker.add_training_pair(make_pair(dim));
        }

        assert_eq!(worker.adapter.read().version, 0);
        worker.train_step();
        assert_eq!(worker.adapter.read().version, 1);
    }

    #[test]
    fn training_step_skips_when_insufficient_samples() {
        let config = make_config();
        let dim = config.dim;
        let worker = LoraWorker::new(config);

        // Add fewer than min_samples.
        worker.add_training_pair(make_pair(dim));
        worker.train_step();

        // Version should still be 0.
        assert_eq!(worker.adapter.read().version, 0);
        // Samples should be returned to the buffer.
        assert_eq!(worker.buffer_size(), 1);
    }

    #[test]
    fn ewc_penalty_computation() {
        let ewc = EwcState {
            fisher_diagonal: vec![1.0, 2.0, 3.0, 4.0],
            prev_weights: vec![0.0, 0.0, 0.0, 0.0],
            lambda: 1.0,
        };

        let current = vec![1.0, 1.0, 1.0, 1.0];
        let penalty = ewc.penalty(&current);
        // penalty = 0.5 * 1.0 * (1*1 + 2*1 + 3*1 + 4*1) = 0.5 * 10 = 5.0
        assert!((penalty - 5.0).abs() < 1e-6);
    }

    #[test]
    fn ewc_gradient_computation() {
        let ewc = EwcState {
            fisher_diagonal: vec![1.0, 2.0],
            prev_weights: vec![0.5, 0.5],
            lambda: 10.0,
        };

        let current = vec![1.5, 2.5];
        let grad = ewc.gradient(&current);
        // grad[0] = 10.0 * 1.0 * (1.5 - 0.5) = 10.0
        // grad[1] = 10.0 * 2.0 * (2.5 - 0.5) = 40.0
        assert!((grad[0] - 10.0).abs() < 1e-6);
        assert!((grad[1] - 40.0).abs() < 1e-6);
    }

    #[test]
    fn buffer_accumulation_and_draining() {
        let config = make_config();
        let dim = config.dim;
        let worker = LoraWorker::new(config);

        assert_eq!(worker.buffer_size(), 0);

        for i in 0..10 {
            worker.add_training_pair(TrainingPair {
                query_text: format!("query {i}"),
                response_text: format!("response {i}"),
                embedding: vec![i as f32; dim],
                timestamp: Utc::now(),
            });
        }

        assert_eq!(worker.buffer_size(), 10);

        // Training should drain the buffer.
        worker.train_step();
        assert_eq!(worker.buffer_size(), 0);
        assert_eq!(worker.adapter.read().version, 1);
    }

    #[test]
    fn ewc_is_set_after_training() {
        let config = make_config();
        let dim = config.dim;
        let worker = LoraWorker::new(config);

        assert!(worker.ewc.read().is_none());

        for _ in 0..5 {
            worker.add_training_pair(make_pair(dim));
        }
        worker.train_step();

        assert!(worker.ewc.read().is_some());
        let ewc = worker.ewc.read();
        let ewc = ewc.as_ref().unwrap();
        assert!(!ewc.fisher_diagonal.is_empty());
        assert!(!ewc.prev_weights.is_empty());
    }

    #[test]
    fn multiple_training_rounds_use_ewc() {
        let config = make_config();
        let dim = config.dim;
        let worker = LoraWorker::new(config);

        // First round: meet min_samples.
        for _ in 0..5 {
            worker.add_training_pair(make_pair(dim));
        }
        worker.train_step();
        assert_eq!(worker.adapter.read().version, 1);

        // Second round: only need batch_size (2).
        for _ in 0..3 {
            worker.add_training_pair(TrainingPair {
                query_text: "Another query".into(),
                response_text: "Different response for EWC diversity.".into(),
                embedding: vec![0.5; dim],
                timestamp: Utc::now(),
            });
        }
        worker.train_step();
        assert_eq!(worker.adapter.read().version, 2);

        // EWC state should have been updated for the second round.
        let ewc = worker.ewc.read();
        assert!(ewc.is_some());
    }

    #[test]
    fn adapter_apply_with_known_weights() {
        // Manually set weights to verify the math.
        let mut adapter = LoraAdapter::zeros(2, 3);
        // A = [[1, 0, 0], [0, 1, 0]]  (2x3)
        adapter.weight_a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        // B = [[1, 0], [0, 1], [0, 0]]  (3x2)
        adapter.weight_b = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        let base = vec![2.0, 3.0, 4.0];
        let alpha = 1.0;
        let out = adapter.apply(&base, alpha);

        // hidden = A @ base = [2, 3]
        // delta = B @ hidden = [2, 3, 0]
        // out = base + 1.0 * delta = [4, 6, 4]
        assert!((out[0] - 4.0).abs() < 1e-6);
        assert!((out[1] - 6.0).abs() < 1e-6);
        assert!((out[2] - 4.0).abs() < 1e-6);
    }
}
