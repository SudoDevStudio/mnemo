//! L3 — Redis-backed persistent cache layer.
//!
//! L3 sits on the **cold path**: it is only consulted after L1 (exact hash) and
//! L2 (HNSW vector) both miss.  Because it lives behind the network, every
//! operation is async and fallible.  Errors are logged but never crash the proxy
//! — L3 is strictly optional.
//!
//! Entries are stored as JSON strings under the key prefix `mnemo:`.
//! Default TTL is 7 days.

use std::time::Duration;

use deadpool_redis::{Config, Pool, Runtime};
use redis::AsyncCommands;
use thiserror::Error;

use crate::CacheEntry;

/// Key prefix for all Mnemo entries in Redis.
const KEY_PREFIX: &str = "mnemo:";

/// Default TTL for cached entries: 7 days.
const DEFAULT_TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum L3Error {
    #[error("redis pool error: {0}")]
    Pool(#[from] deadpool_redis::CreatePoolError),

    #[error("redis connection error: {0}")]
    Connection(#[from] deadpool_redis::PoolError),

    #[error("redis command error: {0}")]
    Redis(#[from] redis::RedisError),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

// ---------------------------------------------------------------------------
// L3RedisCache
// ---------------------------------------------------------------------------

/// Redis-backed persistent cache (L3).
///
/// Uses `deadpool-redis` for connection pooling.  All public methods log errors
/// at `warn` level so callers can treat failures as soft misses.
pub struct L3RedisCache {
    pool: Pool,
    ttl: Duration,
}

impl L3RedisCache {
    /// Create a new L3 cache backed by the Redis instance at `redis_url`.
    ///
    /// `redis_url` should be a full connection string, e.g.
    /// `redis://127.0.0.1:6379` or `redis://:password@host:6379/0`.
    pub fn new(redis_url: &str) -> Result<Self, L3Error> {
        let cfg = Config::from_url(redis_url);
        let pool = cfg.create_pool(Some(Runtime::Tokio1))?;
        Ok(Self {
            pool,
            ttl: DEFAULT_TTL,
        })
    }

    /// Create a new L3 cache with a custom TTL.
    pub fn with_ttl(redis_url: &str, ttl: Duration) -> Result<Self, L3Error> {
        let mut cache = Self::new(redis_url)?;
        cache.ttl = ttl;
        Ok(cache)
    }

    // -- helpers ------------------------------------------------------------

    /// Build the full Redis key for a given cache key.
    #[inline]
    fn redis_key(key: &str) -> String {
        format!("{KEY_PREFIX}{key}")
    }

    // -- public API ---------------------------------------------------------

    /// Retrieve a cached entry.  Returns `Ok(None)` on a cache miss.
    pub async fn get(&self, key: &str) -> Result<Option<CacheEntry>, L3Error> {
        let mut conn = self.pool.get().await?;
        let raw: Option<String> = conn.get(Self::redis_key(key)).await?;

        match raw {
            Some(json) => {
                let entry: CacheEntry = serde_json::from_str(&json)?;
                tracing::debug!(key, "l3 cache hit");
                Ok(Some(entry))
            }
            None => {
                tracing::debug!(key, "l3 cache miss");
                Ok(None)
            }
        }
    }

    /// Store an entry in Redis with the configured TTL.
    pub async fn insert(&self, key: &str, entry: &CacheEntry) -> Result<(), L3Error> {
        let json = serde_json::to_string(entry)?;
        let rkey = Self::redis_key(key);
        let ttl_secs = self.ttl.as_secs() as i64;

        let mut conn = self.pool.get().await?;
        conn.set_ex::<_, _, ()>(&rkey, &json, ttl_secs as u64)
            .await?;

        tracing::debug!(key, ttl_secs, "l3 cache insert");
        Ok(())
    }

    /// Remove a single entry from Redis.
    pub async fn invalidate(&self, key: &str) -> Result<(), L3Error> {
        let mut conn = self.pool.get().await?;
        conn.del::<_, ()>(Self::redis_key(key)).await?;
        tracing::debug!(key, "l3 cache invalidate");
        Ok(())
    }

    /// Flush **all** Mnemo keys from Redis (`mnemo:*`).
    ///
    /// This uses `SCAN` + `DEL` rather than `KEYS` to avoid blocking Redis on
    /// large keyspaces.
    pub async fn flush(&self) -> Result<(), L3Error> {
        let mut conn = self.pool.get().await?;
        let pattern = format!("{KEY_PREFIX}*");
        let mut cursor: u64 = 0;
        let mut total_deleted: u64 = 0;

        loop {
            let (next_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&pattern)
                .arg("COUNT")
                .arg(500)
                .query_async(&mut *conn)
                .await?;

            if !keys.is_empty() {
                let count = keys.len() as u64;
                redis::cmd("DEL")
                    .arg(&keys)
                    .query_async::<()>(&mut *conn)
                    .await?;
                total_deleted += count;
            }

            cursor = next_cursor;
            if cursor == 0 {
                break;
            }
        }

        tracing::info!(total_deleted, "l3 cache flushed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redis_key_has_prefix() {
        assert_eq!(L3RedisCache::redis_key("abc123"), "mnemo:abc123");
    }

    #[test]
    fn default_ttl_is_seven_days() {
        assert_eq!(DEFAULT_TTL, Duration::from_secs(604_800));
    }
}
