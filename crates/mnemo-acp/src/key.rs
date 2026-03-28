use sha2::{Digest, Sha256};

/// Build a cache key for an ACP agent task.
/// Key = hash(agent_id + task_type + normalized_input).
/// Design rule #7: agent_id is ALWAYS part of the key.
pub fn build_acp_key(
    agent_id: &str,
    task: &str,
    input: &serde_json::Value,
    cache_key_fields: &[String],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(agent_id.as_bytes());
    hasher.update(task.as_bytes());

    if cache_key_fields.is_empty() {
        let serialized = serde_json::to_string(input).unwrap_or_default();
        hasher.update(serialized.as_bytes());
    } else {
        for field in cache_key_fields {
            if let Some(val) = input.get(field) {
                hasher.update(field.as_bytes());
                let serialized = serde_json::to_string(val).unwrap_or_default();
                hasher.update(serialized.as_bytes());
            }
        }
    }

    hex::encode(hasher.finalize())
}
