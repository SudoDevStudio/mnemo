use sha2::{Digest, Sha256};

use crate::protocol::Protocol;

/// Normalize an LLM request body and produce a cache key hash.
/// Strips non-deterministic fields (stream, temperature, etc.) so that
/// semantically identical requests map to the same key.
pub fn normalize_and_hash(protocol: Protocol, body: &serde_json::Value) -> String {
    let normalized = match protocol {
        Protocol::Llm => normalize_llm(body),
        Protocol::Mcp => normalize_mcp(body),
        Protocol::Acp => normalize_acp(body),
    };

    let serialized = serde_json::to_string(&normalized).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(serialized.as_bytes());
    hex::encode(hasher.finalize())
}

fn normalize_llm(body: &serde_json::Value) -> serde_json::Value {
    let mut normalized = serde_json::Map::new();

    // Keep only the fields that determine the response content
    if let Some(model) = body.get("model") {
        normalized.insert("model".to_string(), model.clone());
    }
    if let Some(messages) = body.get("messages") {
        normalized.insert("messages".to_string(), messages.clone());
    }
    if let Some(prompt) = body.get("prompt") {
        normalized.insert("prompt".to_string(), prompt.clone());
    }
    // Include tools/functions if present (they affect the response)
    if let Some(tools) = body.get("tools") {
        normalized.insert("tools".to_string(), tools.clone());
    }
    if let Some(functions) = body.get("functions") {
        normalized.insert("functions".to_string(), functions.clone());
    }

    serde_json::Value::Object(normalized)
}

fn normalize_mcp(body: &serde_json::Value) -> serde_json::Value {
    let mut normalized = serde_json::Map::new();

    if let Some(tool_name) = body.get("name") {
        normalized.insert("name".to_string(), tool_name.clone());
    }
    if let Some(arguments) = body.get("arguments") {
        normalized.insert("arguments".to_string(), arguments.clone());
    }

    serde_json::Value::Object(normalized)
}

fn normalize_acp(body: &serde_json::Value) -> serde_json::Value {
    let mut normalized = serde_json::Map::new();

    // ACP keys always include agent_id (design rule #7)
    if let Some(agent_id) = body.get("agent_id") {
        normalized.insert("agent_id".to_string(), agent_id.clone());
    }
    if let Some(task) = body.get("task") {
        normalized.insert("task".to_string(), task.clone());
    }
    if let Some(input) = body.get("input") {
        normalized.insert("input".to_string(), input.clone());
    }

    serde_json::Value::Object(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_llm_normalization_strips_stream() {
        let body1 = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "temperature": 0.7
        });
        let body2 = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "temperature": 0.0
        });

        let hash1 = normalize_and_hash(Protocol::Llm, &body1);
        let hash2 = normalize_and_hash(Protocol::Llm, &body2);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_acp_includes_agent_id() {
        let body1 = json!({"agent_id": "agent-a", "task": "summarise", "input": "doc1"});
        let body2 = json!({"agent_id": "agent-b", "task": "summarise", "input": "doc1"});

        let hash1 = normalize_and_hash(Protocol::Acp, &body1);
        let hash2 = normalize_and_hash(Protocol::Acp, &body2);
        assert_ne!(hash1, hash2, "different agents must produce different keys");
    }
}
