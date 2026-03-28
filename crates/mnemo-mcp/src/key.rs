use sha2::{Digest, Sha256};

/// Build a cache key for an MCP tool call.
/// Key = hash(tool_name + cache_key_fields from arguments).
pub fn build_mcp_key(
    tool_name: &str,
    arguments: &serde_json::Value,
    cache_key_fields: &[String],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(tool_name.as_bytes());

    if cache_key_fields.is_empty() {
        // Use all arguments if no specific fields configured
        let serialized = serde_json::to_string(arguments).unwrap_or_default();
        hasher.update(serialized.as_bytes());
    } else {
        // Only include specified fields
        for field in cache_key_fields {
            if let Some(val) = arguments.get(field) {
                hasher.update(field.as_bytes());
                let serialized = serde_json::to_string(val).unwrap_or_default();
                hasher.update(serialized.as_bytes());
            }
        }
    }

    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_same_inputs_same_key() {
        let args = json!({"query": "rust async", "index": "docs"});
        let fields = vec!["query".to_string(), "index".to_string()];

        let k1 = build_mcp_key("search_docs", &args, &fields);
        let k2 = build_mcp_key("search_docs", &args, &fields);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_different_tools_different_keys() {
        let args = json!({"id": "123"});
        let fields = vec!["id".to_string()];

        let k1 = build_mcp_key("tool_a", &args, &fields);
        let k2 = build_mcp_key("tool_b", &args, &fields);
        assert_ne!(k1, k2);
    }
}
