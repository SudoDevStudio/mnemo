use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Tracks MCP tool schema versions.
/// When a tool's schema changes, all cached entries for that tool should be
/// invalidated.
#[derive(Default)]
pub struct SchemaTracker {
    /// tool_name → hash of its schema
    schemas: HashMap<String, String>,
}

impl SchemaTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the schema for a tool. Returns true if the schema changed
    /// (meaning cached entries should be invalidated).
    pub fn update(&mut self, tool_name: &str, schema: &serde_json::Value) -> bool {
        let serialized = serde_json::to_string(schema).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        let new_hash = hex::encode(hasher.finalize());

        if let Some(existing) = self.schemas.get(tool_name) {
            if *existing == new_hash {
                return false;
            }
        }

        self.schemas.insert(tool_name.to_string(), new_hash);
        true
    }
}
