use serde_json::Value;

/// ACP message types and their cacheability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcpMessageType {
    /// Task messages — cacheable (same task + input → same output).
    Task,
    /// Status messages — never cacheable (live state).
    Status,
    /// Result messages — stored as cache values, not keys.
    Result,
    /// Error messages — never cached.
    Error,
}

impl AcpMessageType {
    /// Determine the message type from the request body.
    pub fn from_body(body: &Value) -> Self {
        if let Some(msg_type) = body.get("type").and_then(|v| v.as_str()) {
            match msg_type {
                "task" => Self::Task,
                "status" => Self::Status,
                "result" => Self::Result,
                "error" => Self::Error,
                _ => Self::Task, // Default to task for unknown types
            }
        } else {
            Self::Task
        }
    }

    /// Whether this message type is cacheable.
    pub fn is_cacheable(&self) -> bool {
        matches!(self, Self::Task)
    }
}
