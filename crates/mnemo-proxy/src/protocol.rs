use axum::http::Uri;

/// Detected protocol type for an incoming request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// Standard LLM chat/completion calls (OpenAI-compatible).
    Llm,
    /// MCP tool calls.
    Mcp,
    /// ACP agent-to-agent messages.
    Acp,
}

/// Detect the protocol from the request URI path.
/// This runs first on every request before any cache lookup.
pub fn detect(uri: &Uri) -> Option<Protocol> {
    let path = uri.path();
    if path.starts_with("/v1/") || path == "/health" {
        Some(Protocol::Llm)
    } else if path.starts_with("/mcp/") {
        Some(Protocol::Mcp)
    } else if path.starts_with("/acp/") {
        Some(Protocol::Acp)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_llm() {
        let uri: Uri = "/v1/chat/completions".parse().unwrap();
        assert_eq!(detect(&uri), Some(Protocol::Llm));
    }

    #[test]
    fn test_detect_mcp() {
        let uri: Uri = "/mcp/tools/call".parse().unwrap();
        assert_eq!(detect(&uri), Some(Protocol::Mcp));
    }

    #[test]
    fn test_detect_acp() {
        let uri: Uri = "/acp/tasks".parse().unwrap();
        assert_eq!(detect(&uri), Some(Protocol::Acp));
    }

    #[test]
    fn test_detect_unknown() {
        let uri: Uri = "/unknown".parse().unwrap();
        assert_eq!(detect(&uri), None);
    }
}
