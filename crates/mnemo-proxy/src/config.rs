use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub upstream: UpstreamConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub intelligence: IntelligenceConfig,
    #[serde(default)]
    pub mcp: Option<McpConfig>,
    #[serde(default)]
    pub acp: Option<AcpConfig>,
    #[serde(default = "default_bind")]
    pub bind: String,
    #[serde(default)]
    pub logging: LoggingConfig,
}

fn default_bind() -> String {
    "0.0.0.0:8080".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoggingConfig {
    /// Log level: error, warn, info, debug, trace.
    /// Can be overridden by MNEMO_LOG or RUST_LOG env vars.
    #[serde(default = "default_log_level")]
    pub level: String,
    /// Output format: "pretty" (human-readable) or "json" (structured).
    #[serde(default = "default_log_format")]
    pub format: LogFormat,
    /// Log individual request/response details at debug level.
    #[serde(default)]
    pub log_requests: bool,
    /// Log cache hit/miss events at info level (default true).
    #[serde(default = "default_true")]
    pub log_cache_events: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
            log_requests: false,
            log_cache_events: true,
        }
    }
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_log_format() -> LogFormat {
    LogFormat::Pretty
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    Pretty,
    Json,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpstreamConfig {
    pub provider: Provider,
    pub base_url: String,
    #[serde(default)]
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    OpenAI,
    Anthropic,
    #[serde(rename = "vertexai")]
    VertexAI,
    Ollama,
    Custom,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CacheConfig {
    #[serde(default = "default_l1_max")]
    pub l1_max_entries: u64,
    #[serde(default = "default_l2_max")]
    pub l2_max_entries: u64,
    #[serde(default)]
    pub l3_redis_url: Option<String>,
    #[serde(default = "default_l3_cost")]
    pub l3_min_cost_threshold: f64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_max_entries: default_l1_max(),
            l2_max_entries: default_l2_max(),
            l3_redis_url: None,
            l3_min_cost_threshold: default_l3_cost(),
        }
    }
}

fn default_l1_max() -> u64 {
    10_000
}
fn default_l2_max() -> u64 {
    100_000
}
fn default_l3_cost() -> f64 {
    0.01
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntelligenceConfig {
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    #[serde(default)]
    pub lora_training_enabled: bool,
    #[serde(default)]
    pub staleness_detection_enabled: bool,
    #[serde(default = "default_batch_size")]
    pub training_batch_size: usize,
    #[serde(default = "default_training_interval")]
    pub training_interval_seconds: u64,
}

impl Default for IntelligenceConfig {
    fn default() -> Self {
        Self {
            embedding_model: default_embedding_model(),
            lora_training_enabled: false,
            staleness_detection_enabled: false,
            training_batch_size: default_batch_size(),
            training_interval_seconds: default_training_interval(),
        }
    }
}

fn default_embedding_model() -> String {
    "bge-small-en".to_string()
}
fn default_batch_size() -> usize {
    32
}
fn default_training_interval() -> u64 {
    300
}

#[derive(Debug, Clone, Deserialize)]
pub struct McpConfig {
    pub server_url: String,
    #[serde(default)]
    pub tools: HashMap<String, McpToolConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct McpToolConfig {
    #[serde(default)]
    pub cacheable: bool,
    #[serde(default)]
    pub cache_key_fields: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AcpConfig {
    pub registry_url: String,
    #[serde(default)]
    pub agents: HashMap<String, AcpAgentConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AcpAgentConfig {
    #[serde(default)]
    pub cacheable: bool,
    #[serde(default)]
    pub cache_key_fields: Vec<String>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::ReadFile(path.display().to_string(), e))?;

        // Expand environment variables in the config string
        let expanded =
            shellexpand::env(&contents).map_err(|e| ConfigError::EnvExpand(e.to_string()))?;

        let config: Config = serde_yaml::from_str(&expanded).map_err(ConfigError::Parse)?;

        Ok(config)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file '{0}': {1}")]
    ReadFile(String, std::io::Error),
    #[error("failed to expand environment variables: {0}")]
    EnvExpand(String),
    #[error("failed to parse config: {0}")]
    Parse(#[from] serde_yaml::Error),
}
