use std::collections::HashMap;

/// Cacheability classification for an MCP tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cacheability {
    /// Always cacheable (static tools like get_user_profile).
    Always,
    /// Never cacheable (live tools like get_weather).
    Never,
    /// Conditionally cacheable based on specific input fields.
    Conditional,
    /// Not yet classified — defaults to non-cacheable (design rule #6).
    Unknown,
}

/// Classifies MCP tools as cacheable or not.
/// Uses explicit config first, falls back to learned classification.
#[derive(Default)]
pub struct ToolClassifier {
    /// Explicit overrides from config.
    explicit: HashMap<String, Cacheability>,
    /// Learned classifications from traffic observation.
    learned: HashMap<String, LearnedClassification>,
}

#[derive(Debug, Clone)]
struct LearnedClassification {
    /// Number of observations.
    observations: u64,
    /// Number of times same inputs produced same outputs.
    consistent_count: u64,
}

impl ToolClassifier {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set an explicit cacheability for a tool (from config).
    pub fn set_explicit(&mut self, tool_name: String, cacheability: Cacheability) {
        self.explicit.insert(tool_name, cacheability);
    }

    /// Get the cacheability classification for a tool.
    /// Explicit config takes precedence. Unknown defaults to non-cacheable.
    pub fn classify(&self, tool_name: &str) -> Cacheability {
        if let Some(&c) = self.explicit.get(tool_name) {
            return c;
        }

        if let Some(learned) = self.learned.get(tool_name) {
            if learned.observations >= 10 {
                let consistency = learned.consistent_count as f64 / learned.observations as f64;
                return if consistency > 0.95 {
                    Cacheability::Always
                } else if consistency < 0.3 {
                    Cacheability::Never
                } else {
                    Cacheability::Conditional
                };
            }
        }

        // Design rule #6: default to non-cacheable when unknown
        Cacheability::Unknown
    }

    /// Record an observation for learning.
    pub fn observe(&mut self, tool_name: &str, consistent: bool) {
        let entry = self
            .learned
            .entry(tool_name.to_string())
            .or_insert(LearnedClassification {
                observations: 0,
                consistent_count: 0,
            });
        entry.observations += 1;
        if consistent {
            entry.consistent_count += 1;
        }
    }
}
