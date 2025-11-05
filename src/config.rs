//! Configuration helpers for the Open Agent SDK

use std::env;

/// Supported provider shortcuts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    LMStudio,
    Ollama,
    LlamaCpp,
    VLLM,
}

impl Provider {
    /// Get the default base URL for this provider
    pub fn default_url(&self) -> &'static str {
        match self {
            Provider::LMStudio => "http://localhost:1234/v1",
            Provider::Ollama => "http://localhost:11434/v1",
            Provider::LlamaCpp => "http://localhost:8080/v1",
            Provider::VLLM => "http://localhost:8000/v1",
        }
    }

    /// Parse a provider from a string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "lmstudio" | "lm-studio" | "lm_studio" => Some(Provider::LMStudio),
            "ollama" => Some(Provider::Ollama),
            "llamacpp" | "llama-cpp" | "llama_cpp" | "llama.cpp" => Some(Provider::LlamaCpp),
            "vllm" => Some(Provider::VLLM),
            _ => None,
        }
    }
}

/// Get the base URL from environment variable or provider default
///
/// Priority:
/// 1. OPEN_AGENT_BASE_URL environment variable
/// 2. Provider default URL (if provider is Some)
/// 3. fallback parameter
///
/// # Examples
///
/// ```rust,no_run
/// use open_agent::{get_base_url, Provider};
///
/// // Read from environment
/// let url = get_base_url(None, None);
///
/// // Use provider default
/// let url = get_base_url(Some(Provider::Ollama), None);
///
/// // With fallback
/// let url = get_base_url(None, Some("http://localhost:1234/v1"));
/// ```
pub fn get_base_url(provider: Option<Provider>, fallback: Option<&str>) -> String {
    // Try environment variable first
    if let Ok(url) = env::var("OPEN_AGENT_BASE_URL") {
        return url;
    }

    // Try provider default
    if let Some(p) = provider {
        return p.default_url().to_string();
    }

    // Use fallback or default to LM Studio
    fallback
        .unwrap_or(Provider::LMStudio.default_url())
        .to_string()
}

/// Get the model name from environment variable or fallback
///
/// Priority:
/// 1. OPEN_AGENT_MODEL environment variable (if prefer_env is true)
/// 2. fallback parameter
///
/// # Examples
///
/// ```rust,no_run
/// use open_agent::get_model;
///
/// // Read from environment
/// let model = get_model(None, true);
///
/// // Use fallback
/// let model = get_model(Some("qwen2.5-32b-instruct"), true);
///
/// // Force fallback (ignore environment)
/// let model = get_model(Some("specific-model"), false);
/// ```
pub fn get_model(fallback: Option<&str>, prefer_env: bool) -> Option<String> {
    // Try environment variable first if preferred
    if prefer_env {
        if let Ok(model) = env::var("OPEN_AGENT_MODEL") {
            return Some(model);
        }
    }

    // Use fallback
    fallback.map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_default_urls() {
        assert_eq!(Provider::LMStudio.default_url(), "http://localhost:1234/v1");
        assert_eq!(Provider::Ollama.default_url(), "http://localhost:11434/v1");
        assert_eq!(Provider::LlamaCpp.default_url(), "http://localhost:8080/v1");
        assert_eq!(Provider::VLLM.default_url(), "http://localhost:8000/v1");
    }

    #[test]
    fn test_provider_from_str() {
        assert_eq!(Provider::from_str("lmstudio"), Some(Provider::LMStudio));
        assert_eq!(Provider::from_str("LM-Studio"), Some(Provider::LMStudio));
        assert_eq!(Provider::from_str("ollama"), Some(Provider::Ollama));
        assert_eq!(Provider::from_str("llamacpp"), Some(Provider::LlamaCpp));
        assert_eq!(Provider::from_str("llama.cpp"), Some(Provider::LlamaCpp));
        assert_eq!(Provider::from_str("vllm"), Some(Provider::VLLM));
        assert_eq!(Provider::from_str("unknown"), None);
    }

    #[test]
    fn test_get_base_url_with_provider() {
        // Clear environment variable
        env::remove_var("OPEN_AGENT_BASE_URL");

        let url = get_base_url(Some(Provider::Ollama), None);
        assert_eq!(url, "http://localhost:11434/v1");
    }

    #[test]
    fn test_get_base_url_with_fallback() {
        // Clear environment variable
        env::remove_var("OPEN_AGENT_BASE_URL");

        let url = get_base_url(None, Some("http://custom:8080/v1"));
        assert_eq!(url, "http://custom:8080/v1");
    }
}
