//! # Configuration Helpers for the Open Agent SDK
//!
//! This module provides convenience utilities for configuring connections to different
//! local LLM server providers. It simplifies the process of setting base URLs and model names
//! by providing well-known defaults and environment variable support.
//!
//! ## Supported Providers
//!
//! - **LM Studio**: Popular local model server with GUI
//! - **Ollama**: Command-line focused local model server
//! - **llama.cpp**: C++ inference engine with server mode
//! - **vLLM**: High-performance inference server
//!
//! ## Environment Variables
//!
//! - `OPEN_AGENT_BASE_URL`: Override base URL for any provider
//! - `OPEN_AGENT_MODEL`: Override model name (when prefer_env is true)
//!
//! ## Examples
//!
//! ```rust
//! use open_agent::{Provider, get_base_url, get_model, AgentOptions};
//!
//! // Quick setup with provider defaults
//! let url = get_base_url(Some(Provider::Ollama), None);
//! let model = get_model(Some("llama3:8b"), false);
//!
//! // Build options
//! let options = AgentOptions::builder()
//!     .system_prompt("You are a helpful assistant")
//!     .model(model.unwrap())
//!     .base_url(&url)
//!     .build()
//!     .unwrap();
//! ```

use std::env;
use std::str::FromStr;

/// Local inference providers with conventional OpenAI-compatible endpoint URLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// LM Studio - Popular GUI-based local model server (default port 1234)
    LMStudio,

    /// Ollama - Command-line focused local model server (default port 11434)
    Ollama,

    /// llama.cpp - C++ inference engine with server mode (default port 8080)
    LlamaCpp,

    /// vLLM - High-performance inference server (default port 8000)
    VLLM,
}

impl Provider {
    /// Returns the conventional local URL, without consulting environment variables.
    ///
    /// LM Studio uses port 1234, Ollama 11434, llama.cpp 8080, and vLLM 8000;
    /// each URL ends in `/v1`.
    pub fn default_url(&self) -> &'static str {
        match self {
            Provider::LMStudio => "http://localhost:1234/v1",

            Provider::Ollama => "http://localhost:11434/v1",

            Provider::LlamaCpp => "http://localhost:8080/v1",

            Provider::VLLM => "http://localhost:8000/v1",
        }
    }
}

impl FromStr for Provider {
    type Err = String;

    /// Parses a case-insensitive provider name.
    ///
    /// Accepts `lmstudio`, `lm-studio`, `lm_studio`, `ollama`, `llamacpp`, `llama-cpp`,
    /// `llama_cpp`, `llama.cpp`, and `vllm`. Unknown names return an error.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "lmstudio" | "lm-studio" | "lm_studio" => Ok(Provider::LMStudio),

            "ollama" => Ok(Provider::Ollama),

            "llamacpp" | "llama-cpp" | "llama_cpp" | "llama.cpp" => Ok(Provider::LlamaCpp),

            "vllm" => Ok(Provider::VLLM),

            _ => Err(format!("Unknown provider: {}", s)),
        }
    }
}

/// Resolves a base URL in this order:
///
/// 1. `OPEN_AGENT_BASE_URL`, when readable as Unicode.
/// 2. The selected provider's default URL.
/// 3. The explicit fallback.
/// 4. LM Studio's default URL.
///
/// The returned value is not validated.
pub fn get_base_url(provider: Option<Provider>, fallback: Option<&str>) -> String {
    // Priority 1: Check environment variable first (allows runtime override)
    if let Ok(url) = env::var("OPEN_AGENT_BASE_URL") {
        return url;
    }

    // Priority 2: Use provider's default URL if specified
    if let Some(p) = provider {
        return p.default_url().to_string();
    }

    // Priority 3: Use explicit fallback, or default to LM Studio
    fallback
        .unwrap_or(Provider::LMStudio.default_url())
        .to_string()
}

/// Resolves a model name with optional environment lookup.
///
/// When `prefer_env` is true, a readable `OPEN_AGENT_MODEL` takes precedence over
/// `fallback`. Otherwise only `fallback` is considered. Neither value is validated;
/// no value yields `None`.
///
/// ```rust
/// use open_agent::get_model;
///
/// assert_eq!(get_model(Some("my-model"), false).as_deref(), Some("my-model"));
/// let from_environment = get_model(None, true);
/// ```
pub fn get_model(fallback: Option<&str>, prefer_env: bool) -> Option<String> {
    // If environment override is preferred, check it first
    if prefer_env {
        if let Ok(model) = env::var("OPEN_AGENT_MODEL") {
            return Some(model);
        }
    }

    // Fall back to the explicit parameter (if provided)
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
        assert_eq!("lmstudio".parse::<Provider>(), Ok(Provider::LMStudio));
        assert_eq!("LM-Studio".parse::<Provider>(), Ok(Provider::LMStudio));
        assert_eq!("ollama".parse::<Provider>(), Ok(Provider::Ollama));
        assert_eq!("llamacpp".parse::<Provider>(), Ok(Provider::LlamaCpp));
        assert_eq!("llama.cpp".parse::<Provider>(), Ok(Provider::LlamaCpp));
        assert_eq!("vllm".parse::<Provider>(), Ok(Provider::VLLM));
        assert!("unknown".parse::<Provider>().is_err());
    }
}
