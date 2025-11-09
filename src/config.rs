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

// ============================================================================
// PROVIDER ENUM
// ============================================================================

/// Enum representing supported local LLM server providers.
///
/// Each provider has a default base URL where its API server typically runs.
/// These are convenience shortcuts to avoid hardcoding URLs in application code.
///
/// ## Provider Details
///
/// | Provider | Default URL | Port | Description |
/// |----------|-------------|------|-------------|
/// | LMStudio | http://localhost:1234/v1 | 1234 | GUI-based local server |
/// | Ollama | http://localhost:11434/v1 | 11434 | CLI-focused server |
/// | LlamaCpp | http://localhost:8080/v1 | 8080 | C++ inference engine |
/// | VLLM | http://localhost:8000/v1 | 8000 | High-performance server |
///
/// All providers implement the OpenAI-compatible API standard, making them
/// interchangeable from the SDK's perspective.
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
    /// Get the default base URL for this provider.
    ///
    /// Returns the standard localhost URL where each provider's API server
    /// typically runs. All URLs include the `/v1` path suffix required by
    /// the OpenAI-compatible API standard.
    ///
    /// # Returns
    ///
    /// A static string slice containing the full base URL including protocol,
    /// host, port, and API version path.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use open_agent::Provider;
    ///
    /// assert_eq!(Provider::Ollama.default_url(), "http://localhost:11434/v1");
    /// assert_eq!(Provider::LMStudio.default_url(), "http://localhost:1234/v1");
    /// ```
    pub fn default_url(&self) -> &'static str {
        match self {
            // LM Studio's default port from their documentation
            Provider::LMStudio => "http://localhost:1234/v1",

            // Ollama's default port from their documentation
            Provider::Ollama => "http://localhost:11434/v1",

            // llama.cpp server's common default port
            Provider::LlamaCpp => "http://localhost:8080/v1",

            // vLLM's default port from their documentation
            Provider::VLLM => "http://localhost:8000/v1",
        }
    }
}

// ============================================================================
// FROMSTR IMPLEMENTATION FOR PARSING
// ============================================================================

impl FromStr for Provider {
    type Err = String;

    /// Parse a provider name from a string.
    ///
    /// This implementation is case-insensitive and supports multiple naming
    /// conventions (dashes, underscores, dots) for flexibility.
    ///
    /// # Supported Formats
    ///
    /// - **LMStudio**: "lmstudio", "lm-studio", "lm_studio" (case-insensitive)
    /// - **Ollama**: "ollama" (case-insensitive)
    /// - **LlamaCpp**: "llamacpp", "llama-cpp", "llama_cpp", "llama.cpp" (case-insensitive)
    /// - **VLLM**: "vllm" (case-insensitive)
    ///
    /// # Errors
    ///
    /// Returns a `String` error message if the provider name is not recognized.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use open_agent::Provider;
    /// use std::str::FromStr;
    ///
    /// let provider = "ollama".parse::<Provider>().unwrap();
    /// assert_eq!(provider, Provider::Ollama);
    ///
    /// let provider = "LM-Studio".parse::<Provider>().unwrap();
    /// assert_eq!(provider, Provider::LMStudio);
    ///
    /// assert!("unknown".parse::<Provider>().is_err());
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Convert to lowercase for case-insensitive matching
        match s.to_lowercase().as_str() {
            // LM Studio accepts multiple common variations
            "lmstudio" | "lm-studio" | "lm_studio" => Ok(Provider::LMStudio),

            // Ollama is simple
            "ollama" => Ok(Provider::Ollama),

            // llama.cpp has many variations in the wild
            "llamacpp" | "llama-cpp" | "llama_cpp" | "llama.cpp" => Ok(Provider::LlamaCpp),

            // vLLM is straightforward
            "vllm" => Ok(Provider::VLLM),

            // Unrecognized provider name
            _ => Err(format!("Unknown provider: {}", s)),
        }
    }
}

// ============================================================================
// CONFIGURATION HELPER FUNCTIONS
// ============================================================================

/// Get the base URL for API requests with environment variable support.
///
/// This function implements a priority-based resolution strategy for determining
/// the API base URL, making it easy to override defaults during development or deployment.
///
/// ## Resolution Priority
///
/// 1. **Environment Variable**: `OPEN_AGENT_BASE_URL` (highest priority)
/// 2. **Provider Default**: The provider's default URL if specified
/// 3. **Fallback Parameter**: Explicit fallback value
/// 4. **Ultimate Default**: LM Studio's default URL (http://localhost:1234/v1)
///
/// ## Use Cases
///
/// - **Development**: Use provider defaults for quick setup
/// - **Testing**: Override with environment variable to point to test server
/// - **Production**: Specify explicit fallback for reliability
///
/// # Arguments
///
/// * `provider` - Optional provider enum to use its default URL
/// * `fallback` - Optional explicit fallback URL string
///
/// # Returns
///
/// The resolved base URL as a `String`. Always returns a value (never None).
///
/// # Examples
///
/// ```rust,no_run
/// use open_agent::{get_base_url, Provider};
///
/// // Use Ollama's default (http://localhost:11434/v1)
/// let url = get_base_url(Some(Provider::Ollama), None);
///
/// // With explicit fallback
/// let url = get_base_url(None, Some("http://localhost:1234/v1"));
///
/// // Override via environment (takes precedence over everything)
/// // SAFETY: This is a doctest example showing how env vars work
/// unsafe { std::env::set_var("OPEN_AGENT_BASE_URL", "http://custom-server:8080/v1"); }
/// let url = get_base_url(Some(Provider::Ollama), None);
/// // Returns "http://custom-server:8080/v1" despite provider being set
/// ```
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

/// Get the model name with optional environment variable override.
///
/// This function provides flexible model name resolution with opt-in environment
/// variable support. Unlike `get_base_url`, environment variable checking is
/// controlled by the `prefer_env` parameter.
///
/// ## Resolution Priority
///
/// If `prefer_env` is `true`:
/// 1. **Environment Variable**: `OPEN_AGENT_MODEL` (if set)
/// 2. **Fallback Parameter**: Explicit fallback value
///
/// If `prefer_env` is `false`:
/// 1. **Fallback Parameter**: Explicit fallback value only
///
/// ## Why Optional Environment Override?
///
/// Model names are often specified explicitly in code for consistency across
/// environments. The `prefer_env` flag gives you control over whether to
/// allow environment overrides.
///
/// # Arguments
///
/// * `fallback` - Optional explicit model name
/// * `prefer_env` - Whether to check environment variable first
///
/// # Returns
///
/// `Some(String)` if a model name was found, `None` if no model specified
///
/// # Examples
///
/// ```rust,no_run
/// use open_agent::get_model;
///
/// // Use explicit model name, allow environment override
/// let model = get_model(Some("llama3:8b"), true);
///
/// // Force specific model (ignore environment)
/// let model = get_model(Some("qwen2.5-32b"), false);
///
/// // Try environment only
/// let model = get_model(None, true);
/// // Returns Some(model) if OPEN_AGENT_MODEL is set, None otherwise
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

// ============================================================================
// TESTS
// ============================================================================

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

    #[test]
    fn test_get_base_url_with_provider() {
        // SAFETY: This test runs in an isolated test environment where environment
        // variable modifications won't affect other tests due to test isolation.
        // The OPEN_AGENT_BASE_URL variable is specific to this library and not
        // used by the Rust standard library or other critical system components.
        unsafe {
            env::remove_var("OPEN_AGENT_BASE_URL");
        }

        let url = get_base_url(Some(Provider::Ollama), None);
        assert_eq!(url, "http://localhost:11434/v1");
    }

    #[test]
    fn test_get_base_url_with_fallback() {
        // SAFETY: This test runs in an isolated test environment where environment
        // variable modifications won't affect other tests due to test isolation.
        // The OPEN_AGENT_BASE_URL variable is specific to this library and not
        // used by the Rust standard library or other critical system components.
        unsafe {
            env::remove_var("OPEN_AGENT_BASE_URL");
        }

        let url = get_base_url(None, Some("http://custom:8080/v1"));
        assert_eq!(url, "http://custom:8080/v1");
    }
}
