//! # Error Types for the Open Agent SDK
//!
//! This module defines all error types used throughout the SDK, providing comprehensive
//! error handling with detailed context for different failure scenarios.
//!
//! ## Design Philosophy
//!
//! - **Explicit Error Handling**: Uses Rust's `Result<T>` type for all fallible operations
//! - **No Silent Failures**: All errors are propagated explicitly to the caller
//! - **Rich Context**: Each error variant provides specific information about what went wrong
//! - **Easy Conversion**: Automatic conversion from common error types (reqwest, serde_json)
//!
//! ## Usage
//!
//! ```ignore
//! use open_agent::{Error, Result};
//!
//! fn example() -> Result<()> {
//!     // Errors can be created using convenience methods
//!     if some_condition {
//!         return Err(Error::config("Invalid model name"));
//!     }
//!
//!     // Or automatically converted from reqwest/serde_json errors
//!     let response = http_client.get(url).send().await?; // Auto-converts to Error::Http
//!     let json = serde_json::from_str(data)?; // Auto-converts to Error::Json
//!
//!     Ok(())
//! }
//! ```

use thiserror::Error;

// ============================================================================
// TYPE ALIASES
// ============================================================================

/// Type alias for `Result<T, Error>` used throughout the SDK.
///
/// This makes function signatures more concise and ensures consistent error handling
/// across the entire API surface. Instead of writing `std::result::Result<T, Error>`,
/// you can simply write `Result<T>`.
///
/// # Example
///
/// ```rust
/// use open_agent::Result;
///
/// async fn send_request() -> Result<String> {
///     // Function body
///     Ok("Success".to_string())
/// }
/// ```
pub type Result<T> = std::result::Result<T, Error>;

// ============================================================================
// ERROR ENUM
// ============================================================================

/// Comprehensive error type covering all failure modes in the SDK.
///
/// This enum uses the `thiserror` crate to automatically implement `std::error::Error`
/// and provide well-formatted error messages. Each variant represents a different
/// category of failure that can occur during SDK operation.
///
/// ## Error Categories
///
/// - **HTTP**: Network communication failures (connection errors, timeouts, etc.)
/// - **JSON**: Serialization/deserialization failures
/// - **Config**: Invalid configuration parameters
/// - **Api**: Error responses from the model server
/// - **Stream**: Failures during streaming response processing
/// - **Tool**: Tool execution or registration failures
/// - **InvalidInput**: User-provided input validation failures
/// - **Timeout**: Request timeout exceeded
/// - **Other**: Catch-all for miscellaneous errors
///
/// ## Automatic Conversions
///
/// The `#[from]` attribute on `Http` and `Json` variants enables automatic conversion
/// from `reqwest::Error` and `serde_json::Error` using the `?` operator, making
/// error propagation seamless.
#[derive(Error, Debug)]
pub enum Error {
    /// HTTP request failed due to network issues, connection problems, or HTTP errors.
    ///
    /// This variant wraps `reqwest::Error` and is automatically created when using
    /// the `?` operator on reqwest operations. Common causes include:
    /// - Connection refused (server not running)
    /// - DNS resolution failures
    /// - TLS/SSL certificate errors
    /// - HTTP status errors (4xx, 5xx)
    /// - Network timeouts
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let response = client.post(url).send().await?; // Auto-converts reqwest::Error
    /// ```
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization or deserialization failed.
    ///
    /// This variant wraps `serde_json::Error` and occurs when:
    /// - Parsing invalid JSON from the API
    /// - Serializing request data fails
    /// - JSON structure doesn't match expected schema
    /// - Required fields are missing in JSON
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let value: MyType = serde_json::from_str(json_str)?; // Auto-converts serde_json::Error
    /// ```
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid configuration provided when building AgentOptions.
    ///
    /// Occurs during the builder pattern validation phase when required fields
    /// are missing or invalid values are provided. Common causes:
    /// - Missing required fields (model, base_url, system_prompt)
    /// - Invalid URL format in base_url
    /// - Invalid timeout values
    /// - Invalid max_tokens or temperature ranges
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// return Err(Error::config("base_url is required"));
    /// ```
    #[error("Invalid configuration: {0}")]
    Config(String),

    /// Error response received from the model server's API.
    ///
    /// This indicates the HTTP request succeeded, but the API returned an error
    /// response. Common causes:
    /// - Model not found on the server
    /// - Invalid API key or authentication failure
    /// - Rate limiting
    /// - Server-side errors (500, 502, 503)
    /// - Invalid request format
    ///
    /// The HTTP status is carried as structured data rather than embedded in the message,
    /// so retry classification can read it directly instead of parsing prose. `status` is
    /// `None` for API errors that did not come from an HTTP response.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // From an HTTP response
    /// return Err(Error::api_status(429, "Rate limit exceeded"));
    ///
    /// // Without a status
    /// return Err(Error::api("Model 'gpt-4' not found on server"));
    /// ```
    #[error("API error{}: {message}", .status.map(|code| format!(" {code}")).unwrap_or_default())]
    Api {
        /// HTTP status code, when the error originated from an HTTP response.
        status: Option<u16>,

        /// Error message or response body reported by the server.
        message: String,
    },

    /// Error occurred while processing the streaming response.
    ///
    /// This happens during Server-Sent Events (SSE) parsing or stream processing.
    /// Common causes:
    /// - Malformed SSE data
    /// - Connection interrupted mid-stream
    /// - Unexpected end of stream
    /// - Invalid chunk format
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// return Err(Error::stream("Unexpected end of SSE stream"));
    /// ```
    #[error("Streaming error: {0}")]
    Stream(String),

    /// Tool execution or registration failed.
    ///
    /// Occurs when there are problems with tool definitions or execution:
    /// - Tool handler returns an error
    /// - Tool input validation fails
    /// - Tool name collision during registration
    /// - Tool not found when executing
    /// - Invalid tool schema
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// return Err(Error::tool("Tool 'calculator' not found"));
    /// ```
    #[error("Tool execution error: {0}")]
    Tool(String),

    /// Invalid input provided by the user.
    ///
    /// Validation error for user-provided data that doesn't meet requirements:
    /// - Empty prompt string
    /// - Invalid parameter format
    /// - Out of range values
    /// - Malformed input data
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// return Err(Error::invalid_input("Prompt cannot be empty"));
    /// ```
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Request exceeded the configured timeout duration.
    ///
    /// The operation took longer than the timeout specified in AgentOptions.
    /// This is a dedicated variant (no message needed) because the cause is clear.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// return Err(Error::timeout());
    /// ```
    #[error("Request timeout")]
    Timeout,

    /// Miscellaneous error that doesn't fit other categories.
    ///
    /// Catch-all variant for unexpected errors or edge cases that don't fit
    /// into the specific categories above. Should be used sparingly.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// return Err(Error::other("Unexpected condition occurred"));
    /// ```
    #[error("Error: {0}")]
    Other(String),
}

// ============================================================================
// CONVENIENCE CONSTRUCTORS
// ============================================================================

/// Implementation of convenience constructors for creating Error instances.
///
/// These methods provide a more ergonomic API for creating errors compared to
/// directly constructing the enum variants. They accept `impl Into<String>`,
/// allowing callers to pass `&str`, `String`, or any other type that converts to `String`.
impl Error {
    /// Create a new configuration error with a descriptive message.
    ///
    /// Use this when validation fails during `AgentOptions` construction or when
    /// invalid configuration values are detected.
    ///
    /// # Arguments
    ///
    /// * `msg` - Error description explaining what configuration is invalid
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Error;
    ///
    /// let err = Error::config("base_url must be a valid HTTP or HTTPS URL");
    /// assert_eq!(err.to_string(), "Invalid configuration: base_url must be a valid HTTP or HTTPS URL");
    /// ```
    pub fn config(msg: impl Into<String>) -> Self {
        Error::Config(msg.into())
    }

    /// Create a new API error with the server's error message.
    ///
    /// Use this when the API returns an error response (even if the HTTP request
    /// itself succeeded). This typically happens when the server rejects the request
    /// due to invalid parameters, missing resources, or server-side failures.
    ///
    /// # Arguments
    ///
    /// * `msg` - Error message from the API server
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Error;
    ///
    /// let err = Error::api("Model 'invalid-model' not found");
    /// assert_eq!(err.to_string(), "API error: Model 'invalid-model' not found");
    /// ```
    pub fn api(msg: impl Into<String>) -> Self {
        Error::Api {
            status: None,
            message: msg.into(),
        }
    }

    /// Create a new API error from an HTTP error response.
    ///
    /// Prefer this over [`Error::api`] whenever a status code is available: retry
    /// classification reads [`Error::status_code`], and an error built without a status is
    /// treated as non-retryable no matter what its message says.
    ///
    /// # Arguments
    ///
    /// * `status` - HTTP status code from the response
    /// * `msg` - Response body or error message from the API server
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Error;
    ///
    /// let err = Error::api_status(429, "Rate limit exceeded");
    /// assert_eq!(err.to_string(), "API error 429: Rate limit exceeded");
    /// assert_eq!(err.status_code(), Some(429));
    /// ```
    pub fn api_status(status: u16, msg: impl Into<String>) -> Self {
        Error::Api {
            status: Some(status),
            message: msg.into(),
        }
    }

    /// The HTTP status code this error carries, if any.
    ///
    /// Returns `Some` only for [`Error::Api`] values built with [`Error::api_status`], and
    /// `None` for every other variant and for API errors raised without a response status.
    ///
    /// This is the basis for retry classification: transient failures are identified by
    /// status code, never by searching the error message for a status-shaped substring. The
    /// difference is not cosmetic — a substring search classifies
    /// `API error 400 Bad Request: max_tokens 500 too small` as a 500 and retries a request
    /// that can never succeed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Error;
    ///
    /// assert_eq!(Error::api_status(429, "slow down").status_code(), Some(429));
    /// assert_eq!(Error::api("Model 'gpt-4' not found").status_code(), None);
    /// assert_eq!(Error::timeout().status_code(), None);
    /// ```
    pub fn status_code(&self) -> Option<u16> {
        match self {
            Error::Api { status, .. } => *status,
            _ => None,
        }
    }

    /// Create a new streaming error for SSE parsing or stream processing failures.
    ///
    /// Use this when errors occur during Server-Sent Events stream parsing,
    /// such as malformed data, unexpected stream termination, or invalid chunks.
    ///
    /// # Arguments
    ///
    /// * `msg` - Description of the streaming failure
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Error;
    ///
    /// let err = Error::stream("Unexpected end of SSE stream");
    /// assert_eq!(err.to_string(), "Streaming error: Unexpected end of SSE stream");
    /// ```
    pub fn stream(msg: impl Into<String>) -> Self {
        Error::Stream(msg.into())
    }

    /// Create a new tool execution error.
    ///
    /// Use this when tool registration, lookup, or execution fails. This includes
    /// tool handler errors, missing tools, and invalid tool inputs.
    ///
    /// # Arguments
    ///
    /// * `msg` - Description of the tool failure
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Error;
    ///
    /// let err = Error::tool("Calculator tool failed: division by zero");
    /// assert_eq!(err.to_string(), "Tool execution error: Calculator tool failed: division by zero");
    /// ```
    pub fn tool(msg: impl Into<String>) -> Self {
        Error::Tool(msg.into())
    }

    /// Create a new invalid input error for user input validation failures.
    ///
    /// Use this when user-provided data doesn't meet requirements, such as
    /// empty strings, out-of-range values, or malformed data.
    ///
    /// # Arguments
    ///
    /// * `msg` - Description of why the input is invalid
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Error;
    ///
    /// let err = Error::invalid_input("Prompt cannot be empty");
    /// assert_eq!(err.to_string(), "Invalid input: Prompt cannot be empty");
    /// ```
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Error::InvalidInput(msg.into())
    }

    /// Create a new miscellaneous error for cases that don't fit other categories.
    ///
    /// Use this sparingly for unexpected conditions that don't fit into the
    /// more specific error variants.
    ///
    /// # Arguments
    ///
    /// * `msg` - Description of the error
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Error;
    ///
    /// let err = Error::other("Unexpected internal state");
    /// assert_eq!(err.to_string(), "Error: Unexpected internal state");
    /// ```
    pub fn other(msg: impl Into<String>) -> Self {
        Error::Other(msg.into())
    }

    /// Create a timeout error indicating the operation exceeded the time limit.
    ///
    /// Use this when the request or operation takes longer than the configured
    /// timeout duration. No message is needed since the cause is self-explanatory.
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Error;
    ///
    /// let err = Error::timeout();
    /// assert_eq!(err.to_string(), "Request timeout");
    /// ```
    pub fn timeout() -> Self {
        Error::Timeout
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_config() {
        let err = Error::config("Invalid model");
        assert!(matches!(err, Error::Config(_)));
        assert_eq!(err.to_string(), "Invalid configuration: Invalid model");
    }

    #[test]
    fn test_error_api() {
        let err = Error::api("Internal Server Error");
        assert!(matches!(err, Error::Api { status: None, .. }));
        assert_eq!(err.to_string(), "API error: Internal Server Error");

        // A status-carrying error renders the code once, not twice, and exposes it directly.
        let err = Error::api_status(500, "Internal Server Error");
        assert!(matches!(
            err,
            Error::Api {
                status: Some(500),
                ..
            }
        ));
        assert_eq!(err.to_string(), "API error 500: Internal Server Error");
        assert_eq!(err.status_code(), Some(500));
    }

    #[test]
    fn test_error_stream() {
        let err = Error::stream("Connection lost");
        assert!(matches!(err, Error::Stream(_)));
        assert_eq!(err.to_string(), "Streaming error: Connection lost");
    }

    #[test]
    fn test_error_tool() {
        let err = Error::tool("Tool not found");
        assert!(matches!(err, Error::Tool(_)));
        assert_eq!(err.to_string(), "Tool execution error: Tool not found");
    }

    #[test]
    fn test_error_invalid_input() {
        let err = Error::invalid_input("Missing parameter");
        assert!(matches!(err, Error::InvalidInput(_)));
        assert_eq!(err.to_string(), "Invalid input: Missing parameter");
    }

    #[test]
    fn test_error_timeout() {
        let err = Error::timeout();
        assert!(matches!(err, Error::Timeout));
        assert_eq!(err.to_string(), "Request timeout");
    }

    #[test]
    fn test_error_other() {
        let err = Error::other("Something went wrong");
        assert!(matches!(err, Error::Other(_)));
        assert_eq!(err.to_string(), "Error: Something went wrong");
    }

    #[test]
    fn test_error_from_reqwest() {
        // Test that reqwest::Error can be converted
        // This is mostly for compile-time checking
        fn _test_conversion(_e: reqwest::Error) -> Error {
            // This function just needs to compile
            Error::Http(_e)
        }
    }

    #[test]
    fn test_error_from_serde_json() {
        // Test that serde_json::Error can be converted
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Json(_)));
    }

    #[test]
    fn test_result_type_alias() {
        // Test that our Result type alias works correctly
        fn _returns_result() -> Result<i32> {
            Ok(42)
        }

        fn _returns_error() -> Result<i32> {
            Err(Error::timeout())
        }
    }
}
