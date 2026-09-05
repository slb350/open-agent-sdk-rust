//! SDK errors and the shared [`Result`] alias.
//!
//! `reqwest::Error` and `serde_json::Error` convert through `From`. Model HTTP
//! responses retain their status in [`Error::Api`] for retry classification.

use thiserror::Error;

/// A result using the SDK [`enum@Error`] type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors from configuration, transport, stream decoding, and tool execution.
///
/// `reqwest::Error` and `serde_json::Error` convert automatically through `?`.
#[derive(Error, Debug)]
pub enum Error {
    /// A reqwest transport error, including connection failures and timeouts.
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// A serde JSON serialization or deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid SDK configuration.
    #[error("Invalid configuration: {0}")]
    Config(String),

    /// A model API error with an optional structured HTTP status.
    ///
    /// Use [`Error::api_status`] when a status is available. Retry classification uses
    /// that field, never status-like text in the message.
    #[error("API error{}: {message}", .status.map(|code| format!(" {code}")).unwrap_or_default())]
    Api {
        /// HTTP status code, when the error originated from an HTTP response.
        status: Option<u16>,

        /// Error message or response body reported by the server.
        message: String,
    },

    /// An SSE or stream-processing error.
    #[error("Streaming error: {0}")]
    Stream(String),

    /// A tool lookup or execution error.
    #[error("Tool execution error: {0}")]
    Tool(String),

    /// Invalid input to an SDK operation.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// An explicit SDK timeout error.
    #[error("Request timeout")]
    Timeout,

    /// An error without a more specific category.
    #[error("Error: {0}")]
    Other(String),
}

impl Error {
    /// Creates a configuration error.
    pub fn config(msg: impl Into<String>) -> Self {
        Error::Config(msg.into())
    }

    /// Creates an API error without a status.
    ///
    /// It is non-retryable regardless of the message. Use [`api_status`](Self::api_status)
    /// when the HTTP status is known.
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

    /// Returns the status carried by [`Error::Api`], or `None`.
    ///
    /// No other variant exposes a status through this method.
    pub fn status_code(&self) -> Option<u16> {
        match self {
            Error::Api { status, .. } => *status,
            _ => None,
        }
    }

    /// Creates a stream-processing error.
    pub fn stream(msg: impl Into<String>) -> Self {
        Error::Stream(msg.into())
    }

    /// Creates a tool execution error.
    pub fn tool(msg: impl Into<String>) -> Self {
        Error::Tool(msg.into())
    }

    /// Creates an invalid-input error.
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Error::InvalidInput(msg.into())
    }

    /// Creates an uncategorized error.
    pub fn other(msg: impl Into<String>) -> Self {
        Error::Other(msg.into())
    }

    /// Creates an explicit timeout error.
    pub fn timeout() -> Self {
        Error::Timeout
    }
}

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
    fn test_error_from_serde_json() {
        // Test that serde_json::Error can be converted
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Json(_)));
    }
}
