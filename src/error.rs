//! Error types for the Open Agent SDK

use thiserror::Error;

/// Result type alias using our Error type
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for the SDK
#[derive(Error, Debug)]
pub enum Error {
    /// HTTP request error
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    Config(String),

    /// API error from the model server
    #[error("API error: {0}")]
    Api(String),

    /// Streaming error
    #[error("Streaming error: {0}")]
    Stream(String),

    /// Tool execution error
    #[error("Tool execution error: {0}")]
    Tool(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Timeout error
    #[error("Request timeout")]
    Timeout,

    /// Other errors
    #[error("Error: {0}")]
    Other(String),
}

impl Error {
    /// Create a new config error
    pub fn config(msg: impl Into<String>) -> Self {
        Error::Config(msg.into())
    }

    /// Create a new API error
    pub fn api(msg: impl Into<String>) -> Self {
        Error::Api(msg.into())
    }

    /// Create a new stream error
    pub fn stream(msg: impl Into<String>) -> Self {
        Error::Stream(msg.into())
    }

    /// Create a new tool error
    pub fn tool(msg: impl Into<String>) -> Self {
        Error::Tool(msg.into())
    }

    /// Create a new invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Error::InvalidInput(msg.into())
    }

    /// Create a new other error
    pub fn other(msg: impl Into<String>) -> Self {
        Error::Other(msg.into())
    }

    /// Create a timeout error
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
        let err = Error::api("500 Internal Server Error");
        assert!(matches!(err, Error::Api(_)));
        assert_eq!(err.to_string(), "API error: 500 Internal Server Error");
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
