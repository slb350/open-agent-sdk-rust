//! Retry utilities with exponential backoff
//!
//! This module provides utilities for retrying operations with configurable
//! backoff strategies. Useful for handling transient failures when communicating
//! with LLM servers.
//!
//! # Examples
//!
//! ```rust,no_run
//! use open_agent::retry::{retry_with_backoff, RetryConfig};
//! use std::time::Duration;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = RetryConfig::default()
//!     .with_max_attempts(3)
//!     .with_initial_delay(Duration::from_secs(1));
//!
//! let result = retry_with_backoff(config, || async {
//!     // Your async operation here
//!     Ok::<_, open_agent::Error>(42)
//! }).await?;
//! # Ok(())
//! # }
//! ```

use crate::{Error, Result};
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,

    /// Initial delay before first retry
    pub initial_delay: Duration,

    /// Maximum delay between retries
    pub max_delay: Duration,

    /// Multiplier for exponential backoff (e.g., 2.0 doubles the delay each time)
    pub backoff_multiplier: f64,

    /// Add random jitter to prevent thundering herd (0.0 to 1.0)
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

impl RetryConfig {
    /// Create a new retry configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of attempts
    pub fn with_max_attempts(mut self, attempts: u32) -> Self {
        self.max_attempts = attempts;
        self
    }

    /// Set initial delay
    pub fn with_initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set maximum delay
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set backoff multiplier
    pub fn with_backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Set jitter factor (0.0 to 1.0)
    pub fn with_jitter_factor(mut self, jitter: f64) -> Self {
        self.jitter_factor = jitter.clamp(0.0, 1.0);
        self
    }

    /// Calculate delay for a given attempt with exponential backoff and jitter
    fn calculate_delay(&self, attempt: u32) -> Duration {
        let base_delay_ms = self.initial_delay.as_millis() as f64;
        let exponential_delay = base_delay_ms * self.backoff_multiplier.powi(attempt as i32);

        // Cap at max delay
        let capped_delay = exponential_delay.min(self.max_delay.as_millis() as f64);

        // Add jitter
        let jitter_range = capped_delay * self.jitter_factor;
        let jitter = rand::random::<f64>() * jitter_range;
        let final_delay = capped_delay + jitter - (jitter_range / 2.0);

        Duration::from_millis(final_delay.max(0.0) as u64)
    }
}

/// Retry an async operation with exponential backoff
///
/// # Arguments
///
/// * `config` - Retry configuration
/// * `operation` - Async function to retry
///
/// # Returns
///
/// The result of the operation if successful, or the last error if all retries failed
///
/// # Examples
///
/// ```rust,no_run
/// use open_agent::retry::{retry_with_backoff, RetryConfig};
/// use open_agent::{Client, AgentOptions};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = RetryConfig::default().with_max_attempts(3);
/// let options = AgentOptions::builder()
///     .model("qwen3:8b")
///     .base_url("http://localhost:11434/v1")
///     .build()?;
///
/// let result = retry_with_backoff(config, || async {
///     let mut client = Client::new(options.clone());
///     client.send("Hello").await?;
///     Ok::<_, open_agent::Error>(())
/// }).await?;
/// # Ok(())
/// # }
/// ```
pub async fn retry_with_backoff<F, Fut, T>(config: RetryConfig, mut operation: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut last_error = None;

    for attempt in 0..config.max_attempts {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(err) => {
                last_error = Some(err);

                // Don't sleep after the last attempt
                if attempt < config.max_attempts - 1 {
                    let delay = config.calculate_delay(attempt);
                    sleep(delay).await;
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| Error::other("Retry failed with no error")))
}

/// Determine if an error is retryable
///
/// Returns true for transient errors like network issues, timeouts, and 5xx server errors.
/// Returns false for client errors like invalid requests (4xx) or configuration errors.
pub fn is_retryable_error(error: &Error) -> bool {
    match error {
        Error::Http(_) => true,   // Network errors are retryable
        Error::Timeout => true,   // Timeouts are retryable
        Error::Stream(_) => true, // Stream errors might be transient
        Error::Api(msg) => {
            // Check if it's a 5xx server error (retryable)
            // vs 4xx client error (not retryable)
            msg.contains("500") || msg.contains("502") || msg.contains("503") || msg.contains("504")
        }
        Error::Config(_) => false, // Configuration errors aren't retryable
        Error::InvalidInput(_) => false, // Invalid input isn't retryable
        _ => false,                // Conservative default
    }
}

/// Retry an async operation with exponential backoff, only retrying on retryable errors
///
/// This is a smarter version of `retry_with_backoff` that only retries transient errors.
///
/// # Examples
///
/// ```rust,no_run
/// use open_agent::retry::{retry_with_backoff_conditional, RetryConfig};
/// use open_agent::{Client, AgentOptions};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = RetryConfig::default();
/// let options = AgentOptions::builder()
///     .model("qwen3:8b")
///     .base_url("http://localhost:11434/v1")
///     .build()?;
///
/// let result = retry_with_backoff_conditional(config, || async {
///     let mut client = Client::new(options.clone());
///     client.send("Hello").await?;
///     Ok::<_, open_agent::Error>(())
/// }).await?;
/// # Ok(())
/// # }
/// ```
pub async fn retry_with_backoff_conditional<F, Fut, T>(
    config: RetryConfig,
    mut operation: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut last_error = None;

    for attempt in 0..config.max_attempts {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(err) => {
                // Check if error is retryable
                if !is_retryable_error(&err) {
                    return Err(err);
                }

                last_error = Some(err);

                // Don't sleep after the last attempt
                if attempt < config.max_attempts - 1 {
                    let delay = config.calculate_delay(attempt);
                    sleep(delay).await;
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| Error::other("Retry failed with no error")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_builder() {
        let config = RetryConfig::new()
            .with_max_attempts(5)
            .with_initial_delay(Duration::from_millis(500))
            .with_max_delay(Duration::from_secs(30))
            .with_backoff_multiplier(1.5)
            .with_jitter_factor(0.2);

        assert_eq!(config.max_attempts, 5);
        assert_eq!(config.initial_delay, Duration::from_millis(500));
        assert_eq!(config.max_delay, Duration::from_secs(30));
        assert_eq!(config.backoff_multiplier, 1.5);
        assert_eq!(config.jitter_factor, 0.2);
    }

    #[test]
    fn test_calculate_delay() {
        let config = RetryConfig::new()
            .with_initial_delay(Duration::from_secs(1))
            .with_backoff_multiplier(2.0)
            .with_jitter_factor(0.0); // No jitter for predictable testing

        let delay0 = config.calculate_delay(0);
        let delay1 = config.calculate_delay(1);
        let delay2 = config.calculate_delay(2);

        // Verify exponential growth
        assert!(delay1 > delay0);
        assert!(delay2 > delay1);
    }

    #[tokio::test]
    async fn test_retry_success_on_first_attempt() {
        let config = RetryConfig::new().with_max_attempts(3);

        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count_clone = call_count.clone();
        let result = retry_with_backoff(config, move || {
            count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async { Ok::<i32, Error>(42) }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1); // Should succeed on first attempt
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let config = RetryConfig::new()
            .with_max_attempts(3)
            .with_initial_delay(Duration::from_millis(10));

        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count_clone = call_count.clone();
        let result = retry_with_backoff(config, move || {
            let count = count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            async move {
                if count < 3 {
                    Err(Error::timeout())
                } else {
                    Ok::<i32, Error>(42)
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 3); // Should succeed on third attempt
    }

    #[tokio::test]
    async fn test_retry_exhausts_attempts() {
        let config = RetryConfig::new()
            .with_max_attempts(2)
            .with_initial_delay(Duration::from_millis(10));

        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count_clone = call_count.clone();
        let result = retry_with_backoff(config, move || {
            count_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async { Err::<i32, Error>(Error::timeout()) }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 2); // Should try twice
    }

    #[test]
    fn test_is_retryable_error() {
        assert!(is_retryable_error(&Error::timeout()));
        assert!(is_retryable_error(&Error::api(
            "500 Internal Server Error".to_string()
        )));
        assert!(is_retryable_error(&Error::api(
            "503 Service Unavailable".to_string()
        )));
        assert!(!is_retryable_error(&Error::config(
            "Invalid config".to_string()
        )));
        assert!(!is_retryable_error(&Error::invalid_input(
            "Bad input".to_string()
        )));
    }
}
