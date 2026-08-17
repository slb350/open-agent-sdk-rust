//! Regression coverage for HTTP status classification in `is_retryable_error`.
//!
//! Classification used to be a substring search over the whole error message, which both
//! missed 429 (the canonical transient error) and produced false positives for any 4xx whose
//! body happened to mention a 5xx-looking number. The status is now structured data on
//! `Error::Api`, so there is nothing left to misparse.

use open_agent::Error;
use open_agent::retry::{RetryConfig, is_retryable_error, retry_with_backoff_conditional};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[test]
fn rate_limiting_is_retryable() {
    assert!(is_retryable_error(&Error::api_status(429, "slow down")));
}

#[test]
fn request_timeout_and_overloaded_are_retryable() {
    assert!(is_retryable_error(&Error::api_status(408, "took too long")));
    assert!(is_retryable_error(&Error::api_status(529, "overloaded")));
}

#[test]
fn server_errors_remain_retryable() {
    for status in [500, 502, 503, 504] {
        assert!(
            is_retryable_error(&Error::api_status(status, "upstream failed")),
            "{status} should be retryable"
        );
    }
}

#[test]
fn a_client_error_mentioning_a_5xx_number_is_not_retryable() {
    // The body contains "500", which a substring-matching classifier read as a server error.
    let error = Error::api_status(400, "max_tokens 500 too small");

    assert!(!is_retryable_error(&error));
    assert_eq!(error.status_code(), Some(400));
}

#[test]
fn other_client_errors_are_not_retryable() {
    for status in [401, 404, 422, 501] {
        assert!(
            !is_retryable_error(&Error::api_status(status, "nope")),
            "{status} should not be retryable"
        );
    }
}

#[test]
fn api_errors_without_a_status_are_not_retryable() {
    let error = Error::api("Model 'invalid-model' not found");

    assert_eq!(error.status_code(), None);
    assert!(!is_retryable_error(&error));
}

#[test]
fn non_api_variants_keep_their_classification() {
    // Stream errors are transient: a dropped connection mid-response is worth retrying.
    assert!(is_retryable_error(&Error::stream("connection reset")));
    assert!(is_retryable_error(&Error::timeout()));
    // Everything else stays on the conservative non-retryable default.
    assert!(!is_retryable_error(&Error::tool("handler panicked")));
    assert!(!is_retryable_error(&Error::config("bad base_url")));
    assert!(!is_retryable_error(&Error::invalid_input("empty prompt")));

    assert_eq!(Error::timeout().status_code(), None);
}

#[test]
fn api_errors_render_their_status_exactly_once() {
    // The message used to embed its own "API error <status>" prefix on top of the variant's
    // Display, rendering as "API error: API error 503 ...".
    assert_eq!(
        Error::api_status(503, "upstream down").to_string(),
        "API error 503: upstream down"
    );
    assert_eq!(
        Error::api("no status here").to_string(),
        "API error: no status here"
    );
}

#[tokio::test(start_paused = true)]
async fn conditional_retry_backs_off_on_429_and_gives_up_on_400() {
    let config = RetryConfig::new()
        .with_max_attempts(3)
        .with_initial_delay(Duration::from_millis(1));

    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let result = retry_with_backoff_conditional(config.clone(), move || {
        counter.fetch_add(1, Ordering::SeqCst);
        async { Err::<(), Error>(Error::api_status(429, "slow down")) }
    })
    .await;
    assert!(result.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 3, "429 should be retried");

    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let result = retry_with_backoff_conditional(config, move || {
        counter.fetch_add(1, Ordering::SeqCst);
        async { Err::<(), Error>(Error::api_status(400, "max_tokens 500 too small")) }
    })
    .await;
    assert!(result.is_err());
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "400 should fail on the first attempt"
    );
}
