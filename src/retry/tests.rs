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
fn calculate_delay_grows_by_exact_multiples_of_the_initial_delay() {
    let config = RetryConfig::new()
        .with_initial_delay(Duration::from_millis(100))
        .with_max_delay(Duration::from_secs(60))
        .with_backoff_multiplier(2.0)
        .with_jitter_factor(0.0);

    // base * multiplier^attempt, not base + multiplier^attempt: the delay must double each
    // time rather than creep upward by a constant.
    assert_eq!(config.calculate_delay(0), Duration::from_millis(100));
    assert_eq!(config.calculate_delay(1), Duration::from_millis(200));
    assert_eq!(config.calculate_delay(2), Duration::from_millis(400));
    assert_eq!(config.calculate_delay(3), Duration::from_millis(800));
}

#[test]
fn calculate_delay_is_exact_when_jitter_is_disabled() {
    let config = RetryConfig::new()
        .with_initial_delay(Duration::from_millis(400))
        .with_max_delay(Duration::from_secs(60))
        .with_backoff_multiplier(1.0)
        .with_jitter_factor(0.0);

    // A zero jitter factor must scale the jitter range to zero, not leave the capped delay
    // untouched as the range. Repeat so a randomised range cannot coincidentally match.
    for _ in 0..64 {
        assert_eq!(config.calculate_delay(0), Duration::from_millis(400));
    }
}

#[test]
fn jitter_never_pushes_the_delay_past_max_delay() {
    let config = RetryConfig::new()
        .with_initial_delay(Duration::from_millis(1_000))
        .with_max_delay(Duration::from_millis(1_000))
        .with_backoff_multiplier(1.0)
        .with_jitter_factor(1.0);

    // Jitter is applied after the cap, so without a second clamp roughly half of these draws
    // would land above max_delay.
    for _ in 0..256 {
        let delay = config.calculate_delay(0);
        assert!(delay <= Duration::from_millis(1_000), "{delay:?} exceeds cap");
    }
}

#[test]
fn calculate_delay_is_capped_at_max_delay() {
    let config = RetryConfig::new()
        .with_initial_delay(Duration::from_millis(1_000))
        .with_max_delay(Duration::from_millis(2_500))
        .with_backoff_multiplier(10.0)
        .with_jitter_factor(0.0);

    assert_eq!(config.calculate_delay(0), Duration::from_millis(1_000));
    assert_eq!(config.calculate_delay(1), Duration::from_millis(2_500));
    assert_eq!(config.calculate_delay(9), Duration::from_millis(2_500));
}

#[test]
fn test_calculate_delay_stays_within_jitter_bounds() {
    let config = RetryConfig::new()
        .with_initial_delay(Duration::from_millis(1_000))
        .with_max_delay(Duration::from_secs(60))
        .with_backoff_multiplier(1.0)
        .with_jitter_factor(0.2);

    let mut saw_below_centre = false;
    let mut saw_above_centre = false;

    for _ in 0..256 {
        let delay = config.calculate_delay(0);

        // A 0.2 jitter factor spreads the 1000ms delay over ±100ms. The upper bound is
        // exclusive because the random draw is drawn from [0, 1).
        assert!(delay >= Duration::from_millis(900), "{delay:?} below range");
        assert!(delay < Duration::from_millis(1_100), "{delay:?} above range");

        saw_below_centre |= delay < Duration::from_millis(1_000);
        saw_above_centre |= delay > Duration::from_millis(1_000);
    }

    // The jitter must scale with the range, not collapse to one end of it. Dividing by the
    // range instead of multiplying, for instance, pins every delay to the floor.
    assert!(saw_below_centre, "jitter never fell below the base delay");
    assert!(saw_above_centre, "jitter never rose above the base delay");
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

#[tokio::test(start_paused = true)]
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

#[tokio::test(start_paused = true)]
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

#[tokio::test]
async fn http_transport_errors_are_retryable() {
    // Port 1 is reserved and refuses connections, which yields a genuine reqwest transport
    // error. `reqwest::Error` cannot be constructed directly, so this test lives inside the
    // crate where the dependency is reachable.
    let transport_error = reqwest::Client::new()
        .get("http://127.0.0.1:1/")
        .send()
        .await
        .expect_err("a request to a closed port must fail");

    assert!(is_retryable_error(&Error::Http(transport_error)));
}

/// Which retry driver a timing test exercises.
#[derive(Clone, Copy)]
enum Driver {
    Unconditional,
    Conditional,
}

/// Exhausts every attempt against `driver` and returns the total time spent sleeping.
///
/// Must be called from a `#[tokio::test(start_paused = true)]` test: with the clock paused,
/// `sleep` returns immediately while the virtual clock advances by exactly the requested
/// duration, so the elapsed time is the exact sum of the backoff sleeps. That makes the
/// number of sleeps directly assertable, with no wall-clock tolerance to turn the assertion
/// flaky on a loaded machine.
async fn elapsed_while_exhausting_attempts(config: RetryConfig, driver: Driver) -> Duration {
    let always_transient = || async { Err::<(), Error>(Error::timeout()) };

    let start = tokio::time::Instant::now();
    let result = match driver {
        Driver::Unconditional => retry_with_backoff(config, always_transient).await,
        Driver::Conditional => retry_with_backoff_conditional(config, always_transient).await,
    };

    assert!(result.is_err());
    start.elapsed()
}

#[tokio::test(start_paused = true)]
async fn backoff_sleeps_between_attempts_but_not_after_the_last_one() {
    let config = RetryConfig::new()
        .with_max_attempts(3)
        .with_initial_delay(Duration::from_millis(100))
        .with_backoff_multiplier(1.0)
        .with_jitter_factor(0.0);

    // Three attempts means exactly two gaps: 100ms + 100ms. Sleeping after the final attempt
    // would add a third, and skipping the guard entirely would remove both.
    assert_eq!(
        elapsed_while_exhausting_attempts(config, Driver::Unconditional).await,
        Duration::from_millis(200)
    );
}

#[tokio::test(start_paused = true)]
async fn a_single_attempt_never_sleeps() {
    let config = RetryConfig::new()
        .with_max_attempts(1)
        .with_initial_delay(Duration::from_millis(100))
        .with_backoff_multiplier(1.0)
        .with_jitter_factor(0.0);

    assert_eq!(
        elapsed_while_exhausting_attempts(config, Driver::Unconditional).await,
        Duration::ZERO
    );
}

#[tokio::test(start_paused = true)]
async fn conditional_backoff_sleeps_between_attempts_but_not_after_the_last_one() {
    let config = RetryConfig::new()
        .with_max_attempts(3)
        .with_initial_delay(Duration::from_millis(100))
        .with_backoff_multiplier(1.0)
        .with_jitter_factor(0.0);

    // The conditional driver must apply the same "no sleep after the final attempt" rule.
    assert_eq!(
        elapsed_while_exhausting_attempts(config, Driver::Conditional).await,
        Duration::from_millis(200)
    );
}

#[tokio::test(start_paused = true)]
async fn conditional_backoff_does_not_sleep_before_rejecting_a_permanent_error() {
    let config = RetryConfig::new()
        .with_max_attempts(3)
        .with_initial_delay(Duration::from_millis(100))
        .with_backoff_multiplier(1.0)
        .with_jitter_factor(0.0);

    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = calls.clone();
    let start = tokio::time::Instant::now();
    let result = retry_with_backoff_conditional(config, move || {
        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        async { Err::<(), Error>(Error::config("bad base_url")) }
    })
    .await;

    assert!(result.is_err());
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(start.elapsed(), Duration::ZERO);
}
