//! Regression coverage for errors a server reports inside an OpenAI-protocol stream.
//!
//! Once the `200 OK` headers are sent a server can no longer change the status, so a failure
//! after that point arrives as an SSE event. Three servers the README lists document three
//! shapes of it:
//!
//! - vLLM sends `data: {"error": {"message", "type", "param", "code": <int>}}`, then `[DONE]`
//!   (`create_streaming_error_response`, `vllm/entrypoints/serve/engine/protocol.py`).
//! - llama.cpp sends `data: {"error": {"code": <int>, "message", "type"}}`
//!   (`format_error_response` and the streaming branch of `server-context.cpp`).
//! - OpenRouter sends a full chunk object with a top-level `error` beside a choice whose
//!   `finish_reason` is `"error"` (<https://openrouter.ai/docs/api-reference/errors>).
//!
//! The SDK used to reject the first two as `Failed to parse SSE event data: missing field
//! `id``, a retryable stream error that hid the server's message, and to swallow the third as
//! an ordinary `Finish(Other("error"))` so a `Client` recorded the partial answer as a
//! completed turn. Each is now an `Error::Api` carrying the message and, when the server
//! gave an HTTP status as `code`, that status, so `is_retryable_error` can tell a rate limit
//! from a prompt that will never fit.

mod common;

use common::{DONE, message_text, options_for, sole_finish_reason, sse_server, text_chunk};
use futures::StreamExt;
use open_agent::retry::is_retryable_error;
use open_agent::{Client, ContentBlock, Error, FinishReason, MessageRole, StreamEvent, query};
use serde_json::json;

/// A frame carrying exactly `payload`, with no chunk fields around it.
fn raw_frame(payload: serde_json::Value) -> String {
    format!("data: {payload}\n\n")
}

/// OpenRouter's documented shape: a complete chunk with `error` beside a terminating choice.
fn openrouter_error_frame(code: u16, message: &str) -> String {
    raw_frame(json!({
        "id": "gen-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "provider": "P",
        "error": { "code": code, "message": message },
        "choices": [{
            "index": 0,
            "delta": { "content": "" },
            "finish_reason": "error",
        }],
    }))
}

/// Every item `query()` yields, errors included, until the stream ends.
async fn results_of(body: String) -> Vec<Result<StreamEvent, Error>> {
    let server = sse_server(body).await;
    let options = options_for(&server);
    let mut stream = query("hi", &options).await.expect("start query");
    let mut results = Vec::new();
    while let Some(item) = stream.next().await {
        results.push(item);
    }
    results
}

/// The one error in `results`, which must be the only one.
fn sole_error(results: &[Result<StreamEvent, Error>]) -> &Error {
    let errors: Vec<&Error> = results.iter().filter_map(|r| r.as_ref().err()).collect();
    assert_eq!(
        errors.len(),
        1,
        "expected exactly one error, got {results:?}"
    );
    errors[0]
}

fn text_before_the_error(results: &[Result<StreamEvent, Error>]) -> String {
    results
        .iter()
        .filter_map(|result| result.as_ref().ok())
        .filter_map(StreamEvent::as_text)
        .collect()
}

#[tokio::test]
async fn a_standalone_error_event_keeps_the_servers_message_and_status() {
    let body = text_chunk("partial", None)
        + &raw_frame(json!({ "error": {
            "code": 400,
            "message": "the prompt is longer than the context",
            "type": "exceed_context_size_error",
        }}))
        + DONE;

    let results = results_of(body).await;
    let error = sole_error(&results);

    assert!(
        matches!(error, Error::Api { status: Some(400), message }
            if message == "exceed_context_size_error: the prompt is longer than the context"),
        "unexpected error: {error:?}"
    );
    // A prompt that never fits will not fit on the next attempt either.
    assert!(!is_retryable_error(error));
    // Content delivered before the failure is still delivered.
    assert_eq!(text_before_the_error(&results), "partial");
}

#[tokio::test]
async fn a_transient_status_in_an_error_event_is_retryable() {
    let body = raw_frame(json!({ "error": {
        "code": 503,
        "message": "no slot available",
        "type": "unavailable_error",
    }})) + DONE;

    let results = results_of(body).await;
    let error = sole_error(&results);

    assert_eq!(error.status_code(), Some(503));
    assert!(is_retryable_error(error));
}

#[tokio::test]
async fn an_error_event_without_an_http_status_is_not_retryable() {
    // OpenAI's own error objects carry a string `code`, which is not a status.
    let body = raw_frame(json!({ "error": {
        "message": "This model's maximum context length is exceeded",
        "type": "invalid_request_error",
        "code": "context_length_exceeded",
    }})) + DONE;

    let results = results_of(body).await;
    let error = sole_error(&results);

    assert!(
        matches!(error, Error::Api { status: None, message }
            if message
                == "invalid_request_error: This model's maximum context length is exceeded"),
        "unexpected error: {error:?}"
    );
    assert!(!is_retryable_error(error));
}

#[tokio::test]
async fn openrouters_terminating_error_chunk_becomes_an_error() {
    let body =
        text_chunk("partial", None) + &openrouter_error_frame(429, "Rate limit exceeded") + DONE;

    let results = results_of(body).await;
    let error = sole_error(&results);

    assert!(
        matches!(error, Error::Api { status: Some(429), message } if message == "Rate limit exceeded"),
        "unexpected error: {error:?}"
    );
    assert!(is_retryable_error(error));
    assert_eq!(text_before_the_error(&results), "partial");
}

#[tokio::test]
async fn a_terminating_error_reason_without_an_error_object_is_still_a_finish() {
    // Nothing says what went wrong, so there is no message to report; the reason itself stays
    // visible to the caller exactly as before.
    let events: Vec<StreamEvent> = results_of(text_chunk("x", Some("error")) + DONE)
        .await
        .into_iter()
        .map(|result| result.expect("no error was reported"))
        .collect();

    assert_eq!(
        sole_finish_reason(&events),
        FinishReason::Other("error".to_string())
    );
}

#[tokio::test]
async fn text_that_merely_quotes_an_error_object_is_content() {
    let quoted = r#"{"error": {"code": 500, "message": "boom"}}"#;
    let events: Vec<StreamEvent> = results_of(text_chunk(quoted, Some("stop")) + DONE)
        .await
        .into_iter()
        .map(|result| result.expect("quoted text is not an error"))
        .collect();

    assert_eq!(common::text_of_events(&events), quoted);
    assert_eq!(sole_finish_reason(&events), FinishReason::Stop);
}

async fn client_for(body: String) -> (wiremock::MockServer, Client) {
    let server = sse_server(body).await;
    let client = Client::new(options_for(&server)).expect("client builds");
    (server, client)
}

#[tokio::test]
async fn a_client_reports_an_error_event_and_discards_the_partial_turn() {
    let error_frames = [
        raw_frame(
            json!({ "error": { "code": 500, "message": "engine died", "type": "server_error" }}),
        ),
        openrouter_error_frame(502, "upstream disconnected"),
    ];
    for error_frame in error_frames {
        let (_server, mut client) =
            client_for(text_chunk("half an ans", None) + &error_frame + DONE).await;
        client.send("question").await.expect("send");

        let first = client.receive().await.expect("text arrives first");
        assert!(matches!(first, Some(ContentBlock::Text(ref t)) if t.text == "half an ans"));

        let error = client.receive().await.expect_err("the failure is reported");
        assert_eq!(error.status_code().map(|s| s / 100), Some(5));
        assert!(is_retryable_error(&error));

        // The turn that failed is not recorded as if it had completed.
        let roles: Vec<_> = client.history().iter().map(|m| m.role.clone()).collect();
        assert_eq!(roles, [MessageRole::User]);
        assert_eq!(message_text(&client.history()[0]), "question");
    }
}
