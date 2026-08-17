//! Regression coverage for leaving `max_tokens` unset.
//!
//! The builder used to substitute an implicit 4096 cap when `.max_tokens()` was never called,
//! so there was no way to express "no cap, let the server decide". Modern long-context and
//! reasoning models are truncated mid-thought by that default, which surfaces as an
//! unparseable partial response rather than an obvious client-imposed limit.
//!
//! Builder-level `max_tokens` validation is covered by the unit tests in
//! `src/types/tests/core.rs`; these assert what actually reaches the wire.

mod common;

use common::{DONE, options_for, sse_server};
use futures::StreamExt;
use open_agent::{AgentOptions, query};

/// Issues one `query()` against a mock server and returns the request body it saw.
async fn captured_request_body(max_tokens: Option<u32>) -> serde_json::Value {
    let server = sse_server(DONE).await;

    let options = match max_tokens {
        Some(tokens) => AgentOptions::builder()
            .model("m")
            .base_url(format!("{}/v1", server.uri()))
            .max_tokens(tokens)
            .build()
            .expect("options build"),
        None => options_for(&server),
    };

    let mut stream = query("hi", &options).await.expect("start query");
    while stream.next().await.is_some() {}

    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(requests.len(), 1);
    requests[0].body_json().expect("request body is JSON")
}

#[test]
fn max_tokens_is_absent_when_never_set() {
    let options = AgentOptions::builder()
        .model("m")
        .base_url("http://x/v1")
        .build()
        .expect("options build");

    assert_eq!(options.max_tokens(), None);
}

#[tokio::test]
async fn unset_max_tokens_is_omitted_from_the_wire_request() {
    let body = captured_request_body(None).await;
    assert!(
        body.get("max_tokens").is_none(),
        "max_tokens should be omitted entirely, got: {body}"
    );
}

#[tokio::test]
async fn set_max_tokens_is_sent_on_the_wire() {
    let body = captured_request_body(Some(256)).await;
    assert_eq!(body.get("max_tokens"), Some(&serde_json::json!(256)));
}
