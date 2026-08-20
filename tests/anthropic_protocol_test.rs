//! End-to-end coverage of the Anthropic protocol through `query()`.
//!
//! The unit tests under `src/` cover the translation and the accumulator in isolation. What
//! only an integration test can establish is that `stream_request` actually selects them:
//! the path it posts to, the headers it sets, the body it serializes, and the vocabulary it
//! parses the answer with. A protocol wired up to the wrong half of any of those still
//! passes every unit test in the crate.

mod common;

use common::{
    anthropic_frame, anthropic_sse_server, anthropic_text_response, sole_finish_reason, sse_server,
    text_chunk,
};
use futures::StreamExt;
use open_agent::{
    AgentOptions, AgentOptionsBuilder, ApiProtocol, ContentBlock, FinishReason, StreamEvent, query,
};
use wiremock::MockServer;

/// A builder pointing at `server` and speaking Anthropic, for tests that need one more field.
///
/// Returning the builder rather than only the built options is what keeps the shared setup
/// shared: a test that copies these lines to append a single call drifts from them silently,
/// which is how two of the tests below came to be missing their API key.
fn anthropic_builder(server: &MockServer) -> AgentOptionsBuilder {
    AgentOptions::builder()
        .model("k3")
        .base_url(format!("{}/v1", server.uri()))
        .api_key("secret")
        .protocol(ApiProtocol::Anthropic)
}

/// Options pointing at `server`, speaking Anthropic.
fn anthropic_options(server: &MockServer) -> AgentOptions {
    anthropic_builder(server).build().expect("options build")
}

/// A builder pointing at `server` and leaving the protocol at its default.
fn openai_builder(server: &MockServer) -> AgentOptionsBuilder {
    AgentOptions::builder()
        .model("m")
        .base_url(format!("{}/v1", server.uri()))
        .api_key("secret")
}

/// Runs one query and collects every event it yields.
async fn collect(options: &AgentOptions, prompt: &str) -> Vec<StreamEvent> {
    let mut stream = query(prompt, options).await.expect("start query");
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("no stream error"));
    }
    events
}

/// The single request the server received, as parsed JSON.
async fn sole_request_body(server: &MockServer) -> serde_json::Value {
    let requests = server
        .received_requests()
        .await
        .expect("the server records requests");
    assert_eq!(requests.len(), 1, "expected exactly one request");
    serde_json::from_slice(&requests[0].body).expect("body is JSON")
}

#[tokio::test]
async fn an_anthropic_query_posts_to_the_messages_path() {
    // The mock is mounted on /v1/messages only, so reaching it at all is the assertion.
    let server = anthropic_sse_server(anthropic_text_response("hi", "end_turn")).await;
    let events = collect(&anthropic_options(&server), "hello").await;

    assert_eq!(
        events
            .iter()
            .filter_map(StreamEvent::as_text)
            .collect::<String>(),
        "hi"
    );
    assert_eq!(sole_finish_reason(&events), FinishReason::Stop);
}

#[tokio::test]
async fn an_anthropic_query_authenticates_with_x_api_key_and_names_its_schema_version() {
    let server = anthropic_sse_server(anthropic_text_response("hi", "end_turn")).await;
    collect(&anthropic_options(&server), "hello").await;

    let requests = server.received_requests().await.expect("records requests");
    let headers = &requests[0].headers;

    assert_eq!(
        headers.get("x-api-key").map(|v| v.to_str().expect("ascii")),
        Some("secret")
    );
    assert_eq!(
        headers
            .get("anthropic-version")
            .map(|v| v.to_str().expect("ascii")),
        Some("2023-06-01")
    );
    assert!(
        headers.get("authorization").is_none(),
        "a bearer token is the other protocol's credential and Anthropic ignores it, so \
         sending one leaks the key to an endpoint that has no use for it"
    );
}

#[tokio::test]
async fn the_system_prompt_is_sent_as_a_field_not_as_a_message() {
    let server = anthropic_sse_server(anthropic_text_response("hi", "end_turn")).await;
    let options = anthropic_builder(&server)
        .system_prompt("be terse")
        .build()
        .expect("options build");

    collect(&options, "hello").await;
    let body = sole_request_body(&server).await;

    assert_eq!(body["system"], serde_json::json!("be terse"));
    let messages = body["messages"].as_array().expect("messages array");
    assert_eq!(messages.len(), 1, "the system prompt is not a turn: {body}");
    assert_eq!(messages[0]["role"], serde_json::json!("user"));
    assert_eq!(messages[0]["content"], serde_json::json!("hello"));
}

#[tokio::test]
async fn an_unset_temperature_is_omitted_from_the_request() {
    // The reason this matters: kimi-for-coding's k3 rejects the request outright with
    // "only temperature 1 is allowed for this model", so a client-invented default is the
    // difference between a working provider and a hard 400.
    let server = anthropic_sse_server(anthropic_text_response("hi", "end_turn")).await;
    collect(&anthropic_options(&server), "hello").await;

    let body = sole_request_body(&server).await;
    assert!(
        body.get("temperature").is_none(),
        "temperature must be absent, got {body}"
    );
    assert!(
        body.get("max_tokens").is_none(),
        "max_tokens must be absent, got {body}"
    );
}

#[tokio::test]
async fn a_set_temperature_reaches_the_request() {
    let server = anthropic_sse_server(anthropic_text_response("hi", "end_turn")).await;
    let options = anthropic_builder(&server)
        .temperature(0.2)
        .build()
        .expect("options build");

    collect(&options, "hello").await;
    let body = sole_request_body(&server).await;

    assert!(
        (body["temperature"].as_f64().expect("a number") - 0.2).abs() < 1e-6,
        "got {body}"
    );
}

#[tokio::test]
async fn an_anthropic_stop_reason_is_mapped_rather_than_passed_through() {
    let server = anthropic_sse_server(anthropic_text_response("cut", "max_tokens")).await;
    let events = collect(&anthropic_options(&server), "hello").await;

    assert_eq!(
        sole_finish_reason(&events),
        FinishReason::Length,
        "a caller branching on Length must see this truncation"
    );
}

#[tokio::test]
async fn thinking_blocks_reach_the_reasoning_channel_and_not_the_text() {
    let body = [
        anthropic_frame(
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start", "index": 0,
                "content_block": { "type": "thinking", "thinking": "" },
            }),
        ),
        anthropic_frame(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta", "index": 0,
                "delta": { "type": "thinking_delta", "thinking": "deliberating" },
            }),
        ),
        anthropic_frame(
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start", "index": 1,
                "content_block": { "type": "text", "text": "" },
            }),
        ),
        anthropic_frame(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta", "index": 1,
                "delta": { "type": "text_delta", "text": "{\"ok\":true}" },
            }),
        ),
        anthropic_frame(
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn", "stop_sequence": null },
            }),
        ),
        anthropic_frame(
            "message_stop",
            serde_json::json!({ "type": "message_stop" }),
        ),
    ]
    .concat();

    let server = anthropic_sse_server(body).await;
    let options = anthropic_builder(&server)
        .include_reasoning(true)
        .build()
        .expect("options build");

    let events = collect(&options, "hello").await;

    assert_eq!(
        events
            .iter()
            .filter_map(StreamEvent::as_text)
            .collect::<String>(),
        "{\"ok\":true}",
        "a caller parsing this as JSON must not find deliberation spliced into it"
    );
    assert_eq!(
        events
            .iter()
            .filter_map(StreamEvent::as_reasoning)
            .collect::<String>(),
        "deliberating"
    );
}

#[tokio::test]
async fn a_tool_call_arrives_as_a_tool_use_block() {
    let body = [
        anthropic_frame(
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start", "index": 0,
                "content_block": { "type": "tool_use", "id": "toolu_1", "name": "search",
                                   "input": {} },
            }),
        ),
        anthropic_frame(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta", "index": 0,
                "delta": { "type": "input_json_delta", "partial_json": "{\"q\":\"rust\"}" },
            }),
        ),
        anthropic_frame(
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": { "stop_reason": "tool_use", "stop_sequence": null },
            }),
        ),
    ]
    .concat();

    let server = anthropic_sse_server(body).await;
    let events = collect(&anthropic_options(&server), "hello").await;

    match events
        .iter()
        .find_map(StreamEvent::as_block)
        .expect("a content block")
    {
        ContentBlock::ToolUse(call) => {
            assert_eq!(call.id(), "toolu_1");
            assert_eq!(call.name(), "search");
            assert_eq!(*call.input(), serde_json::json!({ "q": "rust" }));
        }
        other => panic!("expected a tool call, got {other:?}"),
    }
    assert_eq!(sole_finish_reason(&events), FinishReason::ToolCalls);
}

#[tokio::test]
async fn an_http_error_on_the_anthropic_path_carries_its_status() {
    let server = MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/messages"))
        .respond_with(wiremock::ResponseTemplate::new(400).set_body_string(
            r#"{"type":"error","error":{"type":"invalid_request_error",
                    "message":"invalid temperature: only 1 is allowed for this model"}}"#,
        ))
        .mount(&server)
        .await;

    // `expect_err` is unavailable here: the Ok type is a boxed stream and does not implement
    // Debug.
    let error = match query("hello", &anthropic_options(&server)).await {
        Ok(_) => panic!("a 400 must fail the request"),
        Err(error) => error,
    };

    assert_eq!(error.status_code(), Some(400));
    assert!(
        error.to_string().contains("only 1 is allowed"),
        "the server's own explanation survives: {error}"
    );
}

#[tokio::test]
async fn the_default_protocol_still_posts_to_chat_completions() {
    // The regression guard for the whole change: an existing configuration that names no
    // protocol must behave exactly as it did before 0.9.0. The mock is mounted on
    // /v1/chat/completions, so selecting the wrong path 404s.
    let server = sse_server(format!(
        "{}{}",
        text_chunk("ok", Some("stop")),
        common::DONE
    ))
    .await;
    let options = openai_builder(&server).build().expect("options build");

    let events = collect(&options, "hello").await;

    assert_eq!(
        events
            .iter()
            .filter_map(StreamEvent::as_text)
            .collect::<String>(),
        "ok"
    );
    assert_eq!(sole_finish_reason(&events), FinishReason::Stop);
}

#[tokio::test]
async fn the_default_protocol_still_sends_a_bearer_token() {
    let server = sse_server(format!(
        "{}{}",
        text_chunk("ok", Some("stop")),
        common::DONE
    ))
    .await;
    let options = openai_builder(&server).build().expect("options build");

    collect(&options, "hello").await;

    let requests = server.received_requests().await.expect("records requests");
    assert_eq!(
        requests[0]
            .headers
            .get("authorization")
            .map(|v| v.to_str().expect("ascii")),
        Some("Bearer secret")
    );
    assert!(
        requests[0].headers.get("x-api-key").is_none(),
        "the Anthropic credential must not leak into an OpenAI request"
    );
}
