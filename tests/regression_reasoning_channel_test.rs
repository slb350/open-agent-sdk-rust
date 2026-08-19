//! Regression coverage for the reasoning channel.
//!
//! Reasoning models stream their chain of thought on a side channel: DeepSeek uses
//! `reasoning_content`, OpenRouter uses `reasoning`. Until 0.8.0 the SDK dropped both by
//! accident — `OpenAIDelta` simply did not declare the fields and serde ignored them. That is
//! the right *behaviour* (reasoning prose concatenated into `content` corrupts any structured
//! output a caller parses) but it was not a decision anything defended.
//!
//! These tests make it a contract: reasoning is parsed, routed to its own channel, and never
//! reaches a `ContentBlock::Text`.

mod common;

use common::{
    DONE, blocks_of, collect_events, mixed_chunk, options_with_reasoning, reasoning_chunk,
    reasoning_of, sole_finish_reason, sse_server, text_chunk, text_of_events,
};
use open_agent::{Client, FinishReason};

const DEEPSEEK: &str = "reasoning_content";
const OPENROUTER: &str = "reasoning";

#[tokio::test]
async fn deepseek_reasoning_never_reaches_the_text_channel() {
    let body = reasoning_chunk(DEEPSEEK, "The user wants JSON. Let me think...", None)
        + &text_chunk("{\"ok\":true}", Some("stop"))
        + DONE;

    assert_eq!(
        text_of_events(&collect_events(body, false).await),
        "{\"ok\":true}"
    );
}

#[tokio::test]
async fn openrouter_reasoning_never_reaches_the_text_channel() {
    let body = reasoning_chunk(OPENROUTER, "First I should consider...", None)
        + &text_chunk("{\"ok\":true}", Some("stop"))
        + DONE;

    assert_eq!(
        text_of_events(&collect_events(body, false).await),
        "{\"ok\":true}"
    );
}

#[tokio::test]
async fn reasoning_and_content_in_the_same_delta_stay_separate() {
    let body = mixed_chunk(DEEPSEEK, "thinking out loud", "{\"a\":1}")
        + &text_chunk("", Some("stop"))
        + DONE;

    assert_eq!(
        text_of_events(&collect_events(body, false).await),
        "{\"a\":1}"
    );
}

#[tokio::test]
async fn a_reasoning_only_response_yields_no_text_block() {
    // The failure mode this makes legible: everything went to reasoning, the answer never
    // started, and the cap was hit. Without a Finish event this is an inexplicable empty body.
    let body = reasoning_chunk(DEEPSEEK, "still thinking", Some("length")) + DONE;
    let events = collect_events(body, false).await;

    assert!(
        blocks_of(&events).is_empty(),
        "reasoning must not synthesise a text block: {events:?}"
    );
    assert_eq!(sole_finish_reason(&events), FinishReason::Length);
}

#[tokio::test]
async fn reasoning_is_dropped_entirely_unless_it_is_requested() {
    let body = reasoning_chunk(DEEPSEEK, "private deliberation", None)
        + &text_chunk("answer", Some("stop"))
        + DONE;
    let events = collect_events(body, false).await;

    assert!(
        !events.iter().any(|event| event.as_reasoning().is_some()),
        "reasoning must not be emitted by default: {events:?}"
    );
}

#[tokio::test]
async fn opting_in_surfaces_reasoning_on_its_own_event() {
    let body = reasoning_chunk(DEEPSEEK, "step one. ", None)
        + &reasoning_chunk(DEEPSEEK, "step two.", None)
        + &text_chunk("answer", Some("stop"))
        + DONE;
    let events = collect_events(body, true).await;

    assert_eq!(reasoning_of(&events), "step one. step two.");
    assert_eq!(text_of_events(&events), "answer");
}

#[tokio::test]
async fn reasoning_precedes_the_text_it_produced() {
    let body = mixed_chunk(OPENROUTER, "because", "answer") + &text_chunk("", Some("stop")) + DONE;
    let events = collect_events(body, true).await;

    let reasoning_at = events
        .iter()
        .position(|event| event.as_reasoning().is_some())
        .expect("a reasoning event");
    let text_at = events
        .iter()
        .position(|event| event.as_text().is_some())
        .expect("a text event");

    assert!(reasoning_at < text_at, "unexpected order: {events:?}");
}

#[tokio::test]
async fn a_server_sending_both_channels_does_not_double_count() {
    // Some gateways mirror the same trace on both field names in one delta.
    let body = common::sse_frame(serde_json::json!({
        "index": 0,
        "delta": { "reasoning": "trace", "reasoning_content": "trace" },
        "finish_reason": "stop",
    })) + DONE;

    assert_eq!(reasoning_of(&collect_events(body, true).await), "trace");
}

#[tokio::test]
async fn reasoning_is_not_written_into_conversation_history() {
    let body = reasoning_chunk(DEEPSEEK, "deliberation", None)
        + &text_chunk("answer", Some("stop"))
        + DONE;
    let server = sse_server(body).await;
    let mut client = Client::new(options_with_reasoning(&server, true)).expect("client");

    client.send("hi").await.expect("send");
    while client.receive().await.expect("receive").is_some() {}

    let assistant = client
        .history()
        .iter()
        .find(|message| message.role == open_agent::MessageRole::Assistant)
        .expect("an assistant message");
    assert_eq!(common::message_text(assistant), "answer");
    assert_eq!(client.reasoning(), Some("deliberation"));
}

#[tokio::test]
async fn unknown_delta_fields_are_still_ignored() {
    // Declaring the reasoning fields must not turn serde strict on everything else.
    let body = common::sse_frame(serde_json::json!({
        "index": 0,
        "delta": { "content": "hi", "some_future_field": {"nested": true} },
        "finish_reason": "stop",
    })) + DONE;

    assert_eq!(text_of_events(&collect_events(body, false).await), "hi");
}

#[tokio::test]
async fn reasoning_survives_every_round_of_the_auto_execution_loop() {
    // Each round of the tool loop is its own stream, and starting one resets the per-stream
    // observations. Reasoning must not be reset with them: the deliberation that chose the
    // tools is exactly the part worth keeping, so it accumulates across the whole turn
    // rather than reporting only the final round.
    let body = reasoning_chunk(DEEPSEEK, "round;", None)
        + &common::tool_chunk("call_1", "echo", "{}")
        + &text_chunk("", Some("tool_calls"))
        + DONE;
    let server = sse_server(body).await;

    let echo = open_agent::tool("echo", "Echoes")
        .build(|_| async move { Ok(serde_json::json!({"ok": true})) });

    let options = open_agent::AgentOptions::builder()
        .model("m")
        .base_url(format!("{}/v1", server.uri()))
        .tools(vec![echo])
        .auto_execute_tools(true)
        .max_tool_iterations(2)
        .include_reasoning(true)
        .build()
        .expect("options build");

    let mut client = Client::new(options).expect("client");
    client.send("go").await.expect("send");
    while client.receive().await.expect("receive").is_some() {}

    let reasoning = client.reasoning().expect("reasoning was captured");
    assert!(
        reasoning.matches("round;").count() > 1,
        "reasoning from earlier rounds must not be discarded, got {reasoning:?}"
    );
}
