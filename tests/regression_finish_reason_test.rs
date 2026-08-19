//! Regression coverage for surfacing `finish_reason` to callers.
//!
//! Before 0.8.0 the stream driver consumed `finish_reason` to decide when to flush and then
//! discarded it, so a response truncated at the token cap (`"length"`) was indistinguishable
//! from a complete one (`"stop"`). A caller parsing structured output could not tell "ask
//! again with a larger budget" apart from "this model will not answer in the requested
//! format".
//!
//! The contract asserted here: every stream ends with exactly one [`StreamEvent::Finish`],
//! it is the last event, and a server that never sends `finish_reason` yields
//! [`FinishReason::Unspecified`] rather than being conflated with `"stop"`.

mod common;

use common::{
    DONE, blocks_of, collect_events, options_for, sole_finish_reason, sse_server, text_chunk,
    text_of_events,
};
use open_agent::{Client, ContentBlock, FinishReason};

#[tokio::test]
async fn normal_completion_reports_stop() {
    let events = collect_events(text_chunk("done", Some("stop")) + DONE, false).await;
    assert_eq!(sole_finish_reason(&events), FinishReason::Stop);
    assert_eq!(text_of_events(&events), "done");
}

#[tokio::test]
async fn truncation_at_the_token_cap_reports_length() {
    let events = collect_events(text_chunk("Let me think abo", Some("length")) + DONE, false).await;
    assert_eq!(sole_finish_reason(&events), FinishReason::Length);
    // The partial content is still delivered — the caller decides what to do with it.
    assert_eq!(text_of_events(&events), "Let me think abo");
}

#[tokio::test]
async fn a_server_that_never_sends_finish_reason_reports_unspecified() {
    // llama.cpp and several local gateways stream content and then just stop.
    let events = collect_events(text_chunk("payload", None) + DONE, false).await;
    assert_eq!(sole_finish_reason(&events), FinishReason::Unspecified);
    assert_eq!(text_of_events(&events), "payload");
}

#[tokio::test]
async fn unspecified_is_distinct_from_stop() {
    let unspecified = collect_events(text_chunk("x", None) + DONE, false).await;
    let stopped = collect_events(text_chunk("x", Some("stop")) + DONE, false).await;

    assert_ne!(
        sole_finish_reason(&unspecified),
        sole_finish_reason(&stopped),
        "a silent server must not be reported as a clean stop"
    );
}

#[tokio::test]
async fn tool_call_completion_reports_tool_calls() {
    let body = common::tool_chunk("call_1", "search", "{\"q\":\"rust\"}")
        + &text_chunk("", Some("tool_calls"))
        + DONE;
    let events = collect_events(body, false).await;

    assert_eq!(sole_finish_reason(&events), FinishReason::ToolCalls);
    assert!(matches!(
        blocks_of(&events).as_slice(),
        [ContentBlock::ToolUse(_)]
    ));
}

#[tokio::test]
async fn an_unrecognised_reason_is_preserved_verbatim() {
    let body = text_chunk("x", Some("insufficient_system_resource")) + DONE;
    assert_eq!(
        sole_finish_reason(&collect_events(body, false).await),
        FinishReason::Other("insufficient_system_resource".to_string())
    );
}

#[tokio::test]
async fn an_empty_stream_still_reports_a_finish_event() {
    let events = collect_events(DONE.to_string(), false).await;
    assert_eq!(sole_finish_reason(&events), FinishReason::Unspecified);
    assert!(
        blocks_of(&events).is_empty(),
        "expected no content blocks, got {events:?}"
    );
}

#[tokio::test]
async fn a_finish_reason_stream_emits_exactly_one_finish_event() {
    // The end-of-transport finalize must not append a second Finish after `stop`.
    let body = text_chunk("Hello", None) + &text_chunk(" world", Some("stop")) + DONE;
    let events = collect_events(body, false).await;

    assert_eq!(sole_finish_reason(&events), FinishReason::Stop);
    assert_eq!(text_of_events(&events), "Hello world");
}

#[tokio::test]
async fn the_client_exposes_the_finish_reason_of_the_last_stream() {
    let server = sse_server(text_chunk("capped", Some("length")) + DONE).await;
    let mut client = Client::new(options_for(&server)).expect("client");

    assert_eq!(client.finish_reason(), None, "nothing sent yet");

    client.send("hi").await.expect("send");
    while client.receive().await.expect("receive").is_some() {}

    assert_eq!(client.finish_reason(), Some(&FinishReason::Length));
}

#[tokio::test]
async fn the_client_finish_reason_resets_on_the_next_send() {
    let server = sse_server(text_chunk("x", Some("stop")) + DONE).await;
    let mut client = Client::new(options_for(&server)).expect("client");

    client.send("hi").await.expect("send");
    while client.receive().await.expect("receive").is_some() {}
    assert_eq!(client.finish_reason(), Some(&FinishReason::Stop));

    // A new request must not report the previous turn's reason before it completes.
    client.send("again").await.expect("send");
    assert_eq!(client.finish_reason(), None);
}

#[tokio::test]
async fn the_auto_loop_reports_its_own_iteration_cap_not_the_models_reason() {
    // The mock answers every request with the same tool call, so the loop runs until it hits
    // the cap. The last stream's own reason is "tool_calls" — accurate for that generation,
    // but the operation stopped because the SDK stopped it, and that is what a caller asking
    // "why did this stop?" needs to know.
    let body =
        common::tool_chunk("call_1", "echo", "{}") + &text_chunk("", Some("tool_calls")) + DONE;
    let server = sse_server(body).await;

    let echo = open_agent::tool("echo", "Echoes")
        .build(|_| async move { Ok(serde_json::json!({"ok": true})) });

    let options = open_agent::AgentOptions::builder()
        .model("m")
        .base_url(format!("{}/v1", server.uri()))
        .tools(vec![echo])
        .auto_execute_tools(true)
        .max_tool_iterations(2)
        .build()
        .expect("options build");

    let mut client = Client::new(options).expect("client");
    client.send("go").await.expect("send");
    while client.receive().await.expect("receive").is_some() {}

    assert_eq!(
        client.finish_reason(),
        Some(&FinishReason::MaxToolIterations),
        "the cap is the SDK's decision, not the model's"
    );
}

#[tokio::test]
async fn the_iteration_cap_reason_never_appears_on_a_stream() {
    // MaxToolIterations is client-side only: no server can produce it, and from_wire must not
    // synthesise it from a lookalike string.
    assert_eq!(
        FinishReason::from_wire("max_tool_iterations"),
        FinishReason::Other("max_tool_iterations".to_string())
    );
    assert_eq!(
        sole_finish_reason(
            &collect_events(text_chunk("x", Some("tool_calls")) + DONE, false).await
        ),
        FinishReason::ToolCalls
    );
}
