//! Regression coverage for OpenAI-compatible servers that omit a tool call's `index`.
//!
//! Google's Gemini OpenAI-compatibility endpoint streams each tool call whole, in one delta,
//! with an `id`, a `name` and complete `arguments` but no `index`
//! (<https://github.com/n8n-io/n8n/issues/39339>; the Vercel AI SDK's chunk schema carries the
//! comment "google does not send index").
//!
//! `index` was required, so such a chunk failed the whole stream with `missing field
//! `index``, a retryable stream error, and the model's tool call was lost. Without an index
//! there is nothing to route by, so a delta that names a new call (it carries an `id`) opens a
//! new call, and one that does not continues the most recent unindexed call. Defaulting every
//! missing index to 0 would instead merge parallel calls into one.

mod common;

use common::{DONE, blocks_of, collect_events, options_for, sole_finish_reason, sse_server};
use open_agent::{Client, ContentBlock, FinishReason};
use serde_json::json;

fn frame(chunk: serde_json::Value) -> String {
    format!("data: {chunk}\n\n")
}

fn chunk(tool_calls: serde_json::Value, finish: Option<&str>) -> String {
    frame(json!({
        "choices": [{
            "index": 0,
            "delta": { "role": "assistant", "tool_calls": tool_calls },
            "finish_reason": finish,
        }],
    }))
}

fn calls_of(blocks: &[ContentBlock]) -> Vec<(String, String, serde_json::Value)> {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolUse(call) => Some((
                call.id().to_string(),
                call.name().to_string(),
                call.input().clone(),
            )),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_whole_tool_call_without_an_index_is_delivered() {
    // Gemini's shape: id, name and full arguments in one delta, no index.
    let body = chunk(
        json!([{
            "id": "function-call-1",
            "type": "function",
            "function": { "name": "ask-user", "arguments": "{\"question\":\"Which CRM?\"}" },
        }]),
        Some("tool_calls"),
    ) + DONE;

    let events = collect_events(body, false).await;

    assert_eq!(
        calls_of(&blocks_of(&events)),
        [(
            "function-call-1".to_string(),
            "ask-user".to_string(),
            json!({ "question": "Which CRM?" })
        )]
    );
    assert_eq!(sole_finish_reason(&events), FinishReason::ToolCalls);
}

#[tokio::test]
async fn parallel_calls_without_indexes_stay_separate_in_arrival_order() {
    // Two calls in one delta, then a third in its own chunk.
    let body = chunk(
        json!([
            { "id": "call_a", "function": { "name": "first", "arguments": "{\"n\":1}" } },
            { "id": "call_b", "function": { "name": "second", "arguments": "{\"n\":2}" } },
        ]),
        None,
    ) + &chunk(
        json!([{ "id": "call_c", "function": { "name": "third", "arguments": "{\"n\":3}" } }]),
        Some("tool_calls"),
    ) + DONE;

    let events = collect_events(body, false).await;

    let calls = calls_of(&blocks_of(&events));
    let seen: Vec<(&str, &str, &serde_json::Value)> = calls
        .iter()
        .map(|(id, name, input)| (id.as_str(), name.as_str(), input))
        .collect();
    assert_eq!(
        seen,
        [
            ("call_a", "first", &json!({ "n": 1 })),
            ("call_b", "second", &json!({ "n": 2 })),
            ("call_c", "third", &json!({ "n": 3 })),
        ]
    );
}

#[tokio::test]
async fn argument_fragments_without_an_id_continue_the_unindexed_call() {
    let body = chunk(
        json!([{ "id": "call_a", "function": { "name": "search", "arguments": "{\"q\":" } }]),
        None,
    ) + &chunk(
        json!([{ "function": { "arguments": "\"rust\"}" } }]),
        Some("tool_calls"),
    ) + DONE;

    let events = collect_events(body, false).await;

    assert_eq!(
        calls_of(&blocks_of(&events)),
        [(
            "call_a".to_string(),
            "search".to_string(),
            json!({ "q": "rust" })
        )]
    );
}

#[tokio::test]
async fn a_call_after_the_finish_reason_drain_is_delivered_separately() {
    // The finish_reason chunk drains `call_a`; the next call is its own, not a continuation.
    let body = chunk(
        json!([{ "id": "call_a", "function": { "name": "first", "arguments": "{}" } }]),
        Some("tool_calls"),
    ) + &chunk(
        json!([{ "id": "call_b", "function": { "name": "second", "arguments": "{}" } }]),
        None,
    ) + DONE;

    let events = collect_events(body, false).await;

    let names: Vec<String> = calls_of(&blocks_of(&events))
        .into_iter()
        .map(|(_, name, _)| name)
        .collect();
    assert_eq!(names, ["first", "second"]);
}

#[tokio::test]
async fn indexed_calls_are_unaffected() {
    let body = chunk(
        json!([
            { "index": 1, "id": "call_b", "function": { "name": "second", "arguments": "{}" } },
            { "index": 0, "id": "call_a", "function": { "name": "first", "arguments": "{}" } },
        ]),
        Some("tool_calls"),
    ) + DONE;

    let events = collect_events(body, false).await;

    let names: Vec<String> = calls_of(&blocks_of(&events))
        .into_iter()
        .map(|(_, name, _)| name)
        .collect();
    assert_eq!(names, ["first", "second"]);
}

#[tokio::test]
async fn a_null_index_is_treated_as_missing() {
    let body = chunk(
        json!([{ "index": null, "id": "call_a", "function": { "name": "first", "arguments": "{}" } }]),
        Some("tool_calls"),
    ) + DONE;

    let events = collect_events(body, false).await;

    assert_eq!(calls_of(&blocks_of(&events)).len(), 1);
}

#[tokio::test]
async fn an_unindexed_call_reaches_a_client_with_its_finish_reason() {
    let body = chunk(
        json!([{ "id": "call_a", "function": { "name": "lookup", "arguments": "{\"k\":\"v\"}" } }]),
        Some("tool_calls"),
    ) + DONE;
    let server = sse_server(body).await;
    let mut client = Client::new(options_for(&server)).expect("client");

    client.send("hi").await.expect("send");
    let mut received = Vec::new();
    while let Some(block) = client.receive().await.expect("receive") {
        received.push(block);
    }

    assert_eq!(
        calls_of(&received),
        [(
            "call_a".to_string(),
            "lookup".to_string(),
            json!({ "k": "v" })
        )]
    );
    assert_eq!(client.finish_reason(), Some(&FinishReason::ToolCalls));
}
