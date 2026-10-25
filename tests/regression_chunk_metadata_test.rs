//! Regression coverage for OpenAI-compatible servers that omit chunk metadata.
//!
//! `id`, `object`, `created` and `model` describe a chunk but nothing in the SDK reads them;
//! only `choices` drives the stream. Servers in the wild leave some of them out:
//!
//! - NVIDIA's TensorRT-Edge-LLM experimental server sends `{"id", "choices"}` with no `object`,
//!   `created` or `model` (<https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/93>).
//! - Bifrost v1.6.9 sends `{"choices", "model", "object"}` with no `id` or `created`
//!   (<https://github.com/maximhq/bifrost/issues/7397>).
//!
//! Requiring those fields made every chunk from such a server a `Failed to parse SSE event
//! data: missing field` stream error, which is retryable, so the whole response was lost for
//! a field nobody reads. Each omission must now be tolerated.

mod common;

use common::{DONE, collect_events, options_for, sole_finish_reason, sse_server, text_of_events};
use futures::StreamExt;
use open_agent::{Client, ContentBlock, FinishReason, query};
use serde_json::json;

fn frame(chunk: serde_json::Value) -> String {
    format!("data: {chunk}\n\n")
}

fn choice(content: &str, finish: Option<&str>) -> serde_json::Value {
    json!({ "index": 0, "delta": { "content": content }, "finish_reason": finish })
}

#[tokio::test]
async fn chunks_with_only_an_id_and_choices_are_delivered() {
    // TensorRT-Edge-LLM's shape.
    let body = frame(json!({ "id": "r1", "choices": [choice("Hel", None)] }))
        + &frame(json!({ "id": "r1", "choices": [choice("lo", Some("stop"))] }))
        + DONE;

    let events = collect_events(body, false).await;

    assert_eq!(text_of_events(&events), "Hello");
    assert_eq!(sole_finish_reason(&events), FinishReason::Stop);
}

#[tokio::test]
async fn chunks_without_id_or_created_are_delivered() {
    // Bifrost's shape.
    let body = frame(json!({
        "model": "openai/gpt-4o-mini",
        "object": "chat.completion.chunk",
        "choices": [choice("Bonjour", None)],
    })) + &frame(json!({
        "model": "openai/gpt-4o-mini",
        "object": "chat.completion.chunk",
        "choices": [choice("", Some("stop"))],
    })) + DONE;

    let events = collect_events(body, false).await;

    assert_eq!(text_of_events(&events), "Bonjour");
    assert_eq!(sole_finish_reason(&events), FinishReason::Stop);
}

#[tokio::test]
async fn a_chunk_with_choices_alone_is_delivered_through_a_client() {
    let body = frame(json!({ "choices": [choice("ok", Some("stop"))] })) + DONE;
    let server = sse_server(body).await;
    let mut client = Client::new(options_for(&server)).expect("client");

    client.send("hi").await.expect("send");
    let mut text = String::new();
    while let Some(block) = client.receive().await.expect("receive") {
        if let ContentBlock::Text(block) = block {
            text.push_str(&block.text);
        }
    }

    assert_eq!(text, "ok");
    assert_eq!(client.finish_reason(), Some(&FinishReason::Stop));
}

#[tokio::test]
async fn an_error_payload_that_carries_metadata_but_no_choices_is_still_an_api_error() {
    // Relaxing the metadata fields must not turn a server-reported failure into an empty
    // chunk: `choices` is what makes a payload a chunk at all.
    let body = frame(json!({
        "id": "r1",
        "object": "error",
        "model": "m",
        "error": { "code": 429, "message": "slow down", "type": "rate_limit" },
    })) + DONE;
    let server = sse_server(body).await;
    let options = options_for(&server);

    let mut stream = query("hi", &options).await.expect("start query");
    let error = stream
        .next()
        .await
        .expect("an item")
        .expect_err("the payload is an error");

    assert_eq!(error.status_code(), Some(429));
    assert_eq!(error.to_string(), "API error 429: rate_limit: slow down");
}
