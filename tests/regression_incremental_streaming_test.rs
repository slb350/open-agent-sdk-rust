//! Regression coverage for incremental text delivery.
//!
//! Before 0.10.0 the accumulators concatenated every text fragment and emitted one
//! `ContentBlock::Text` at the end of the stream, so a caller printing blocks as they arrived
//! saw nothing until generation finished — while the SDK advertised token-by-token streaming.
//! Text and reasoning now reach the caller as they arrive; tool calls still cannot, because
//! their arguments are only valid JSON once the last fragment lands.

mod common;

use common::{
    DONE, anthropic_frame, anthropic_sse_server, blocks_of, collect_events, mixed_chunk,
    options_for, reasoning_chunk, sse_server, text_chunk, text_of_events, tool_chunk,
};
use futures::StreamExt;
use open_agent::{Client, ContentBlock, StreamEvent, query};

/// The text of every `StreamEvent::Block(Text)`, one entry per event.
fn text_blocks(events: &[StreamEvent]) -> Vec<String> {
    blocks_of(events)
        .into_iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.text),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn each_text_delta_arrives_as_its_own_block() {
    let body = text_chunk("Hello", None)
        + &text_chunk(", ", None)
        + &text_chunk("world", Some("stop"))
        + DONE;

    let events = collect_events(body, false).await;

    assert_eq!(text_blocks(&events), vec!["Hello", ", ", "world"]);
}

#[tokio::test]
async fn an_empty_delta_emits_no_block() {
    // The first chunk of an OpenAI stream routinely carries an empty content string alongside
    // the role. Emitting a block for it would hand callers a stream of empty strings.
    let body = text_chunk("", None) + &text_chunk("real", Some("stop")) + DONE;

    assert_eq!(
        text_blocks(&collect_events(body, false).await),
        vec!["real"]
    );
}

#[tokio::test]
async fn text_reaches_the_caller_before_the_stream_reports_a_finish_reason() {
    // The point of the change: a block must be observable while the stream is still open, not
    // only once the server says why it stopped.
    let body = text_chunk("first", None) + &text_chunk("second", Some("stop")) + DONE;
    let server = sse_server(body).await;
    let options = options_for(&server);

    let mut stream = query("hi", &options).await.expect("start query");
    let first = stream
        .next()
        .await
        .expect("a first event")
        .expect("no error");

    match first {
        StreamEvent::Block(ContentBlock::Text(text)) => assert_eq!(text.text, "first"),
        other => panic!("expected the first text delta as the first event, got {other:?}"),
    }
}

#[tokio::test]
async fn anthropic_text_deltas_arrive_one_block_each() {
    let body = [
        anthropic_frame(
            "message_start",
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": "msg_1", "type": "message", "role": "assistant", "content": [],
                    "model": "m", "stop_reason": null, "stop_sequence": null,
                    "usage": { "input_tokens": 1, "output_tokens": 1 },
                },
            }),
        ),
        anthropic_frame(
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start", "index": 0,
                "content_block": { "type": "text", "text": "" },
            }),
        ),
        anthropic_frame(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta", "index": 0,
                "delta": { "type": "text_delta", "text": "Hel" },
            }),
        ),
        anthropic_frame(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta", "index": 0,
                "delta": { "type": "text_delta", "text": "lo" },
            }),
        ),
        anthropic_frame(
            "content_block_stop",
            serde_json::json!({ "type": "content_block_stop", "index": 0 }),
        ),
        anthropic_frame(
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn", "stop_sequence": null },
                "usage": { "output_tokens": 2 },
            }),
        ),
        anthropic_frame(
            "message_stop",
            serde_json::json!({ "type": "message_stop" }),
        ),
    ]
    .concat();

    let server = anthropic_sse_server(body).await;
    let options = open_agent::AgentOptions::builder()
        .model("m")
        .base_url(format!("{}/v1", server.uri()))
        .api_key("k")
        .protocol(open_agent::ApiProtocol::Anthropic)
        .build()
        .expect("options build");

    let mut stream = query("hi", &options).await.expect("start query");
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("no stream error"));
    }

    assert_eq!(text_blocks(&events), vec!["Hel", "lo"]);
}

#[tokio::test]
async fn reasoning_still_precedes_the_text_it_produced() {
    // Reasoning used to be emitted at flush, ahead of the single text block. Emitting text as
    // it arrives while holding reasoning to the end would have silently reversed that order.
    let body = reasoning_chunk("reasoning_content", "thinking", None)
        + &text_chunk("answer", Some("stop"))
        + DONE;

    let events = collect_events(body, true).await;
    let shape: Vec<&str> = events
        .iter()
        .map(|event| match event {
            StreamEvent::Reasoning(_) => "reasoning",
            StreamEvent::Block(_) => "block",
            StreamEvent::Finish(_) => "finish",
            _ => "other",
        })
        .collect();

    assert_eq!(shape, vec!["reasoning", "block", "finish"]);
}

#[tokio::test]
async fn reasoning_and_content_in_one_delta_keep_their_channels() {
    let body = mixed_chunk("reasoning", "why", "what") + &text_chunk("", Some("stop")) + DONE;

    let events = collect_events(body, true).await;

    assert_eq!(text_of_events(&events), "what");
    assert_eq!(
        events
            .iter()
            .filter_map(StreamEvent::as_reasoning)
            .collect::<String>(),
        "why"
    );
}

#[tokio::test]
async fn a_tool_call_still_emits_once_its_arguments_are_complete() {
    // Tool arguments assemble from fragments that are not valid JSON in isolation, so they
    // cannot stream. Text emitted before the call must still precede it.
    let body = text_chunk("calling", None)
        + &tool_chunk("call_1", "add", r#"{"a":1}"#)
        + &text_chunk("", Some("tool_calls"))
        + DONE;

    let events = collect_events(body, false).await;
    let kinds: Vec<&str> = blocks_of(&events)
        .iter()
        .map(|block| match block {
            ContentBlock::Text(_) => "text",
            ContentBlock::ToolUse(_) => "tool_use",
            _ => "other",
        })
        .collect();

    assert_eq!(kinds, vec!["text", "tool_use"]);
}

#[tokio::test]
async fn history_records_one_assistant_message_with_the_text_joined() {
    // Callers see the deltas; history must not. A message per fragment would multiply the
    // assistant turns replayed to the model on the next request.
    let body = text_chunk("Hel", None) + &text_chunk("lo", Some("stop")) + DONE;
    let server = sse_server(body).await;
    let mut client = Client::new(options_for(&server)).expect("client");

    client.send("hi").await.expect("send");
    while client.receive().await.expect("receive").is_some() {}

    let assistant = client
        .history()
        .iter()
        .filter(|message| message.role == open_agent::MessageRole::Assistant)
        .collect::<Vec<_>>();

    assert_eq!(assistant.len(), 1, "one assistant turn, got {assistant:?}");
    assert_eq!(
        assistant[0].content.len(),
        1,
        "text fragments must be joined in history, got {:?}",
        assistant[0].content
    );
    match &assistant[0].content[0] {
        ContentBlock::Text(text) => assert_eq!(text.text, "Hello"),
        other => panic!("expected a single joined text block, got {other:?}"),
    }
}
