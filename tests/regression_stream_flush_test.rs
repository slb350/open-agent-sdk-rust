//! Regression coverage for stream termination without an explicit `finish_reason`.
//!
//! Several OpenAI-compatible servers (llama.cpp, vLLM, and some local gateways) stream content
//! deltas and then send `data: [DONE]` without ever setting `finish_reason`. The aggregator
//! must flush whatever it accumulated when the transport ends, otherwise the caller sees an
//! empty successful response and cannot distinguish it from a model that genuinely returned
//! nothing.

mod common;

use common::{DONE, options_for, sse_server, text_chunk, text_of, tool_chunk};
use futures::StreamExt;
use open_agent::{ContentBlock, query};

/// Serves `body` as an SSE response and collects every block `query()` yields.
async fn collect_blocks(body: String) -> Vec<ContentBlock> {
    let server = sse_server(body).await;
    let options = options_for(&server);

    let mut stream = query("hi", &options).await.expect("start query");
    let mut blocks = Vec::new();
    while let Some(block) = stream.next().await {
        blocks.push(block.expect("stream yields no errors"));
    }
    blocks
}

#[tokio::test]
async fn text_is_not_lost_when_stream_ends_without_finish_reason() {
    let body = text_chunk("IMPORTANT PAYLOAD", None) + DONE;
    assert_eq!(text_of(&collect_blocks(body).await), "IMPORTANT PAYLOAD");
}

#[tokio::test]
async fn multi_chunk_text_is_flushed_when_the_transport_simply_ends() {
    // No [DONE] sentinel at all: the connection just closes.
    let body = text_chunk("Hello", None) + &text_chunk(" world", None);
    assert_eq!(text_of(&collect_blocks(body).await), "Hello world");
}

#[tokio::test]
async fn tool_calls_are_flushed_when_stream_ends_without_finish_reason() {
    let body = tool_chunk("call_1", "search", "{\"q\":\"rust\"}") + DONE;
    let blocks = collect_blocks(body).await;

    assert_eq!(blocks.len(), 1, "expected one tool use block: {blocks:?}");
    match &blocks[0] {
        ContentBlock::ToolUse(tool_use) => {
            assert_eq!(tool_use.id(), "call_1");
            assert_eq!(tool_use.name(), "search");
            assert_eq!(tool_use.input(), &serde_json::json!({"q": "rust"}));
        }
        other => panic!("expected ToolUse, got {other:?}"),
    }
}

#[tokio::test]
async fn finish_reason_stream_still_emits_exactly_one_text_block() {
    // The flush must not double-emit content that finish_reason already flushed.
    let body = text_chunk("Hello", None) + &text_chunk(" world", Some("stop")) + DONE;
    let blocks = collect_blocks(body).await;

    assert_eq!(blocks.len(), 1, "expected exactly one block: {blocks:?}");
    assert_eq!(text_of(&blocks), "Hello world");
}

#[tokio::test]
async fn a_genuinely_empty_stream_still_yields_no_blocks() {
    let blocks = collect_blocks(DONE.to_string()).await;
    assert!(blocks.is_empty(), "expected no blocks, got {blocks:?}");
}
