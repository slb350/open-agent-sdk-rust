//! Shared helpers for integration tests.
//!
//! Declared with `mod common;` by each consumer. Cargo compiles every file directly under
//! `tests/` as its own test binary, but subdirectories are not, so helpers live here rather
//! than in a top-level file. This module must contain no `#[test]` functions.

#![allow(dead_code)] // Each test binary uses only the helpers it needs.

use futures::StreamExt;
use open_agent::{AgentOptions, ContentBlock, FinishReason, Message, StreamEvent, query};
use tokio::io::AsyncReadExt;
use tokio::net::TcpStream;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Reads one complete HTTP request, including a body identified by `Content-Length`.
pub async fn read_request(socket: &mut TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4_096];

    loop {
        let read = socket.read(&mut buffer).await.expect("read SDK request");
        assert!(read > 0, "SDK closed the request before sending its body");
        request.extend_from_slice(&buffer[..read]);

        let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") else {
            continue;
        };
        let body_start = header_end + 4;
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .filter_map(|line| line.split_once(':'))
            .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
            .map(|(_, value)| {
                value
                    .trim()
                    .parse::<usize>()
                    .expect("request content-length is numeric")
            })
            .unwrap_or_default();

        if request.len() >= body_start + content_length {
            return request;
        }
    }
}

/// Returns every value for `expected_name`, matching the HTTP header name case-insensitively.
pub fn header_values(request: &str, expected_name: &str) -> Vec<String> {
    request
        .lines()
        .filter_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.eq_ignore_ascii_case(expected_name))
        .map(|(_, value)| value.trim().to_string())
        .collect()
}

/// Starts a mock server that answers the chat-completions endpoint with `body` as an SSE
/// stream.
///
/// The returned server must be kept alive for the duration of the request; dropping it shuts
/// the listener down.
pub async fn sse_server(body: impl Into<String>) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body.into()),
        )
        .mount(&server)
        .await;
    server
}

/// Builds minimal `AgentOptions` pointing at `server`.
///
/// Only the required fields are set, so anything a test does not configure keeps its default
/// — which is what makes this usable for asserting on omitted request fields.
pub fn options_for(server: &MockServer) -> AgentOptions {
    options_with_reasoning(server, false)
}

/// Builds minimal `AgentOptions` pointing at `server`, selecting reasoning capture.
pub fn options_with_reasoning(server: &MockServer, include_reasoning: bool) -> AgentOptions {
    AgentOptions::builder()
        .model("m")
        .base_url(format!("{}/v1", server.uri()))
        .api_key("k")
        .include_reasoning(include_reasoning)
        .build()
        .expect("minimal options build")
}

/// Builds a single SSE `data:` frame carrying a text delta.
///
/// `finish` is the `finish_reason` to report, or `None` for a server that never sends one.
pub fn text_chunk(content: &str, finish: Option<&str>) -> String {
    sse_frame(serde_json::json!({
        "index": 0,
        "delta": { "content": content },
        "finish_reason": finish,
    }))
}

/// Builds a single SSE `data:` frame carrying a complete tool-call delta.
pub fn tool_chunk(id: &str, name: &str, arguments: &str) -> String {
    sse_frame(serde_json::json!({
        "index": 0,
        "delta": {
            "tool_calls": [{
                "index": 0,
                "id": id,
                "type": "function",
                "function": { "name": name, "arguments": arguments },
            }],
        },
        "finish_reason": serde_json::Value::Null,
    }))
}

/// Wraps one choice in a full chunk object and formats it as an SSE `data:` frame.
pub fn sse_frame(choice: serde_json::Value) -> String {
    let chunk = serde_json::json!({
        "id": "1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [choice],
    });
    format!("data: {chunk}\n\n")
}

/// The SSE terminator most OpenAI-compatible servers send.
pub const DONE: &str = "data: [DONE]\n\n";

/// Starts a mock server that answers the Anthropic messages endpoint with `body` as an SSE
/// stream.
///
/// Mounted on `/v1/messages` specifically, so a request that went to the chat-completions
/// path would 404 rather than quietly succeeding — which is what makes the path itself part
/// of what these tests assert.
pub async fn anthropic_sse_server(body: impl Into<String>) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body.into()),
        )
        .mount(&server)
        .await;
    server
}

/// Formats one Anthropic SSE frame, with both the `event:` label and the `data:` payload.
///
/// Real servers send both. Emitting only the payload here would let a parser that reads the
/// label pass a test it should fail.
pub fn anthropic_frame(event: &str, data: serde_json::Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

/// A complete minimal Anthropic response: one text block, stopping for `stop_reason`.
pub fn anthropic_text_response(text: &str, stop_reason: &str) -> String {
    [
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
                "delta": { "type": "text_delta", "text": text },
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
                "delta": { "stop_reason": stop_reason, "stop_sequence": null },
                "usage": { "output_tokens": 2 },
            }),
        ),
        anthropic_frame(
            "message_stop",
            serde_json::json!({ "type": "message_stop" }),
        ),
    ]
    .concat()
}

/// Concatenates the text of every `ContentBlock::Text` in `blocks`.
pub fn text_of(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// Concatenates the text of a message's text blocks, for identifying messages in assertions.
pub fn message_text(message: &Message) -> String {
    text_of(&message.content)
}

/// Builds a single SSE `data:` frame carrying a reasoning delta on `field`.
///
/// `field` is the channel name the server uses: DeepSeek streams `reasoning_content`,
/// OpenRouter streams `reasoning`.
pub fn reasoning_chunk(field: &str, value: &str, finish: Option<&str>) -> String {
    sse_frame(serde_json::json!({
        "index": 0,
        "delta": { field: value },
        "finish_reason": finish,
    }))
}

/// Builds a frame whose delta carries reasoning and content simultaneously.
pub fn mixed_chunk(reasoning_field: &str, reasoning: &str, content: &str) -> String {
    sse_frame(serde_json::json!({
        "index": 0,
        "delta": { reasoning_field: reasoning, "content": content },
        "finish_reason": serde_json::Value::Null,
    }))
}

/// Serves `body` as an SSE response and collects every event `query()` yields.
///
/// `include_reasoning` selects whether reasoning deltas are surfaced or dropped.
pub async fn collect_events(body: String, include_reasoning: bool) -> Vec<StreamEvent> {
    let server = sse_server(body).await;
    let options = options_with_reasoning(&server, include_reasoning);

    let mut stream = query("hi", &options).await.expect("start query");
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("stream yields no errors"));
    }
    events
}

/// Extracts the content blocks from a collected event sequence.
pub fn blocks_of(events: &[StreamEvent]) -> Vec<ContentBlock> {
    events
        .iter()
        .filter_map(|event| event.as_block().cloned())
        .collect()
}

/// Concatenates every reasoning event in a collected event sequence.
pub fn reasoning_of(events: &[StreamEvent]) -> String {
    events
        .iter()
        .filter_map(StreamEvent::as_reasoning)
        .collect()
}

/// Concatenates the text of every text block in a collected event sequence.
pub fn text_of_events(events: &[StreamEvent]) -> String {
    events.iter().filter_map(StreamEvent::as_text).collect()
}

/// Returns the finish reason of a collected event sequence.
///
/// Asserts the stream's two structural guarantees along the way: exactly one `Finish` event,
/// and it is the last one. Every event-collecting test gets that check for free.
pub fn sole_finish_reason(events: &[StreamEvent]) -> FinishReason {
    let reasons: Vec<&FinishReason> = events
        .iter()
        .filter_map(StreamEvent::finish_reason)
        .collect();
    assert_eq!(
        reasons.len(),
        1,
        "expected exactly one Finish event, got {events:?}"
    );
    assert!(
        events
            .last()
            .is_some_and(|event| event.finish_reason().is_some()),
        "Finish must be the last event, got {events:?}"
    );
    reasons[0].clone()
}
