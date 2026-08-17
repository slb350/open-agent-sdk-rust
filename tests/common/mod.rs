//! Shared helpers for integration tests.
//!
//! Declared with `mod common;` by each consumer. Cargo compiles every file directly under
//! `tests/` as its own test binary, but subdirectories are not, so helpers live here rather
//! than in a top-level file. This module must contain no `#[test]` functions.

#![allow(dead_code)] // Each test binary uses only the helpers it needs.

use open_agent::{AgentOptions, ContentBlock, Message};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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
    AgentOptions::builder()
        .model("m")
        .base_url(format!("{}/v1", server.uri()))
        .api_key("k")
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
fn sse_frame(choice: serde_json::Value) -> String {
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
