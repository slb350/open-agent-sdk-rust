//! Shared helpers and a lightweight loopback HTTP mock server for integration tests.
//!
//! Declared with `mod common;` by `tests/integration.rs` (and `tests/debug_logging_test.rs`).
//! This module contains no `#[test]` functions.

#![allow(dead_code)] // debug_logging_test uses only a subset of helpers.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use open_agent::{AgentOptions, ContentBlock, FinishReason, Message, StreamEvent, query};
use serde::de::DeserializeOwned;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Reads one complete HTTP request, including a body identified by `Content-Length`.
pub async fn read_request(socket: &mut TcpStream) -> Vec<u8> {
    try_read_request(socket)
        .await
        .expect("SDK closed the request before sending its body")
}

async fn try_read_request(socket: &mut TcpStream) -> Option<Vec<u8>> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4_096];

    loop {
        let read = socket.read(&mut buffer).await.ok()?;
        if read == 0 {
            return None;
        }
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
            request.truncate(body_start + content_length);
            return Some(request);
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

/// One HTTP request recorded by [`MockServer`].
#[derive(Debug, Clone)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl RecordedRequest {
    /// First header value matching `name` ASCII-case-insensitively.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Parses the recorded request body as JSON.
    pub fn body_json<T: DeserializeOwned>(&self) -> serde_json::Result<T> {
        serde_json::from_slice(&self.body)
    }
}

/// HTTP response returned by a [`MockServer`] handler.
#[derive(Debug, Clone)]
pub struct MockResponse {
    status: u16,
    content_type: Option<&'static str>,
    body: String,
}

impl MockResponse {
    pub fn new(status: u16) -> Self {
        Self {
            status,
            content_type: None,
            body: String::new(),
        }
    }

    pub fn sse(body: impl Into<String>) -> Self {
        Self {
            status: 200,
            content_type: Some("text/event-stream"),
            body: body.into(),
        }
    }

    pub fn body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    fn format_http(&self) -> String {
        let reason = match self.status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            _ => "Status",
        };
        let mut head = format!("HTTP/1.1 {} {reason}\r\n", self.status);
        if let Some(content_type) = self.content_type {
            head.push_str("content-type: ");
            head.push_str(content_type);
            head.push_str("\r\n");
        }
        head.push_str(&format!(
            "content-length: {}\r\nconnection: close\r\n\r\n{}",
            self.body.len(),
            self.body
        ));
        head
    }
}

/// Lightweight loopback HTTP/1.1 mock server backed directly by Tokio's `TcpListener`.
///
/// Matches `POST` requests to one exact path (answering `404` for anything else) and records
/// every matched request before sending its response. Dropping the server aborts the listener.
pub struct MockServer {
    uri: String,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    handle: tokio::task::JoinHandle<()>,
}

impl MockServer {
    /// Starts a server answering `POST expected_path` via `handler`.
    pub async fn start_handler<F>(expected_path: impl Into<String>, handler: F) -> Self
    where
        F: Fn(&RecordedRequest) -> MockResponse + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback mock server");
        let addr = listener.local_addr().expect("mock server address");
        let expected_path = Arc::new(expected_path.into());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let handler = Arc::new(handler);

        let handle = tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let expected_path = Arc::clone(&expected_path);
                let recorded = Arc::clone(&recorded);
                let handler = Arc::clone(&handler);
                tokio::spawn(async move {
                    let Some(raw) = try_read_request(&mut socket).await else {
                        return;
                    };
                    let Some(request) = parse_http_request(&raw) else {
                        return;
                    };
                    let response = if request.method == "POST" && request.path == *expected_path {
                        let response = handler(&request);
                        recorded.lock().expect("record request").push(request);
                        response
                    } else {
                        MockResponse::new(404)
                    };
                    let wire = response.format_http();
                    let _ = socket.write_all(wire.as_bytes()).await;
                    let _ = socket.shutdown().await;
                });
            }
        });

        Self {
            uri: format!("http://{addr}"),
            requests,
            handle,
        }
    }

    /// Starts a server answering `POST expected_path` with a fixed `response`.
    pub async fn start_static(expected_path: impl Into<String>, response: MockResponse) -> Self {
        Self::start_handler(expected_path, move |_| response.clone()).await
    }

    /// Base URL without trailing slash (e.g. `http://127.0.0.1:12345`).
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Snapshot of all matched requests received so far, in arrival order.
    pub fn received_requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().expect("read requests").clone()
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

fn parse_http_request(raw: &[u8]) -> Option<RecordedRequest> {
    let header_end = raw.windows(4).position(|bytes| bytes == b"\r\n\r\n")?;
    let head = std::str::from_utf8(&raw[..header_end]).ok()?;
    let mut lines = head.lines();
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let headers = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    let body = raw[header_end + 4..].to_vec();
    Some(RecordedRequest {
        method,
        path,
        headers,
        body,
    })
}

/// Starts a mock server that answers the chat-completions endpoint with `body` as an SSE
/// stream.
pub async fn sse_server(body: impl Into<String>) -> MockServer {
    MockServer::start_static("/v1/chat/completions", MockResponse::sse(body)).await
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
/// path would 404 rather than quietly succeeding.
pub async fn anthropic_sse_server(body: impl Into<String>) -> MockServer {
    MockServer::start_static("/v1/messages", MockResponse::sse(body)).await
}

/// Formats one Anthropic SSE frame, with both the `event:` label and the `data:` payload.
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
