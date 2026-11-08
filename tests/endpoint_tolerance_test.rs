//! Endpoint-shape tolerance: a trailing slash on `base_url`, and SSE events with no payload.
//!
//! `base_url` is documented without a trailing slash, but `http://localhost:11434/v1/` is how
//! many users copy an endpoint, and the SDK appends the protocol path itself. The result was
//! `/v1//chat/completions`, which strict servers answer with 404. Separately, an SSE frame whose
//! `data` field is empty is valid framing that carries no JSON; the decoders treated it as a
//! malformed payload and failed a stream that was otherwise complete.

mod common;

use common::{
    DONE, anthropic_frame, anthropic_text_response, sole_finish_reason, text_chunk, text_of_events,
};
use futures::StreamExt;
use open_agent::{
    AgentOptions, ApiProtocol, Client, ContentBlock, FinishReason, StreamEvent, query,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn collect(options: &AgentOptions) -> Vec<StreamEvent> {
    let mut stream = query("hi", options).await.expect("start query");
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("stream yields no errors"));
    }
    events
}

/// Serves `body` on exactly `endpoint`; any other path is a 404.
async fn server_at(endpoint: &str, body: String) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(endpoint))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;
    server
}

fn options(server: &MockServer, suffix: &str, protocol: ApiProtocol) -> AgentOptions {
    AgentOptions::builder()
        .model("m")
        .base_url(format!("{}/v1{suffix}", server.uri()))
        .api_key("k")
        .protocol(protocol)
        .build()
        .expect("options build")
}

#[tokio::test]
async fn a_trailing_slash_on_base_url_does_not_double_the_path_separator() {
    for suffix in ["/", "//"] {
        let server = server_at(
            "/v1/chat/completions",
            text_chunk("ok", Some("stop")) + DONE,
        )
        .await;
        let events = collect(&options(&server, suffix, ApiProtocol::OpenAiChat)).await;
        assert_eq!(text_of_events(&events), "ok", "suffix {suffix:?}");
    }
}

#[tokio::test]
async fn a_trailing_slash_is_tolerated_on_the_anthropic_path_too() {
    let server = server_at("/v1/messages", anthropic_text_response("ok", "end_turn")).await;
    let events = collect(&options(&server, "/", ApiProtocol::Anthropic)).await;
    assert_eq!(text_of_events(&events), "ok");
}

#[tokio::test]
async fn a_trailing_slash_is_tolerated_by_client_requests() {
    let server = server_at(
        "/v1/chat/completions",
        text_chunk("ok", Some("stop")) + DONE,
    )
    .await;
    let mut client =
        Client::new(options(&server, "/", ApiProtocol::OpenAiChat)).expect("client builds");
    client.send("hi").await.expect("send reaches the endpoint");
    let mut text = String::new();
    while let Some(block) = client.receive().await.expect("receive") {
        if let ContentBlock::Text(block) = block {
            text.push_str(&block.text);
        }
    }
    assert_eq!(text, "ok");
}

#[tokio::test]
async fn the_configured_base_url_is_reported_unchanged() {
    let server = MockServer::start().await;
    let configured = format!("{}/v1/", server.uri());
    let options = AgentOptions::builder()
        .model("m")
        .base_url(configured.clone())
        .build()
        .expect("options build");
    assert_eq!(options.base_url(), configured);
}

#[tokio::test]
async fn openai_events_with_no_payload_are_skipped() {
    // An empty `data:` field, a whitespace-only one and a comment keep-alive, around real chunks.
    let body = format!(
        "data:\n\n{}data: \n\ndata:   \n\n: keep-alive\n\n{}{DONE}",
        text_chunk("a", None),
        text_chunk("b", Some("stop")),
    );
    let server = server_at("/v1/chat/completions", body).await;
    let events = collect(&options(&server, "", ApiProtocol::OpenAiChat)).await;
    assert_eq!(text_of_events(&events), "ab");
    assert_eq!(sole_finish_reason(&events), FinishReason::Stop);
}

#[tokio::test]
async fn anthropic_events_with_no_payload_are_skipped() {
    let ping = anthropic_frame("ping", serde_json::json!({ "type": "ping" }));
    let body = format!(
        "data:\n\n{}data: \n\n{ping}",
        anthropic_text_response("ok", "end_turn")
    );
    let server = server_at("/v1/messages", body).await;
    let events = collect(&options(&server, "", ApiProtocol::Anthropic)).await;
    assert_eq!(text_of_events(&events), "ok");
    assert_eq!(sole_finish_reason(&events), FinishReason::Stop);
}

#[tokio::test]
async fn a_malformed_payload_is_still_a_stream_error() {
    // Only a payload with nothing in it is skipped; text that is not JSON must keep failing.
    let body = format!("data: not json\n\n{}{DONE}", text_chunk("a", Some("stop")));
    let server = server_at("/v1/chat/completions", body).await;
    let mut stream = query("hi", &options(&server, "", ApiProtocol::OpenAiChat))
        .await
        .expect("start query");
    let mut saw_error = false;
    while let Some(event) = stream.next().await {
        saw_error |= event.is_err();
    }
    assert!(saw_error, "a non-JSON payload must surface as an error");
}
