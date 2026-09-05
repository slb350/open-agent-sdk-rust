//! Capture actual request bodies instead of inspecting messages inserted into history.

mod common;

use common::{DONE, options_for, sse_server};
use open_agent::{Client, ContentBlock, ImageBlock, ImageDetail, Message, MessageRole, TextBlock};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn text_roles_and_empty_or_multiblock_content_reach_the_wire() {
    let server = sse_server(DONE).await;
    let cases = [
        (
            Message::user("Hello"),
            json!({"role": "user", "content": "Hello"}),
        ),
        (
            Message::system("Instructions"),
            json!({"role": "system", "content": "Instructions"}),
        ),
        (
            Message::assistant(vec![ContentBlock::Text(TextBlock::new("Reply"))]),
            json!({"role": "assistant", "content": "Reply"}),
        ),
        (
            Message::new(
                MessageRole::User,
                vec![
                    ContentBlock::Text(TextBlock::new("Line 1")),
                    ContentBlock::Text(TextBlock::new("Line 2")),
                ],
            ),
            json!({"role": "user", "content": "Line 1\nLine 2"}),
        ),
        (
            Message::new(MessageRole::User, vec![]),
            json!({"role": "user", "content": ""}),
        ),
    ];
    for (message, expected) in cases {
        let mut client = Client::new(options_for(&server)).unwrap();
        client.send_message(message).await.unwrap();
        let requests = server.received_requests().await.unwrap();
        let body: Value = requests.last().unwrap().body_json().unwrap();
        assert_eq!(body["messages"], json!([expected]));
        assert!(body.get("tools").is_none(), "no tools were registered");
    }
}

#[tokio::test]
async fn image_content_preserves_order_data_detail_and_empty_unicode_text_on_the_wire() {
    let server = sse_server(DONE).await;
    let url = "https://example.com/photo.jpg";
    let data = "data:image/png;base64,AAAA";
    let message = Message::new(
        MessageRole::User,
        vec![
            ContentBlock::Text(TextBlock::new("こんにちは 🌸")),
            ContentBlock::Image(
                ImageBlock::from_url(url)
                    .unwrap()
                    .with_detail(ImageDetail::High),
            ),
            ContentBlock::Text(TextBlock::new("")),
            ContentBlock::Image(ImageBlock::from_base64("AAAA", "image/png").unwrap()),
            ContentBlock::Text(TextBlock::new("   ")),
            ContentBlock::Image(
                ImageBlock::from_url(url)
                    .unwrap()
                    .with_detail(ImageDetail::Low),
            ),
            ContentBlock::Text(TextBlock::new("مرحبا 🎨")),
        ],
    );
    let original = serde_json::to_value(&message).unwrap();
    let mut client = Client::new(options_for(&server)).unwrap();
    client.send_message(message).await.unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: Value = requests[0].body_json().unwrap();
    assert_eq!(
        body["messages"],
        json!([{"role": "user", "content": [
            {"type": "text", "text": "こんにちは 🌸"},
            {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
            {"type": "text", "text": ""},
            {"type": "image_url", "image_url": {"url": data, "detail": "auto"}},
            {"type": "text", "text": "   "},
            {"type": "image_url", "image_url": {"url": url, "detail": "low"}},
            {"type": "text", "text": "مرحبا 🎨"}
        ]}])
    );
    assert_eq!(
        serde_json::to_value(&client.history()[0]).unwrap(),
        original
    );

    let mut image_only = Client::new(options_for(&server)).unwrap();
    image_only
        .send_message(Message::new(
            MessageRole::User,
            vec![
                ContentBlock::Image(ImageBlock::from_url(url).unwrap()),
                ContentBlock::Image(ImageBlock::from_base64("AAAA", "image/png").unwrap()),
            ],
        ))
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: Value = requests[1].body_json().unwrap();
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
            {"type": "image_url", "image_url": {"url": data, "detail": "auto"}}
        ])
    );
}

#[tokio::test]
async fn failed_request_keeps_the_original_message_in_history() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(503))
        .expect(1)
        .mount(&server)
        .await;
    let mut client = Client::new(options_for(&server)).unwrap();
    let message = Message::user_with_base64_image("inspect", "AAAA", "image/png").unwrap();
    let original = serde_json::to_value(&message).unwrap();
    let error = client.send_message(message).await.unwrap_err();
    assert_eq!(error.status_code(), Some(503));
    assert_eq!(
        serde_json::to_value(client.history()).unwrap(),
        json!([original])
    );
}
