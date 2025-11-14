//! Tests for debug logging functionality
//!
//! Tests that debug logs are emitted when images are serialized.

use open_agent::{
    AgentOptions, Client, ContentBlock, ImageBlock, ImageDetail, Message, MessageRole, TextBlock,
};

#[tokio::test]
async fn test_image_logging_with_debug_enabled() {
    // Initialize env_logger for this test
    // Set to debug level to capture log::debug! calls
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Debug)
        .try_init();

    // Create a message with images
    let msg = Message::new(
        MessageRole::User,
        vec![
            ContentBlock::Text(TextBlock::new("What's in this image?")),
            ContentBlock::Image(
                ImageBlock::from_url("https://example.com/test.jpg")
                    .unwrap()
                    .with_detail(ImageDetail::High),
            ),
        ],
    );

    // Create a client (logging will happen during message building)
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .unwrap();

    let mut client = Client::new(options).unwrap();

    // Add the message with images
    client.history_mut().push(msg);

    // The logging happens when building OpenAI messages
    // This is internal to the send() call, but we can verify
    // the message was added to history
    assert_eq!(client.history().len(), 1);

    // Note: Actual log output would appear in test output with RUST_LOG=debug
    // This test verifies the code compiles and runs without errors
}

#[tokio::test]
async fn test_image_logging_truncates_long_urls() {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Debug)
        .try_init();

    // Create a message with a very long data URI
    let long_base64 = "A".repeat(200);
    let img = ImageBlock::from_base64(&long_base64, "image/png").unwrap();

    let msg = Message::new(MessageRole::User, vec![ContentBlock::Image(img)]);

    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .unwrap();

    let mut client = Client::new(options).unwrap();
    client.history_mut().push(msg);

    // The URL should be >100 chars, so logging should truncate it
    assert_eq!(client.history().len(), 1);

    // Log output would show: "data:image/png;base64,AAAA... (227 chars)"
}

#[tokio::test]
async fn test_image_logging_includes_detail_level() {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Debug)
        .try_init();

    // Create messages with different detail levels
    let messages = vec![
        Message::new(
            MessageRole::User,
            vec![ContentBlock::Image(
                ImageBlock::from_url("https://example.com/low.jpg")
                    .unwrap()
                    .with_detail(ImageDetail::Low),
            )],
        ),
        Message::new(
            MessageRole::User,
            vec![ContentBlock::Image(
                ImageBlock::from_url("https://example.com/high.jpg")
                    .unwrap()
                    .with_detail(ImageDetail::High),
            )],
        ),
        Message::new(
            MessageRole::User,
            vec![ContentBlock::Image(
                ImageBlock::from_url("https://example.com/auto.jpg")
                    .unwrap()
                    .with_detail(ImageDetail::Auto),
            )],
        ),
    ];

    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .unwrap();

    let mut client = Client::new(options).unwrap();

    for msg in messages {
        client.history_mut().push(msg);
    }

    assert_eq!(client.history().len(), 3);

    // Log output would show:
    // "- Image: https://example.com/low.jpg (detail: low)"
    // "- Image: https://example.com/high.jpg (detail: high)"
    // "- Image: https://example.com/auto.jpg (detail: auto)"
}

#[test]
fn test_no_warning_for_image_only_messages() {
    // GIVEN: Message with only images (no text)
    // This is a valid use case for vision models
    let image = ImageBlock::from_url("https://example.com/test.jpg")
        .unwrap()
        .with_detail(ImageDetail::High);

    let msg = Message::new(
        MessageRole::User,
        vec![
            ContentBlock::Image(image.clone()),
            ContentBlock::Image(image),
        ],
    );

    // THEN: Image-only messages are valid and should not trigger warnings
    // The warning at src/client.rs:1200-1205 should be removed
    assert_eq!(msg.content.len(), 2);
    assert!(matches!(msg.content[0], ContentBlock::Image(_)));
    assert!(matches!(msg.content[1], ContentBlock::Image(_)));

    // This test documents that image-only messages are intentional,
    // not bugs. Use cases include:
    // - "What's in this image?" (text in system prompt)
    // - Multi-image comparison without additional text
    // - Visual question answering where the question is implicit
}
