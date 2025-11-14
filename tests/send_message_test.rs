//! Tests for Client::send_message() API
//!
//! Verifies that pre-built Message objects (especially with images) can be sent
//! through the public API without requiring manual history manipulation.

use open_agent::{AgentOptions, Client, ImageBlock, ImageDetail, Message};

#[tokio::test]
async fn test_send_message_with_image() {
    // GIVEN: Client configured for local server
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .expect("Valid options");

    let mut client = Client::new(options).expect("Valid client");

    // WHEN: User creates message with image helper and sends it
    let msg = Message::user_with_image("What's in this image?", "https://example.com/photo.jpg")
        .expect("Valid image URL");

    // This should work without manual history manipulation
    // Note: Will fail with connection error since server doesn't exist,
    // but that's fine - we just need to verify the API exists
    let result = client.send_message(msg).await;

    // THEN: The API should exist (even if request fails)
    // The error will be a connection error, not a compilation error
    assert!(result.is_err()); // Expected to fail - no server running

    // AND: Message should be in history
    assert_eq!(client.history().len(), 1);

    // Verify it's the image message, not a text-only message
    let stored_msg = &client.history()[0];
    assert_eq!(stored_msg.content.len(), 2); // Text + Image
}

#[tokio::test]
async fn test_send_message_with_base64_image() {
    // GIVEN: Client configured for local server
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .expect("Valid options");

    let mut client = Client::new(options).expect("Valid client");

    // WHEN: User creates message with base64 image helper
    let base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    let msg = Message::user_with_base64_image("Analyze this image", base64_data, "image/png")
        .expect("Valid base64");

    let result = client.send_message(msg).await;

    // THEN: Message should be in history with data URI
    assert!(result.is_err()); // Expected - no server
    assert_eq!(client.history().len(), 1);

    let stored_msg = &client.history()[0];
    assert_eq!(stored_msg.content.len(), 2); // Text + Image
}

#[tokio::test]
async fn test_send_message_with_image_detail() {
    // GIVEN: Client configured for local server
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .expect("Valid options");

    let mut client = Client::new(options).expect("Valid client");

    // WHEN: User creates message with specific image detail level
    let msg = Message::user_with_image_detail(
        "Analyze this diagram in detail",
        "https://example.com/diagram.png",
        ImageDetail::High,
    )
    .expect("Valid image URL");

    let result = client.send_message(msg).await;

    // THEN: Message should be in history with correct detail level
    assert!(result.is_err()); // Expected - no server
    assert_eq!(client.history().len(), 1);

    // Verify detail level is preserved
    use open_agent::ContentBlock;
    let stored_msg = &client.history()[0];
    match &stored_msg.content[1] {
        ContentBlock::Image(img) => {
            assert_eq!(img.detail(), ImageDetail::High);
        }
        _ => panic!("Expected Image block"),
    }
}

#[tokio::test]
async fn test_send_message_with_custom_blocks() {
    // GIVEN: Client configured for local server
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .expect("Valid options");

    let mut client = Client::new(options).expect("Valid client");

    // WHEN: User manually constructs complex message with multiple blocks
    use open_agent::{ContentBlock, MessageRole, TextBlock};

    let image1 = ImageBlock::from_url("https://example.com/img1.jpg").expect("Valid URL");
    let image2 = ImageBlock::from_url("https://example.com/img2.jpg").expect("Valid URL");

    let msg = Message::new(
        MessageRole::User,
        vec![
            ContentBlock::Text(TextBlock::new("Compare these images:")),
            ContentBlock::Image(image1),
            ContentBlock::Image(image2),
        ],
    );

    let result = client.send_message(msg).await;

    // THEN: Multi-image message should be in history
    assert!(result.is_err()); // Expected - no server
    assert_eq!(client.history().len(), 1);

    let stored_msg = &client.history()[0];
    assert_eq!(stored_msg.content.len(), 3); // Text + 2 Images
}
