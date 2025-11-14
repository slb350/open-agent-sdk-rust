//! Integration tests for Client image serialization
//!
//! These tests verify that the Client properly handles ImageBlock instances
//! throughout the message lifecycle, ensuring validation is preserved and
//! images are correctly stored in history.

use open_agent::{
    AgentOptions, Client, ContentBlock, ImageBlock, ImageDetail, Message, MessageRole,
};

#[test]
fn test_client_preserves_http_image_url() {
    // GIVEN: Client with HTTP image URL message
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .expect("Valid options");

    let mut client = Client::new(options).expect("Valid client");

    // Create a validated ImageBlock with HTTP URL
    let image_url = "https://example.com/test.jpg";
    let image = ImageBlock::from_url(image_url).expect("Valid HTTPS URL");
    let msg = Message::new(
        MessageRole::User,
        vec![
            ContentBlock::Text(open_agent::TextBlock::new("Here's an image:")),
            ContentBlock::Image(image),
        ],
    );
    client.history_mut().push(msg);

    // WHEN: We retrieve the message from history
    let stored_msg = &client.history()[0];

    // THEN: The image URL should be preserved
    assert_eq!(stored_msg.content.len(), 2, "Should have 2 content blocks");

    match &stored_msg.content[1] {
        ContentBlock::Image(img) => {
            assert_eq!(img.url(), image_url, "Image URL should be preserved");
            assert_eq!(
                img.detail(),
                ImageDetail::Auto,
                "Default detail should be Auto"
            );
        }
        _ => panic!("Expected Image content block"),
    }
}

#[test]
fn test_client_preserves_base64_data_uri() {
    // GIVEN: Client with base64 image message
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .expect("Valid options");

    let mut client = Client::new(options).expect("Valid client");

    // Create a validated ImageBlock from base64
    let base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    let image = ImageBlock::from_base64(base64_data, "image/png").expect("Valid base64");
    let msg = Message::new(MessageRole::User, vec![ContentBlock::Image(image)]);
    client.history_mut().push(msg);

    // WHEN: We retrieve the message from history
    let stored_msg = &client.history()[0];

    // THEN: The image should preserve the data URI format
    assert_eq!(stored_msg.content.len(), 1, "Should have 1 content block");

    match &stored_msg.content[0] {
        ContentBlock::Image(img) => {
            let url = img.url();
            assert!(
                url.starts_with("data:image/png;base64,"),
                "Base64 image should use data URI format, got: {}",
                url
            );
            // Verify the base64 data is preserved in the URL
            assert!(
                url.contains(base64_data),
                "Should contain original base64 data"
            );
        }
        _ => panic!("Expected Image content block"),
    }
}

#[test]
fn test_client_preserves_image_detail_level() {
    // GIVEN: Client with image having specific detail level
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .expect("Valid options");

    let mut client = Client::new(options).expect("Valid client");

    // Test all three detail levels
    let test_cases = vec![
        (ImageDetail::Low, "Low detail"),
        (ImageDetail::High, "High detail"),
        (ImageDetail::Auto, "Auto detail"),
    ];

    for (detail, _description) in &test_cases {
        let image = ImageBlock::from_url("https://example.com/image.jpg")
            .expect("Valid URL")
            .with_detail(*detail);

        let msg = Message::new(MessageRole::User, vec![ContentBlock::Image(image)]);
        client.history_mut().push(msg);
    }

    // WHEN: We retrieve the messages from history
    // THEN: Each detail level should be preserved
    for (i, (expected_detail, description)) in test_cases.iter().enumerate() {
        let stored_msg = &client.history()[i];
        match &stored_msg.content[0] {
            ContentBlock::Image(img) => {
                assert_eq!(
                    img.detail(),
                    *expected_detail,
                    "{} should be preserved",
                    description
                );
            }
            _ => panic!("Expected Image content block"),
        }
    }
}

#[test]
fn test_client_preserves_validation_in_conversation() {
    // GIVEN: Client with multi-turn conversation including images
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .expect("Valid options");

    let mut client = Client::new(options).expect("Valid client");

    // Add user message with validated image
    let image_url = "https://trusted-cdn.example.com/image.png";
    let image = ImageBlock::from_url(image_url)
        .expect("Valid URL")
        .with_detail(ImageDetail::High);
    client.history_mut().push(Message::new(
        MessageRole::User,
        vec![
            ContentBlock::Text(open_agent::TextBlock::new("What's in this image?")),
            ContentBlock::Image(image),
        ],
    ));

    // Add assistant response (text only)
    client.history_mut().push(Message::new(
        MessageRole::Assistant,
        vec![ContentBlock::Text(open_agent::TextBlock::new(
            "I see the image",
        ))],
    ));

    // Add another user message (text only)
    client.history_mut().push(Message::new(
        MessageRole::User,
        vec![ContentBlock::Text(open_agent::TextBlock::new(
            "Can you describe it?",
        ))],
    ));

    // WHEN: We retrieve all messages from history
    // THEN: All messages should be preserved correctly
    assert_eq!(
        client.history().len(),
        3,
        "Should have 3 messages in history"
    );

    // First message should have text + image with validated URL
    let user_msg_1 = &client.history()[0];
    assert_eq!(
        user_msg_1.content.len(),
        2,
        "First message should have 2 blocks"
    );
    match &user_msg_1.content[1] {
        ContentBlock::Image(img) => {
            assert_eq!(img.url(), image_url, "Image URL should be preserved");
            assert_eq!(
                img.detail(),
                ImageDetail::High,
                "Detail level should be preserved"
            );
        }
        _ => panic!("Expected Image content block"),
    }

    // Second message should be text-only assistant response
    let assistant_msg = &client.history()[1];
    assert_eq!(
        assistant_msg.content.len(),
        1,
        "Assistant message should have 1 block"
    );
    match &assistant_msg.content[0] {
        ContentBlock::Text(text) => {
            assert_eq!(text.text, "I see the image");
        }
        _ => panic!("Expected Text content block"),
    }

    // Third message should be text-only user message
    let user_msg_2 = &client.history()[2];
    assert_eq!(
        user_msg_2.content.len(),
        1,
        "Second user message should have 1 block"
    );
    match &user_msg_2.content[0] {
        ContentBlock::Text(text) => {
            assert_eq!(text.text, "Can you describe it?");
        }
        _ => panic!("Expected Text content block"),
    }
}
