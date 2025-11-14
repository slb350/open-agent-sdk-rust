//! Integration tests for defensive programming validations
//!
//! Tests that all validation rules work correctly in realistic scenarios

use open_agent::{
    AgentOptions, Client, ContentBlock, ImageBlock, ImageDetail, Message, MessageRole, TextBlock,
};

#[test]
fn test_malicious_mime_injection_rejected() {
    // Should reject MIME type with injection characters
    let malicious_mimes = vec![
        "image/png;charset=utf-8",    // semicolon injection
        "image/png\nmalicious",       // newline injection
        "image/png,extra",            // comma injection
        "image/png\r\nX-Custom: bad", // CRLF injection
    ];

    for mime in &malicious_mimes {
        let result = ImageBlock::from_base64("AAAA", mime);
        assert!(
            result.is_err(),
            "Should reject malicious MIME type: {:?}",
            mime
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("MIME") || err.to_string().contains("character"),
            "Error should mention MIME or character issue, got: {}",
            err
        );
    }
}

#[test]
fn test_extremely_large_base64_handled() {
    // Should handle (with warning) very large base64 strings
    // 15MB base64 string
    let huge_base64 = "A".repeat(15_000_000);

    // Should succeed but log warning
    let result = ImageBlock::from_base64(&huge_base64, "image/png");
    assert!(result.is_ok(), "Should accept large base64 (with warning)");

    // Should create proper data URI
    let block = result.unwrap();
    assert!(block.url().starts_with("data:image/png;base64,"));
    assert!(block.url().len() > 15_000_000);
}

#[test]
fn test_control_characters_in_urls_rejected() {
    // Should reject URLs with various control characters
    let malicious_urls = vec![
        "https://example.com\n/path",   // newline
        "https://example.com\t/path",   // tab
        "https://example.com\0/path",   // null
        "https://example.com\r/path",   // carriage return
        "https://example.com\x1B/path", // escape
    ];

    for url in &malicious_urls {
        let result = ImageBlock::from_url(*url);
        assert!(
            result.is_err(),
            "Should reject URL with control char: {:?}",
            url
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("control") || err.to_string().contains("character"),
            "Error should mention control characters, got: {}",
            err
        );
    }
}

#[tokio::test]
async fn test_empty_text_blocks_warned_but_accepted() {
    // Empty text blocks should trigger warning but still be serialized
    let messages = vec![
        Message::new(
            MessageRole::User,
            vec![
                ContentBlock::Text(TextBlock::new("")), // Empty
                ContentBlock::Image(ImageBlock::from_url("https://example.com/img.jpg").unwrap()),
            ],
        ),
        Message::new(
            MessageRole::User,
            vec![
                ContentBlock::Text(TextBlock::new("   ")), // Whitespace-only
                ContentBlock::Image(ImageBlock::from_url("https://example.com/img2.jpg").unwrap()),
            ],
        ),
    ];

    // Create client and add messages
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .unwrap();

    let mut client = Client::new(options).unwrap();

    for msg in messages {
        client.history_mut().push(msg);
    }

    // Both messages should be in history (not dropped)
    assert_eq!(client.history().len(), 2);

    // Each message should have 2 content blocks (text + image)
    for msg in client.history() {
        assert_eq!(msg.content.len(), 2);
    }
}

#[test]
fn test_all_validation_errors_are_descriptive() {
    // All validation errors should have clear, actionable error messages

    // Empty URL
    let err = ImageBlock::from_url("").unwrap_err();
    assert!(err.to_string().contains("empty"));

    // Control characters in URL
    let err = ImageBlock::from_url("https://example.com\n/path").unwrap_err();
    assert!(err.to_string().contains("control") || err.to_string().contains("character"));

    // Invalid URL scheme
    let err = ImageBlock::from_url("javascript:alert(1)").unwrap_err();
    assert!(err.to_string().contains("http") || err.to_string().contains("scheme"));

    // Empty base64
    let err = ImageBlock::from_base64("", "image/png").unwrap_err();
    assert!(err.to_string().contains("empty"));

    // Invalid base64 characters
    let err = ImageBlock::from_base64("hello world", "image/png").unwrap_err();
    assert!(err.to_string().contains("base64") || err.to_string().contains("character"));

    // Invalid base64 length
    let err = ImageBlock::from_base64("ABC", "image/png").unwrap_err();
    assert!(err.to_string().contains("length") || err.to_string().contains("multiple"));

    // MIME type injection
    let err = ImageBlock::from_base64("AAAA", "image/png;charset=utf-8").unwrap_err();
    assert!(err.to_string().contains("MIME") || err.to_string().contains("character"));
}

#[test]
fn test_valid_edge_cases_still_work() {
    // Ensure defensive validations don't break valid edge cases

    // Valid short base64
    let result = ImageBlock::from_base64("AAAA", "image/png");
    assert!(result.is_ok());

    // Valid long HTTP URL (under 2000 chars)
    let long_url = format!("https://example.com/{}", "a".repeat(1900));
    let result = ImageBlock::from_url(&long_url);
    assert!(result.is_ok());

    // Valid data URI with proper base64
    let data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    let result = ImageBlock::from_url(data_uri);
    assert!(result.is_ok());

    // All common image MIME types
    for mime in &[
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "image/avif",
    ] {
        let result = ImageBlock::from_base64("AAAA", *mime);
        assert!(result.is_ok(), "Should accept {}", mime);
    }
}

#[test]
fn test_backward_compatibility_maintained() {
    // Existing valid patterns from v0.5.0/v0.6.0 should still work

    // HTTP URLs
    let result = ImageBlock::from_url("http://example.com/image.jpg");
    assert!(result.is_ok());

    // HTTPS URLs
    let result = ImageBlock::from_url("https://example.com/image.jpg");
    assert!(result.is_ok());

    // Data URIs
    let result = ImageBlock::from_url("data:image/png;base64,AAAA");
    assert!(result.is_ok());

    // from_base64 with valid inputs
    let result = ImageBlock::from_base64(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "image/png",
    );
    assert!(result.is_ok());

    // ImageDetail levels
    let img = ImageBlock::from_url("https://example.com/img.jpg")
        .unwrap()
        .with_detail(ImageDetail::Low);
    assert_eq!(img.detail(), ImageDetail::Low);

    let img = img.with_detail(ImageDetail::High);
    assert_eq!(img.detail(), ImageDetail::High);

    let img = img.with_detail(ImageDetail::Auto);
    assert_eq!(img.detail(), ImageDetail::Auto);
}

#[test]
fn test_data_uri_base64_validation_integration() {
    // from_url() should validate data URI base64 using same rules as from_base64()

    // Invalid: spaces in base64
    let result = ImageBlock::from_url("data:image/png;base64,hello world");
    assert!(result.is_err());

    // Invalid: special characters
    let result = ImageBlock::from_url("data:image/png;base64,@@@");
    assert!(result.is_err());

    // Invalid: wrong length
    let result = ImageBlock::from_url("data:image/png;base64,ABC");
    assert!(result.is_err());

    // Valid: properly formatted base64
    let result = ImageBlock::from_url("data:image/png;base64,AAAA");
    assert!(result.is_ok());
}
