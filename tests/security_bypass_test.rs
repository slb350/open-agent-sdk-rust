//! Tests for security validation bypass fix
//!
//! These tests verify that image content parts MUST be constructed from validated
//! ImageBlock instances, preventing bypass of security validation (XSS, file disclosure, etc.)

use open_agent::{ImageBlock, OpenAIContentPart};

#[test]
fn test_from_image_requires_validated_imageblock() {
    // Create a validated ImageBlock (will pass validation checks)
    let image_block =
        ImageBlock::from_url("https://example.com/image.jpg").expect("Valid HTTPS URL should pass");

    // Construct OpenAIContentPart from validated ImageBlock
    let content_part = OpenAIContentPart::from_image(&image_block);

    // Verify it serializes correctly
    let json = serde_json::to_value(&content_part).expect("Should serialize");
    assert_eq!(json["type"], "image_url");
    assert_eq!(json["image_url"]["url"], "https://example.com/image.jpg");
}

#[test]
fn test_javascript_uri_cannot_bypass_validation() {
    // Malicious javascript: URI should be rejected by ImageBlock validation
    let result = ImageBlock::from_url("javascript:alert('XSS')");
    assert!(result.is_err(), "JavaScript URI should be rejected");

    // Since we can't create an ImageBlock with javascript: URI,
    // we also can't create an OpenAIContentPart with it via from_image()
    // This is the desired behavior - validation cannot be bypassed
}

#[test]
fn test_file_uri_cannot_bypass_validation() {
    // file:// URI should be rejected by ImageBlock validation
    let result = ImageBlock::from_url("file:///etc/passwd");
    assert!(result.is_err(), "File URI should be rejected");

    // Since we can't create an ImageBlock with file: URI,
    // we also can't create an OpenAIContentPart with it via from_image()
    // This is the desired behavior - validation cannot be bypassed
}

#[test]
fn test_data_uri_with_validated_base64() {
    // Create a validated ImageBlock from base64 (will pass validation)
    let base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    let image_block =
        ImageBlock::from_base64(base64_data, "image/png").expect("Valid base64 should pass");

    // Construct OpenAIContentPart from validated ImageBlock
    let content_part = OpenAIContentPart::from_image(&image_block);

    // Verify it serializes correctly with data URI
    let json = serde_json::to_value(&content_part).expect("Should serialize");
    assert_eq!(json["type"], "image_url");
    assert!(
        json["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
}
