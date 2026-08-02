    #[test]
    fn test_image_block_from_url() {
        // Should create ImageBlock from URL
        let block = ImageBlock::from_url("https://example.com/image.jpg").unwrap();
        assert_eq!(block.url(), "https://example.com/image.jpg");
        assert!(matches!(block.detail(), ImageDetail::Auto));
    }

    #[test]
    fn test_image_block_from_base64() {
        // Should create ImageBlock from base64
        let block = ImageBlock::from_base64("iVBORw0KGgoAAAA=", "image/jpeg").unwrap();
        assert!(block.url().starts_with("data:image/jpeg;base64,"));
        assert!(matches!(block.detail(), ImageDetail::Auto));
    }

    #[test]
    fn test_image_block_from_file_path() {
        use base64::{Engine as _, engine::general_purpose};
        use std::io::Write;

        // Create a temporary test file
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_image.png");

        // Write a minimal valid 1x1 PNG (red pixel)
        let png_bytes = general_purpose::STANDARD
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==")
            .unwrap();
        std::fs::File::create(&test_file)
            .unwrap()
            .write_all(&png_bytes)
            .unwrap();

        // Test: Should successfully load the file
        let block = ImageBlock::from_file_path(&test_file).unwrap();
        assert!(block.url().starts_with("data:image/png;base64,"));
        assert!(matches!(block.detail(), ImageDetail::Auto));

        // Test: Missing extension should fail
        let no_ext_file = temp_dir.join("test_image_no_ext");
        std::fs::File::create(&no_ext_file)
            .unwrap()
            .write_all(&png_bytes)
            .unwrap();
        let result = ImageBlock::from_file_path(&no_ext_file);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("extension"));

        // Test: Unsupported extension should fail
        let bad_ext_file = temp_dir.join("test_image.txt");
        std::fs::File::create(&bad_ext_file)
            .unwrap()
            .write_all(&png_bytes)
            .unwrap();
        let result = ImageBlock::from_file_path(&bad_ext_file);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported"));

        // Cleanup
        let _ = std::fs::remove_file(&test_file);
        let _ = std::fs::remove_file(&no_ext_file);
        let _ = std::fs::remove_file(&bad_ext_file);
    }

    #[test]
    fn test_image_block_with_detail() {
        // Should set detail level
        let block = ImageBlock::from_url("https://example.com/image.jpg")
            .unwrap()
            .with_detail(ImageDetail::High);
        assert!(matches!(block.detail(), ImageDetail::High));
    }

    #[test]
    fn test_image_detail_serialization() {
        // Should serialize ImageDetail to correct strings
        let json = serde_json::to_string(&ImageDetail::Low).unwrap();
        assert_eq!(json, "\"low\"");

        let json = serde_json::to_string(&ImageDetail::High).unwrap();
        assert_eq!(json, "\"high\"");

        let json = serde_json::to_string(&ImageDetail::Auto).unwrap();
        assert_eq!(json, "\"auto\"");
    }

    #[test]
    fn test_content_block_image_variant() {
        // Should add Image variant to ContentBlock
        let image = ImageBlock::from_url("https://example.com/image.jpg").unwrap();
        let block = ContentBlock::Image(image);

        match block {
            ContentBlock::Image(img) => {
                assert_eq!(img.url(), "https://example.com/image.jpg");
            }
            _ => panic!("Expected Image variant"),
        }
    }

    #[test]
    fn test_openai_content_text_format() {
        // Should serialize text-only as string (backward compat)
        let content = OpenAIContent::Text("Hello".to_string());
        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(json, serde_json::json!("Hello"));
    }

    #[test]
    #[allow(deprecated)]
    fn test_openai_content_parts_format() {
        // Should serialize mixed content as array
        let parts = vec![
            OpenAIContentPart::text("What's in this image?"),
            OpenAIContentPart::image_url("https://example.com/img.jpg", ImageDetail::High),
        ];
        let content = OpenAIContent::Parts(parts);
        let json = serde_json::to_value(&content).unwrap();

        assert!(json.is_array());
        assert_eq!(json[0]["type"], "text");
        assert_eq!(json[0]["text"], "What's in this image?");
        assert_eq!(json[1]["type"], "image_url");
        assert_eq!(json[1]["image_url"]["url"], "https://example.com/img.jpg");
        assert_eq!(json[1]["image_url"]["detail"], "high");
    }

    // ========================================================================
    // OpenAIContentPart Enum Tests (Phase 4 - PR #3 Fixes)
    // ========================================================================

    #[test]
    fn test_openai_content_part_text_serialization() {
        // RED: Test that text variant serializes correctly with enum
        let part = OpenAIContentPart::text("Hello world");
        let json = serde_json::to_value(&part).unwrap();

        // Should have type field with value "text"
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "Hello world");
        // Should not have image_url field
        assert!(json.get("image_url").is_none());
    }

    #[test]
    #[allow(deprecated)]
    fn test_openai_content_part_image_serialization() {
        // RED: Test that image_url variant serializes correctly with enum
        let part = OpenAIContentPart::image_url("https://example.com/img.jpg", ImageDetail::Low);
        let json = serde_json::to_value(&part).unwrap();

        // Should have type field with value "image_url"
        assert_eq!(json["type"], "image_url");
        assert_eq!(json["image_url"]["url"], "https://example.com/img.jpg");
        assert_eq!(json["image_url"]["detail"], "low");
        // Should not have text field
        assert!(json.get("text").is_none());
    }

    #[test]
    #[allow(deprecated)]
    fn test_openai_content_part_enum_exhaustiveness() {
        // RED: Test that enum prevents invalid states
        // With tagged enum, it should be impossible to create a part with both text and image_url
        // or a part with neither. This test documents expected enum behavior.

        let text_part = OpenAIContentPart::text("test");
        let image_part = OpenAIContentPart::image_url("url", ImageDetail::Auto);

        // Pattern matching should be exhaustive
        match text_part {
            OpenAIContentPart::Text { .. } => {
                // Expected for text part
            }
            OpenAIContentPart::ImageUrl { .. } => {
                panic!("Text part should not match ImageUrl variant");
            }
        }

        match image_part {
            OpenAIContentPart::Text { .. } => {
                panic!("Image part should not match Text variant");
            }
            OpenAIContentPart::ImageUrl { .. } => {
                // Expected for image part
            }
        }
    }

    #[test]
    fn test_image_detail_display() {
        // Should convert ImageDetail to string
        assert_eq!(ImageDetail::Low.to_string(), "low");
        assert_eq!(ImageDetail::High.to_string(), "high");
        assert_eq!(ImageDetail::Auto.to_string(), "auto");
    }

    // ========================================================================
    // ImageBlock Validation Tests (Phase 1 - PR #3 Fixes)
    // ========================================================================

    #[test]
    fn test_image_block_from_url_rejects_empty() {
        // Should reject empty URL strings
        let result = ImageBlock::from_url("");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_image_block_from_url_rejects_invalid_scheme() {
        // Should reject non-HTTP/HTTPS/data schemes
        let result = ImageBlock::from_url("ftp://example.com/image.jpg");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("scheme") || err.to_string().contains("http"));
    }

    #[test]
    fn test_image_block_from_url_rejects_relative_path() {
        // Should reject relative paths
        let result = ImageBlock::from_url("/images/photo.jpg");
        assert!(result.is_err());
        // Error message should mention URL requirements
        assert!(matches!(result.unwrap_err(), crate::Error::InvalidInput(_)));
    }

    #[test]
    fn test_image_block_from_url_accepts_http() {
        // Should accept HTTP URLs
        let result = ImageBlock::from_url("http://example.com/image.jpg");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().url(), "http://example.com/image.jpg");
    }

    #[test]
    fn test_image_block_from_url_accepts_https() {
        // Should accept HTTPS URLs
        let result = ImageBlock::from_url("https://example.com/image.jpg");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().url(), "https://example.com/image.jpg");
    }

    #[test]
    fn test_image_block_from_url_accepts_data_uri() {
        // Should accept data URIs
        let data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let result = ImageBlock::from_url(data_uri);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().url(), data_uri);
    }

    #[test]
    fn test_image_block_from_url_rejects_malformed_data_uri() {
        // Should reject malformed data URIs
        let result = ImageBlock::from_url("data:notanimage");
        assert!(result.is_err());
        // Should return InvalidInput error for malformed data URI
        assert!(matches!(result.unwrap_err(), crate::Error::InvalidInput(_)));
    }

    // Phase 2: Enhanced URL validation tests (RED)

    #[test]
    fn test_from_url_rejects_control_characters() {
        // Should reject URLs with control characters
        let invalid_urls = [
            "https://example.com\n/image.jpg", // newline
            "https://example.com\t/image.jpg", // tab
            "https://example.com\0/image.jpg", // null
            "https://example.com\r/image.jpg", // carriage return
        ];

        for url in &invalid_urls {
            let result = ImageBlock::from_url(*url);
            assert!(
                result.is_err(),
                "Should reject URL with control characters: {:?}",
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

    #[test]
    fn test_from_url_warns_very_long_url() {
        // Should warn (but accept) very long URLs (>2000 chars)
        // 3000-char URL
        let long_url = format!("https://example.com/{}", "a".repeat(2980));

        // Should succeed but log a warning
        let result = ImageBlock::from_url(&long_url);
        assert!(result.is_ok(), "Should accept long URL (with warning)");

        // Verify the URL was stored
        let block = result.unwrap();
        assert_eq!(block.url().len(), 3000);
    }

    #[test]
    fn test_from_url_validates_data_uri_base64() {
        // Should validate base64 portion of data URIs
        let invalid_data_uris = [
            "data:image/png;base64,",            // empty base64
            "data:image/png;base64,hello world", // spaces in base64
            "data:image/png;base64,@@@",         // invalid chars
            "data:image/png;base64,ABC",         // invalid length (not divisible by 4)
            "data:image/png;base64,==abc",       // padding in middle (not at end)
            "data:image/png;base64,ab==cd",      // padding in middle
        ];

        for uri in &invalid_data_uris {
            let result = ImageBlock::from_url(*uri);
            assert!(
                result.is_err(),
                "Should reject data URI with invalid base64: {}",
                uri
            );
        }
    }

    #[test]
    fn test_from_url_rejects_javascript_scheme() {
        // Should explicitly reject javascript: scheme (XSS risk)
        let result = ImageBlock::from_url("javascript:alert(1)");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("http") || err.to_string().contains("scheme"),
            "Error should mention scheme requirements, got: {}",
            err
        );
    }

    #[test]
    fn test_from_url_rejects_file_scheme() {
        // Should reject file: scheme (security risk)
        let result = ImageBlock::from_url("file:///etc/passwd");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("http") || err.to_string().contains("scheme"),
            "Error should mention scheme requirements, got: {}",
            err
        );
    }

    #[test]
    fn test_image_block_from_base64_rejects_empty() {
        // Should reject empty base64 data
        let result = ImageBlock::from_base64("", "image/png");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_image_block_from_base64_rejects_invalid_mime() {
        // Should reject non-image MIME types
        let result = ImageBlock::from_base64("somedata", "text/plain");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("MIME") || err.to_string().contains("image"));
    }

    #[test]
    fn test_image_block_from_base64_accepts_valid_input() {
        // Should accept valid base64 data with image MIME type
        let base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let result = ImageBlock::from_base64(base64, "image/png");
        assert!(result.is_ok());
        let block = result.unwrap();
        assert!(block.url().starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_image_block_from_base64_accepts_all_image_types() {
        // Should accept all common image MIME types
        let base64 = "iVBORw0KGgo="; // Valid base64 (length multiple of 4)
        let mime_types = ["image/jpeg", "image/png", "image/gif", "image/webp"];

        for mime in &mime_types {
            let result = ImageBlock::from_base64(base64, *mime);
            assert!(result.is_ok(), "Should accept {}", mime);
            let block = result.unwrap();
            assert!(block.url().starts_with(&format!("data:{};base64,", mime)));
        }
    }

    // Phase 1: Enhanced base64 validation tests (RED)

    #[test]
    fn test_from_base64_rejects_invalid_characters() {
        // Should reject base64 with invalid characters
        let invalid_inputs = [
            "hello world", // spaces
            "test@data",   // @
            "test#data",   // #
            "test$data",   // $
            "test%data",   // %
            "abc\ndef",    // newline
        ];

        for invalid in &invalid_inputs {
            let result = ImageBlock::from_base64(invalid, "image/png");
            assert!(
                result.is_err(),
                "Should reject base64 with invalid characters: {}",
                invalid
            );
            let err = result.unwrap_err();
            assert!(
                err.to_string().contains("base64") || err.to_string().contains("character"),
                "Error should mention base64 or character issue, got: {}",
                err
            );
        }
    }

    #[test]
    fn test_from_base64_rejects_malformed_padding() {
        // Should reject base64 with incorrect padding
        let invalid_padding = [
            "A",       // Length 1 (not divisible by 4)
            "AB",      // Length 2 (not divisible by 4)
            "ABC",     // Length 3 (not divisible by 4)
            "ABCD===", // Too many padding characters
        ];

        for invalid in &invalid_padding {
            let result = ImageBlock::from_base64(invalid, "image/png");
            assert!(
                result.is_err(),
                "Should reject malformed padding: {}",
                invalid
            );
        }
    }

    #[test]
    fn test_from_base64_rejects_mime_with_semicolon() {
        // Should reject MIME types with injection characters (semicolon)
        // Use valid base64 (length divisible by 4)
        let result = ImageBlock::from_base64("AAAA", "image/png;charset=utf-8");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("MIME") || err.to_string().contains("character"),
            "Error should mention MIME or character issue, got: {}",
            err
        );
    }

    #[test]
    fn test_from_base64_rejects_mime_with_newline() {
        // Should reject MIME types with control characters (newline)
        let invalid_mimes = [
            "image/png\n",
            "image/png\r",
            "image/png\r\n",
            "image/png,extra",
        ];

        for mime in &invalid_mimes {
            // Use valid base64 (length divisible by 4)
            let result = ImageBlock::from_base64("AAAA", mime);
            assert!(
                result.is_err(),
                "Should reject MIME with control/injection chars: {:?}",
                mime
            );
        }
    }

    #[test]
    fn test_from_base64_warns_large_data() {
        // Should warn (but accept) very large base64 strings (>10MB)
        // 15MB base64 = 15,000,000 chars
        let large_base64 = "A".repeat(15_000_000);

        // This should succeed but log a warning
        let result = ImageBlock::from_base64(&large_base64, "image/png");
        assert!(result.is_ok(), "Should accept large base64 (with warning)");

        // Verify the data URI was created
        let block = result.unwrap();
        assert!(block.url().len() > 15_000_000);
    }

    #[test]
    fn test_from_base64_accepts_all_image_mime_types() {
        // Should accept all common image MIME types
        let valid_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let mime_types = [
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
            "image/avif",
            "image/bmp",
            "image/tiff",
        ];

        for mime in &mime_types {
            let result = ImageBlock::from_base64(valid_data, *mime);
            assert!(result.is_ok(), "Should accept valid MIME type: {}", mime);
        }
    }

    #[test]
    fn test_image_block_from_base64_rejects_empty_mime() {
        // Should reject empty MIME type
        let result = ImageBlock::from_base64("somedata", "");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("MIME") || err.to_string().contains("empty"));
    }
