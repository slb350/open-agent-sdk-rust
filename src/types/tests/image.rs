#[test]
fn image_urls_preserve_supported_sources_and_default_detail() {
    for url in [
        "http://example.com/image.jpg",
        "https://example.com/image.jpg",
        "data:image/png;base64,AAAA",
        "data:image/png;charset=utf-8;base64,iVBORw0KGgoAAAA=",
    ] {
        let image = ImageBlock::from_url(url).unwrap();
        assert_eq!(image.url(), url);
        assert_eq!(image.detail(), ImageDetail::Auto);
    }
}

#[test]
fn image_urls_reject_empty_unsupported_and_malformed_sources() {
    for (url, expected_error) in [
        ("", "empty"),
        ("ftp://example.com/image.jpg", "http"),
        ("/images/photo.jpg", "http"),
        ("javascript:alert('XSS')", "http"),
        ("file:///etc/passwd", "http"),
        ("data:notanimage", "format"),
    ] {
        let error = ImageBlock::from_url(url).unwrap_err();
        assert!(matches!(error, crate::Error::InvalidInput(_)));
        assert!(
            error.to_string().contains(expected_error),
            "{url:?}: {error}"
        );
    }
}

#[test]
fn image_urls_reject_control_characters() {
    for control in ['\n', '\t', '\0', '\r', '\x1b'] {
        let error =
            ImageBlock::from_url(format!("https://example.com{control}/image.jpg")).unwrap_err();
        assert!(error.to_string().contains("control"), "{error}");
    }
}

#[test]
fn long_image_urls_remain_accepted_without_truncation() {
    for path_length in [1900, 2980] {
        let url = format!("https://example.com/{}", "a".repeat(path_length));
        assert_eq!(ImageBlock::from_url(&url).unwrap().url(), url);
    }
}

#[test]
fn base64_images_preserve_data_and_every_supported_mime_type() {
    for mime in [
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "image/avif",
        "image/bmp",
        "image/tiff",
    ] {
        let image = ImageBlock::from_base64("iVBORw0KGgoAAAA=", mime).unwrap();
        assert_eq!(image.url(), format!("data:{mime};base64,iVBORw0KGgoAAAA="));
        assert_eq!(image.detail(), ImageDetail::Auto);
    }
}

#[test]
fn base64_images_reject_empty_nonimage_and_injected_mime_types() {
    for mime in [
        "",
        "text/plain",
        "image/png;charset=utf-8",
        "image/png\n",
        "image/png\r",
        "image/png\r\n",
        "image/png,extra",
        "image/png\r\nX-Custom: bad",
    ] {
        let error = ImageBlock::from_base64("AAAA", mime).unwrap_err();
        assert!(matches!(error, crate::Error::InvalidInput(_)));
        assert!(error.to_string().contains("MIME"), "{mime:?}: {error}");
    }
}

#[test]
fn both_image_constructors_reject_invalid_base64() {
    for data in [
        "",
        "hello world",
        "test@data",
        "test#data",
        "test$data",
        "test%data",
        "abc\ndef",
        "@@@",
        "A",
        "AB",
        "ABC",
        "ABCD===",
        "==abc",
        "ab==cd",
        "ab==cdef",
        "A===",
        "éé",
    ] {
        for result in [
            ImageBlock::from_base64(data, "image/png"),
            ImageBlock::from_url(format!("data:image/png;base64,{data}")),
        ] {
            assert!(
                matches!(result.unwrap_err(), crate::Error::InvalidInput(_)),
                "{data:?}"
            );
        }
    }
}

#[test]
fn large_base64_image_remains_accepted_without_truncation() {
    let data = "A".repeat(15_000_000);
    let image = ImageBlock::from_base64(&data, "image/png").unwrap();
    assert_eq!(
        image.url().strip_prefix("data:image/png;base64,"),
        Some(data.as_str())
    );
}

#[test]
fn file_images_encode_bytes_and_validate_extensions() {
    use base64::{Engine as _, engine::general_purpose};

    let directory = tempfile::tempdir().unwrap();
    let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";
    let bytes = general_purpose::STANDARD.decode(png).unwrap();
    let image_path = directory.path().join("image.png");
    std::fs::write(&image_path, &bytes).unwrap();
    let image = ImageBlock::from_file_path(&image_path).unwrap();
    assert_eq!(
        image.url().strip_prefix("data:image/png;base64,"),
        Some(png)
    );
    assert_eq!(image.detail(), ImageDetail::Auto);

    for (filename, expected_error) in [("image", "extension"), ("image.txt", "Unsupported")] {
        let path = directory.path().join(filename);
        std::fs::write(&path, &bytes).unwrap();
        let error = ImageBlock::from_file_path(path).unwrap_err();
        assert!(error.to_string().contains(expected_error), "{error}");
    }
}

#[test]
fn image_detail_builder_display_and_serialization_agree() {
    for (detail, expected) in [
        (ImageDetail::Low, "low"),
        (ImageDetail::High, "high"),
        (ImageDetail::Auto, "auto"),
    ] {
        let image = ImageBlock::from_url("https://example.com/image.jpg")
            .unwrap()
            .with_detail(detail);
        assert_eq!(image.detail(), detail);
        assert_eq!(detail.to_string(), expected);
        assert_eq!(serde_json::to_value(detail).unwrap(), expected);
    }
}

#[test]
fn openai_text_content_roundtrips_as_a_string() {
    for text in ["", "Hello world"] {
        let content = OpenAIContent::Text(text.to_string());
        let value = serde_json::to_value(content).unwrap();
        assert_eq!(value, text);
        let decoded: OpenAIContent = serde_json::from_value(value).unwrap();
        assert!(matches!(decoded, OpenAIContent::Text(actual) if actual == text));
    }
}

#[test]
fn openai_parts_serialize_validated_images_and_text_as_distinct_variants() {
    let image = ImageBlock::from_base64("AAAA", "image/png")
        .unwrap()
        .with_detail(ImageDetail::High);
    let content = OpenAIContent::Parts(vec![
        OpenAIContentPart::text("Inspect:"),
        OpenAIContentPart::from_image(&image),
    ]);
    assert_eq!(
        serde_json::to_value(content).unwrap(),
        serde_json::json!([
            {"type": "text", "text": "Inspect:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}}
        ])
    );
}

#[test]
#[allow(deprecated)]
fn deprecated_image_part_constructor_keeps_its_wire_format() {
    let part = OpenAIContentPart::image_url("https://example.com/image.jpg", ImageDetail::Low);
    assert_eq!(
        serde_json::to_value(part).unwrap(),
        serde_json::json!({
            "type": "image_url", "image_url": {"url": "https://example.com/image.jpg", "detail": "low"}
        })
    );
}
