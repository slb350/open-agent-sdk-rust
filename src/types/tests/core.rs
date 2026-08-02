
    #[test]
    fn test_agent_options_builder() {
        let options = AgentOptions::builder()
            .system_prompt("Test prompt")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .api_key("test-key")
            .max_turns(5)
            .max_tokens(1000)
            .temperature(0.5)
            .timeout(30)
            .auto_execute_tools(true)
            .max_tool_iterations(10)
            .build()
            .unwrap();

        assert_eq!(options.system_prompt, "Test prompt");
        assert_eq!(options.model, "test-model");
        assert_eq!(options.base_url, "http://localhost:1234/v1");
        assert_eq!(options.api_key, "test-key");
        assert_eq!(options.max_turns, 5);
        assert_eq!(options.max_tokens, Some(1000));
        assert_eq!(options.temperature, 0.5);
        assert_eq!(options.timeout, 30);
        assert!(options.auto_execute_tools);
        assert_eq!(options.max_tool_iterations, 10);
    }

    #[test]
    fn test_agent_options_builder_defaults() {
        let options = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        assert_eq!(options.system_prompt, "");
        assert_eq!(options.api_key, "not-needed");
        assert_eq!(options.max_turns, 1);
        assert_eq!(options.max_tokens, Some(4096));
        assert_eq!(options.temperature, 0.7);
        assert_eq!(options.timeout, 60);
        assert!(!options.auto_execute_tools);
        assert_eq!(options.max_tool_iterations, 5);
    }

    #[test]
    fn test_agent_options_builder_missing_required() {
        // Missing model
        let result = AgentOptions::builder()
            .base_url("http://localhost:1234/v1")
            .build();
        assert!(result.is_err());

        // Missing base_url
        let result = AgentOptions::builder().model("test-model").build();
        assert!(result.is_err());
    }

    #[test]
    fn test_message_user() {
        let msg = Message::user("Hello");
        assert!(matches!(msg.role, MessageRole::User));
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            ContentBlock::Text(text) => assert_eq!(text.text, "Hello"),
            _ => panic!("Expected TextBlock"),
        }
    }

    #[test]
    fn test_message_system() {
        let msg = Message::system("System prompt");
        assert!(matches!(msg.role, MessageRole::System));
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            ContentBlock::Text(text) => assert_eq!(text.text, "System prompt"),
            _ => panic!("Expected TextBlock"),
        }
    }

    #[test]
    fn test_message_assistant() {
        let content = vec![ContentBlock::Text(TextBlock::new("Response"))];
        let msg = Message::assistant(content);
        assert!(matches!(msg.role, MessageRole::Assistant));
        assert_eq!(msg.content.len(), 1);
    }

    #[test]
    fn test_message_user_with_image() {
        let msg =
            Message::user_with_image("What's in this image?", "https://example.com/image.jpg")
                .unwrap();
        assert!(matches!(msg.role, MessageRole::User));
        assert_eq!(msg.content.len(), 2);

        // Should have text first, then image
        match &msg.content[0] {
            ContentBlock::Text(text) => assert_eq!(text.text, "What's in this image?"),
            _ => panic!("Expected TextBlock at position 0"),
        }
        match &msg.content[1] {
            ContentBlock::Image(image) => {
                assert_eq!(image.url(), "https://example.com/image.jpg");
                assert_eq!(image.detail(), ImageDetail::Auto);
            }
            _ => panic!("Expected ImageBlock at position 1"),
        }
    }

    #[test]
    fn test_message_user_with_image_and_detail() {
        let msg = Message::user_with_image_detail(
            "Analyze this in detail",
            "https://example.com/diagram.png",
            ImageDetail::High,
        )
        .unwrap();
        assert!(matches!(msg.role, MessageRole::User));
        assert_eq!(msg.content.len(), 2);

        match &msg.content[1] {
            ContentBlock::Image(image) => {
                assert_eq!(image.detail(), ImageDetail::High);
            }
            _ => panic!("Expected ImageBlock"),
        }
    }

    #[test]
    fn test_message_user_with_base64_image() {
        let base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ";
        let msg =
            Message::user_with_base64_image("What's this?", base64_data, "image/png").unwrap();
        assert!(matches!(msg.role, MessageRole::User));
        assert_eq!(msg.content.len(), 2);

        match &msg.content[1] {
            ContentBlock::Image(image) => {
                assert!(image.url().starts_with("data:image/png;base64,"));
                assert!(image.url().contains(base64_data));
            }
            _ => panic!("Expected ImageBlock"),
        }
    }

    #[test]
    fn test_text_block() {
        let block = TextBlock::new("Hello");
        assert_eq!(block.text, "Hello");
    }

    #[test]
    fn test_tool_use_block() {
        let input = serde_json::json!({"arg": "value"});
        let block = ToolUseBlock::new("call_123", "tool_name", input.clone());
        assert_eq!(block.id(), "call_123");
        assert_eq!(block.name(), "tool_name");
        assert_eq!(block.input(), &input);
    }

    #[test]
    fn test_tool_result_block() {
        let content = serde_json::json!({"result": "success"});
        let block = ToolResultBlock::new("call_123", content.clone());
        assert_eq!(block.tool_use_id(), "call_123");
        assert_eq!(block.content(), &content);
    }

    // ========================================================================
    // Private Field Getters Tests (Issue #3 - RED Phase)
    // ========================================================================

    #[test]
    fn test_tool_use_block_getters() {
        // RED: Test getter methods for ToolUseBlock (don't exist yet)
        let input = serde_json::json!({"x": 5});
        let block = ToolUseBlock::new("call_123", "calculator", input.clone());

        // These should compile with getters
        assert_eq!(block.id(), "call_123");
        assert_eq!(block.name(), "calculator");
        assert_eq!(block.input(), &input);
    }

    #[test]
    fn test_tool_result_block_getters() {
        // RED: Test getter methods for ToolResultBlock (don't exist yet)
        let content = serde_json::json!({"answer": 42});
        let result = ToolResultBlock::new("call_123", content.clone());

        assert_eq!(result.tool_use_id(), "call_123");
        assert_eq!(result.content(), &content);
    }

    #[test]
    fn test_message_role_serialization() {
        assert_eq!(
            serde_json::to_string(&MessageRole::User).unwrap(),
            "\"user\""
        );
        assert_eq!(
            serde_json::to_string(&MessageRole::System).unwrap(),
            "\"system\""
        );
        assert_eq!(
            serde_json::to_string(&MessageRole::Assistant).unwrap(),
            "\"assistant\""
        );
        assert_eq!(
            serde_json::to_string(&MessageRole::Tool).unwrap(),
            "\"tool\""
        );
    }

    #[test]
    fn test_openai_request_serialization() {
        let request = OpenAIRequest {
            model: "gpt-3.5".to_string(),
            messages: vec![OpenAIMessage {
                role: "user".to_string(),
                content: Some(OpenAIContent::Text("Hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: true,
            max_tokens: Some(100),
            temperature: Some(0.7),
            tools: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("gpt-3.5"));
        assert!(json.contains("Hello"));
        assert!(json.contains("\"stream\":true"));
    }

    #[test]
    fn test_openai_chunk_deserialization() {
        let json = r#"{
            "id": "chunk_1",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-3.5",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "Hello"
                },
                "finish_reason": null
            }]
        }"#;

        let chunk: OpenAIChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.id, "chunk_1");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    }

    #[test]
    fn test_content_block_serialization() {
        let text_block = ContentBlock::Text(TextBlock::new("Hello"));
        let json = serde_json::to_string(&text_block).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_agent_options_clone() {
        let options1 = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let options2 = options1.clone();
        assert_eq!(options1.model, options2.model);
        assert_eq!(options1.base_url, options2.base_url);
    }

    #[test]
    fn test_temperature_validation() {
        // Temperature too low (< 0.0)
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .temperature(-0.1)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("temperature"));

        // Temperature too high (> 2.0)
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .temperature(2.1)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("temperature"));

        // Valid temperatures should work
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .temperature(0.0)
            .build();
        assert!(result.is_ok());

        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .temperature(2.0)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_url_validation() {
        // Empty URL should fail
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("")
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("base_url"));

        // Invalid URL format should fail
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("not-a-url")
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("base_url"));

        // Valid URLs should work
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build();
        assert!(result.is_ok());

        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("https://api.openai.com/v1")
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_validation() {
        // Empty model should fail
        let result = AgentOptions::builder()
            .model("")
            .base_url("http://localhost:1234/v1")
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("model"));

        // Whitespace-only model should fail
        let result = AgentOptions::builder()
            .model("   ")
            .base_url("http://localhost:1234/v1")
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("model"));
    }

    #[test]
    fn test_max_tokens_validation() {
        // max_tokens = 0 should fail
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .max_tokens(0)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_tokens"));

        // Valid max_tokens should work
        let result = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .max_tokens(1)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_agent_options_getters() {
        // Test that AgentOptions provides getter methods for field access
        let options = AgentOptions::builder()
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .system_prompt("Test prompt")
            .api_key("test-key")
            .max_turns(5)
            .max_tokens(1000)
            .temperature(0.5)
            .timeout(30)
            .auto_execute_tools(true)
            .max_tool_iterations(10)
            .build()
            .unwrap();

        // All fields should be accessible via getter methods, not direct field access
        assert_eq!(options.system_prompt(), "Test prompt");
        assert_eq!(options.model(), "test-model");
        assert_eq!(options.base_url(), "http://localhost:1234/v1");
        assert_eq!(options.api_key(), "test-key");
        assert_eq!(options.max_turns(), 5);
        assert_eq!(options.max_tokens(), Some(1000));
        assert_eq!(options.temperature(), 0.5);
        assert_eq!(options.timeout(), 30);
        assert!(options.auto_execute_tools());
        assert_eq!(options.max_tool_iterations(), 10);
        assert_eq!(options.tools().len(), 0);
    }

    // ========================================================================
    // Image Support Tests (Phase 1 - TDD RED)
    // ========================================================================
