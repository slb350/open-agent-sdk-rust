    use crate::types::{OpenAIChoice, OpenAIDelta, OpenAIFunctionDelta, OpenAIToolCallDelta};

    #[test]
    fn test_tool_call_aggregator_text() {
        let mut aggregator = ToolCallAggregator::new();

        let chunk = OpenAIChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: Some("Hello ".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        };

        let blocks = aggregator.process_chunk(chunk).unwrap();
        assert_eq!(blocks.len(), 0); // Not finished yet

        let chunk2 = OpenAIChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: Some("world".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };

        let blocks = aggregator.process_chunk(chunk2).unwrap();
        assert_eq!(blocks.len(), 1);

        match &blocks[0] {
            ContentBlock::Text(text) => assert_eq!(text.text, "Hello world"),
            _ => panic!("Expected text block"),
        }
    }

    #[test]
    fn test_tool_call_aggregator_tool() {
        let mut aggregator = ToolCallAggregator::new();

        let chunk = OpenAIChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(vec![OpenAIToolCallDelta {
                        index: 0,
                        id: Some("call_123".to_string()),
                        call_type: Some("function".to_string()),
                        function: Some(OpenAIFunctionDelta {
                            name: Some("get_weather".to_string()),
                            arguments: Some(r#"{"location":"#.to_string()),
                        }),
                    }]),
                },
                finish_reason: None,
            }],
        };

        let blocks = aggregator.process_chunk(chunk).unwrap();
        assert_eq!(blocks.len(), 0);

        let chunk2 = OpenAIChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(vec![OpenAIToolCallDelta {
                        index: 0,
                        id: None,
                        call_type: None,
                        function: Some(OpenAIFunctionDelta {
                            name: None,
                            arguments: Some(r#""Paris"}"#.to_string()),
                        }),
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
        };

        let blocks = aggregator.process_chunk(chunk2).unwrap();
        assert_eq!(blocks.len(), 1);

        match &blocks[0] {
            ContentBlock::ToolUse(tool) => {
                assert_eq!(tool.id(), "call_123");
                assert_eq!(tool.name(), "get_weather");
                assert_eq!(tool.input()["location"], "Paris");
            }
            _ => panic!("Expected tool use block"),
        }
    }
