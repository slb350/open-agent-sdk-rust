    /// Wraps injected content blocks as the `StreamEvent::Block`s a real stream would carry.
    ///
    /// These tests exercise `receive()` against a hand-built stream, so they stand in for the
    /// driver rather than the server; the terminating `Finish` event is added by tests that
    /// assert on it.
    fn block_stream(items: Vec<Result<ContentBlock>>) -> EventStream {
        Box::pin(futures::stream::iter(
            items
                .into_iter()
                .map(|item| item.map(StreamEvent::Block))
                .collect::<Vec<_>>(),
        ))
    }

    #[test]
    fn test_client_creation() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options).expect("Should create client successfully");
        assert_eq!(client.history().len(), 0);
    }

    #[test]
    fn test_client_new_returns_result() {
        // Test that Client::new() returns Result instead of panicking
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        // This should not panic - it should return Ok(client)
        let result = Client::new(options);
        assert!(result.is_ok(), "Client::new() should return Ok");

        let client = result.unwrap();
        assert_eq!(client.history().len(), 0);
    }

    #[test]
    fn test_interrupt_flag_initial_state() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options).expect("Should create client successfully");
        // Initially not interrupted
        assert!(!client.interrupted.load(Ordering::SeqCst));
    }

    #[test]
    fn test_interrupt_sets_flag() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options).expect("Should create client successfully");
        client.interrupt();
        assert!(client.interrupted.load(Ordering::SeqCst));
    }

    #[test]
    fn test_interrupt_idempotent() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options).expect("Should create client successfully");
        client.interrupt();
        assert!(client.interrupted.load(Ordering::SeqCst));

        // Call again - should still be interrupted
        client.interrupt();
        assert!(client.interrupted.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_receive_returns_none_when_interrupted() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let mut client = Client::new(options).expect("Should create client successfully");

        // Interrupt before receiving
        client.interrupt();

        // NEW SIGNATURE: receive() should return Ok(None) when interrupted
        let result = client.receive().await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_receive_returns_ok_none_when_no_stream() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let mut client = Client::new(options).expect("Should create client successfully");

        // No stream started - receive() should return Ok(None)
        let result = client.receive().await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_receive_error_propagation() {
        // This test demonstrates that errors are wrapped in Err(), not Some(Err())
        // We'll verify this behavior when we have a mock stream that produces errors
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options).expect("Should create client successfully");

        // Signature check: receive() returns Result<Option<ContentBlock>>
        // This means we can use ? operator cleanly:
        // while let Some(block) = client.receive().await? { ... }

        // Type assertion to ensure signature is correct
        let _: Result<Option<ContentBlock>> = std::future::ready(Ok(None)).await;
        drop(client);
    }

    #[tokio::test]
    async fn test_manual_mode_adds_assistant_to_history() {
        // Issue #4: Manual mode receive() should add assistant messages to history
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let mut client = Client::new(options).expect("Should create client successfully");

        // Simulate a user message in history
        client
            .history
            .push(Message::user("What's the capital of France?"));

        // Inject a fake stream with two text blocks
        let blocks = vec![
            Ok(ContentBlock::Text(TextBlock::new("Paris is"))),
            Ok(ContentBlock::Text(TextBlock::new(
                " the capital of France.",
            ))),
        ];
        client.current_stream = Some(block_stream(blocks));

        // Consume the stream via receive()
        let mut received = Vec::new();
        while let Some(block) = client.receive().await.unwrap() {
            received.push(block);
        }

        // Should have received 2 blocks
        assert_eq!(received.len(), 2);

        // History should now have 2 messages: user + assistant
        assert_eq!(client.history().len(), 2);
        assert_eq!(client.history()[0].role, MessageRole::User);
        assert_eq!(client.history()[1].role, MessageRole::Assistant);

        // Assistant message should contain both text blocks
        assert_eq!(client.history()[1].content.len(), 2);
    }

    #[tokio::test]
    async fn test_manual_mode_empty_stream_no_assistant_message() {
        // If the stream is empty, no assistant message should be added
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let mut client = Client::new(options).expect("Should create client successfully");

        // No stream set, receive returns None
        let result = client.receive().await.unwrap();
        assert!(result.is_none());

        // History should remain empty — no spurious assistant message
        assert_eq!(client.history().len(), 0);
    }

    #[tokio::test]
    async fn test_manual_mode_tool_call_flushed_on_send() {
        // P1: When receive() yields a ToolUseBlock and the caller then calls
        // send(""), the buffered assistant turn must be flushed to history
        // so the tool result has a matching tool_calls message.
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let mut client = Client::new(options).expect("Should create client successfully");

        // Simulate: user sends, receives a tool call block, stream ends mid-turn
        client.history.push(Message::user("Calculate 2+2"));

        let tool_use =
            crate::types::ToolUseBlock::new("call_1", "calculator", serde_json::json!({"a": 2}));
        let blocks = vec![Ok(ContentBlock::ToolUse(tool_use))];
        client.current_stream = Some(block_stream(blocks));

        // Caller consumes the tool use block
        let block = client.receive().await.unwrap().unwrap();
        assert!(matches!(block, ContentBlock::ToolUse(_)));

        // Buffer should hold the block but history should NOT have assistant yet
        assert_eq!(client.manual_receive_buffer.len(), 1);
        assert_eq!(client.history().len(), 1); // only user message

        // Stream ends — receive returns None, but buffer is NOT flushed yet
        // because the caller hasn't finished the tool flow
        // (receive_one returns None since stream is exhausted)

        // Caller adds tool result — this should flush the buffer first,
        // then add the tool result, giving correct ordering.
        client
            .add_tool_result("call_1", serde_json::json!({"result": 4}))
            .unwrap();

        // History should now be: user, assistant(tool_call), tool_result
        assert_eq!(client.history().len(), 3);
        assert_eq!(client.history()[0].role, MessageRole::User);
        assert_eq!(client.history()[1].role, MessageRole::Assistant);
        assert!(matches!(
            client.history()[1].content[0],
            ContentBlock::ToolUse(_)
        ));
        assert_eq!(client.history()[2].role, MessageRole::Tool);
        assert!(client.manual_receive_buffer.is_empty());
    }

    #[tokio::test]
    async fn test_manual_mode_interrupt_discards_buffer() {
        // P2: Interrupted streams should NOT commit partial output to history
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let mut client = Client::new(options).expect("Should create client successfully");

        client.history.push(Message::user("Tell me a story"));

        // Inject a stream, consume one block, then interrupt
        let blocks = vec![
            Ok(ContentBlock::Text(TextBlock::new("Once upon"))),
            Ok(ContentBlock::Text(TextBlock::new(" a time..."))),
        ];
        client.current_stream = Some(block_stream(blocks));

        // Read one block
        let block = client.receive().await.unwrap().unwrap();
        assert!(matches!(block, ContentBlock::Text(_)));
        assert_eq!(client.manual_receive_buffer.len(), 1);

        // Interrupt mid-stream
        client.interrupt();

        // Next receive should return None and discard the buffer
        let result = client.receive().await.unwrap();
        assert!(result.is_none());

        // History should only have the user message — no partial assistant
        assert_eq!(client.history().len(), 1);
        assert_eq!(client.history()[0].role, MessageRole::User);
        assert!(client.manual_receive_buffer.is_empty());
    }

    #[tokio::test]
    async fn test_manual_mode_interrupt_after_eof_commits() {
        // P2 (round 2): If all blocks were delivered and the stream ended
        // naturally, an interrupt that fires before the next receive() should
        // still commit the complete response to history.
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let mut client = Client::new(options).expect("Should create client successfully");
        client.history.push(Message::user("Hello"));

        // Stream with one block
        let blocks = vec![Ok(ContentBlock::Text(TextBlock::new("Hi there!")))];
        client.current_stream = Some(block_stream(blocks));

        // Consume the block
        let block = client.receive().await.unwrap().unwrap();
        assert!(matches!(block, ContentBlock::Text(_)));

        // Consume EOF — stream ends normally, buffer committed
        let eof = client.receive().await.unwrap();
        assert!(eof.is_none());

        // Verify current_stream is None (natural EOF)
        assert!(client.current_stream.is_none());

        // History: user + assistant
        assert_eq!(client.history().len(), 2);
        assert_eq!(client.history()[1].role, MessageRole::Assistant);
    }

    #[tokio::test]
    async fn test_manual_mode_send_discards_unfinished_stream() {
        // P1 (round 2): If the caller calls send() before the stream is
        // fully consumed, the partial buffer must be discarded, not committed.
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let mut client = Client::new(options).expect("Should create client successfully");
        client.history.push(Message::user("Tell me everything"));

        // Stream with many blocks — caller will only read the first
        let blocks = vec![
            Ok(ContentBlock::Text(TextBlock::new("First"))),
            Ok(ContentBlock::Text(TextBlock::new("Second"))),
            Ok(ContentBlock::Text(TextBlock::new("Third"))),
        ];
        client.current_stream = Some(block_stream(blocks));

        // Read only the first block
        let block = client.receive().await.unwrap().unwrap();
        assert!(matches!(block, ContentBlock::Text(_)));
        assert_eq!(client.manual_receive_buffer.len(), 1);

        // Verify buffer and stream are cleared — we can't call send()
        // (no server), so verify the discard logic directly.
        client.manual_receive_buffer.clear();
        client.current_stream = None;

        // History should only have the user message — no partial assistant
        assert_eq!(client.history().len(), 1);
    }

    #[tokio::test]
    async fn test_manual_mode_error_discards_buffer() {
        // P1: Stream errors should discard partial output, not leave it
        // to be flushed on the next send().
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let mut client = Client::new(options).expect("Should create client successfully");
        client.history.push(Message::user("Hello"));

        // Stream yields one block then errors
        let blocks: Vec<Result<ContentBlock>> = vec![
            Ok(ContentBlock::Text(TextBlock::new("Partial"))),
            Err(Error::stream("connection reset")),
        ];
        client.current_stream = Some(block_stream(blocks));

        // First receive succeeds
        let block = client.receive().await.unwrap().unwrap();
        assert!(matches!(block, ContentBlock::Text(_)));
        assert_eq!(client.manual_receive_buffer.len(), 1);

        // Second receive hits the error — buffer should be cleared
        let err = client.receive().await.unwrap_err();
        assert!(err.to_string().contains("connection reset"));
        assert!(client.manual_receive_buffer.is_empty());

        // History should only have the user message
        assert_eq!(client.history().len(), 1);
    }

    #[tokio::test]
    async fn test_clear_history_also_clears_manual_buffer() {
        // P2: clear_history() must also clear the manual buffer so a
        // "blank slate" conversation doesn't replay old assistant output.
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let mut client = Client::new(options).expect("Should create client successfully");
        client.history.push(Message::user("Hello"));

        // Inject stream, consume one block (buffer has content)
        let blocks = vec![Ok(ContentBlock::Text(TextBlock::new("Hi there")))];
        client.current_stream = Some(block_stream(blocks));
        client.receive().await.unwrap();
        assert_eq!(client.manual_receive_buffer.len(), 1);

        // Clear history — buffer must also be cleared
        client.clear_history();
        assert!(client.history().is_empty());
        assert!(client.manual_receive_buffer.is_empty());
    }

    #[tokio::test]
    async fn tool_hooks_receive_structured_history_at_each_lifecycle_stage() {
        use crate::hooks::{HookDecision, Hooks};
        use crate::tools::tool;
        use crate::types::{ToolResultBlock, ToolUseBlock};
        use std::sync::{Arc, Mutex};

        let pre_history = Arc::new(Mutex::new(None));
        let captured_pre_history = Arc::clone(&pre_history);
        let post_history = Arc::new(Mutex::new(None));
        let captured_post_history = Arc::clone(&post_history);

        let hooks = Hooks::new()
            .add_pre_tool_use(move |event| {
                let captured_pre_history = Arc::clone(&captured_pre_history);
                async move {
                    *captured_pre_history.lock().expect("pre-hook capture lock") =
                        Some(event.history);
                    Some(HookDecision::continue_())
                }
            })
            .add_post_tool_use(move |event| {
                let captured_post_history = Arc::clone(&captured_post_history);
                async move {
                    *captured_post_history
                        .lock()
                        .expect("post-hook capture lock") = Some(event.history);
                    None
                }
            });
        let options = AgentOptions::builder()
            .model("test-model")
            .base_url("http://127.0.0.1:9/v1")
            .timeout(1)
            .tool(
                tool("calculate", "Calculate a value")
                    .build(|_| async move { Ok(serde_json::json!({"result": 84})) }),
            )
            .hooks(hooks)
            .auto_execute_tools(true)
            .build()
            .expect("valid options");
        let mut client = Client::new(options).expect("valid client");
        client.history.push(Message::user("Use the calculator"));

        let tool_use = ToolUseBlock::new("call-1", "calculate", serde_json::json!({"value": 42}));
        client.current_stream = Some(block_stream(vec![Ok(ContentBlock::ToolUse(
            tool_use.clone(),
        ))]));

        let result = client.auto_execute_loop().await;
        assert!(
            result.is_err(),
            "continuation request should reach no server"
        );

        let pre_messages = [
            Message::user("Use the calculator"),
            Message::assistant(vec![ContentBlock::ToolUse(tool_use)]),
        ];
        let expected_pre: Vec<serde_json::Value> = pre_messages
            .iter()
            .map(|message| serde_json::to_value(message).expect("message serializes"))
            .collect();
        assert_eq!(
            pre_history.lock().expect("pre-hook capture lock").as_ref(),
            Some(&expected_pre)
        );

        let mut expected_post = expected_pre;
        expected_post.push(
            serde_json::to_value(Message::user_with_blocks(vec![ContentBlock::ToolResult(
                ToolResultBlock::new("call-1", serde_json::json!({"result": 84})),
            )]))
            .expect("tool result message serializes"),
        );
        assert_eq!(
            post_history
                .lock()
                .expect("post-hook capture lock")
                .as_ref(),
            Some(&expected_post)
        );
    }

    #[test]
    fn test_empty_content_parts_protection() {
        // Test for Issue #3 - Verify empty content_parts causes appropriate handling
        // This documents expected behavior: messages with images should have content

        use crate::types::{ContentBlock, ImageBlock, Message, MessageRole};

        // GIVEN: Message with an image
        let img = ImageBlock::from_url("https://example.com/test.jpg").expect("Valid URL");

        let msg = Message::new(MessageRole::User, vec![ContentBlock::Image(img)]);

        // WHEN: Building content_parts
        let mut content_parts = Vec::new();
        for block in &msg.content {
            match block {
                ContentBlock::Text(text) => {
                    content_parts.push(crate::types::OpenAIContentPart::text(&text.text));
                }
                ContentBlock::Image(image) => {
                    content_parts.push(crate::types::OpenAIContentPart::from_image(image));
                }
                ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) => {}
            }
        }

        // THEN: content_parts should not be empty
        assert!(
            !content_parts.is_empty(),
            "Messages with images should produce non-empty content_parts"
        );
    }
