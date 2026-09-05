fn test_options() -> AgentOptions {
    AgentOptions::builder()
        .system_prompt("Test")
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .unwrap()
}

fn block_stream(items: Vec<Result<ContentBlock>>) -> EventStream {
    Box::pin(futures::stream::iter(
        items
            .into_iter()
            .map(|item| item.map(StreamEvent::Block)),
    ))
}

#[test]
fn test_client_creation() {
    let options = test_options();

    let client = Client::new(options).expect("Should create client successfully");
    assert_eq!(client.history().len(), 0);
}

#[test]
fn test_interrupt_idempotent() {
    let options = test_options();

    let client = Client::new(options).expect("Should create client successfully");
    assert!(!client.interrupted.load(Ordering::SeqCst));
    client.interrupt();
    assert!(client.interrupted.load(Ordering::SeqCst));

    client.interrupt();
    assert!(client.interrupted.load(Ordering::SeqCst));
}

#[tokio::test]
async fn test_manual_mode_adds_assistant_to_history() {
    let options = test_options();

    let mut client = Client::new(options).expect("Should create client successfully");

    client
        .history
        .push(Message::user("What's the capital of France?"));

    let blocks = vec![
        Ok(ContentBlock::Text(TextBlock::new("Paris is"))),
        Ok(ContentBlock::Text(TextBlock::new(
            " the capital of France.",
        ))),
    ];
    client.current_stream = Some(block_stream(blocks));

    let mut received = Vec::new();
    while let Some(block) = client.receive().await.unwrap() {
        received.push(block);
    }

    assert_eq!(received.len(), 2);

    assert_eq!(client.history().len(), 2);
    assert_eq!(client.history()[0].role, MessageRole::User);
    assert_eq!(client.history()[1].role, MessageRole::Assistant);

    assert_eq!(client.history()[1].content.len(), 1);
    match &client.history()[1].content[0] {
        ContentBlock::Text(text) => {
            assert_eq!(text.text, "Paris is the capital of France.");
        }
        other => panic!("expected one joined text block, got {other:?}"),
    }
}

#[tokio::test]
async fn test_manual_mode_empty_stream_no_assistant_message() {
    let options = test_options();

    let mut client = Client::new(options).expect("Should create client successfully");

    let result = client.receive().await.unwrap();
    assert!(result.is_none());

    assert_eq!(client.history().len(), 0);
}

#[tokio::test]
async fn add_tool_result_flushes_pending_assistant_before_result() {
    let options = test_options();

    let mut client = Client::new(options).expect("Should create client successfully");

    client.history.push(Message::user("Calculate 2+2"));

    let tool_use =
        crate::types::ToolUseBlock::new("call_1", "calculator", serde_json::json!({"a": 2}));
    let blocks = vec![Ok(ContentBlock::ToolUse(tool_use))];
    client.current_stream = Some(block_stream(blocks));

    let block = client.receive().await.unwrap().unwrap();
    assert!(matches!(block, ContentBlock::ToolUse(_)));

    assert_eq!(client.manual_receive_buffer.len(), 1);
    assert_eq!(client.history().len(), 1); // only user message

    client
        .add_tool_result("call_1", serde_json::json!({"result": 4}))
        .unwrap();

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
async fn cancellation_between_stream_events_discards_partial_history_in_both_modes() {
    for (auto, cancel_on_text) in [(false, false), (false, true), (true, false), (true, true)] {
        let options = AgentOptions::builder()
            .model("m")
            .base_url("http://localhost:1234/v1")
            .auto_execute_tools(auto)
            .build()
            .unwrap();
        let mut client = Client::new(options).unwrap();
        client.history.push(Message::user("Tell me a story"));
        let interrupted = client.interrupt_handle();
        client.current_stream = Some(Box::pin(
            futures::stream::iter(vec![
                Ok(StreamEvent::Block(ContentBlock::Text(TextBlock::new(
                    "Partial",
                )))),
                Ok(StreamEvent::Reasoning("thinking".into())),
                Ok(StreamEvent::Block(ContentBlock::Text(TextBlock::new(
                    "Must not arrive",
                )))),
            ])
            .inspect(move |event| {
                let cancel = if cancel_on_text {
                    matches!(event, Ok(StreamEvent::Block(ContentBlock::Text(text))) if text.text == "Must not arrive")
                } else {
                    matches!(event, Ok(StreamEvent::Reasoning(_)))
                };
                if cancel {
                    interrupted.store(true, Ordering::SeqCst);
                }
            }),
        ));
        if !auto {
            assert!(client.receive().await.unwrap().is_some());
        }
        assert!(
            client.receive().await.unwrap().is_none(),
            "auto={auto}, cancel_on_text={cancel_on_text}"
        );
        assert!(client.receive().await.unwrap().is_none());
        assert_eq!(client.history().len(), 1);
    }
}

#[tokio::test]
async fn test_manual_mode_interrupt_after_eof_commits() {
    let options = test_options();

    let mut client = Client::new(options).expect("Should create client successfully");
    client.history.push(Message::user("Hello"));

    let blocks = vec![Ok(ContentBlock::Text(TextBlock::new("Hi there!")))];
    client.current_stream = Some(block_stream(blocks));

    let block = client.receive().await.unwrap().unwrap();
    assert!(matches!(block, ContentBlock::Text(_)));

    let eof = client.receive().await.unwrap();
    assert!(eof.is_none());

    assert!(client.current_stream.is_none());
    client.interrupt();
    assert!(client.receive().await.unwrap().is_none());

    assert_eq!(client.history().len(), 2);
    assert_eq!(client.history()[1].role, MessageRole::Assistant);
}

#[tokio::test]
async fn stream_errors_discard_partial_history_and_close_both_client_modes() {
    for auto in [false, true] {
        let options = AgentOptions::builder()
            .model("m")
            .base_url("http://localhost:1234/v1")
            .auto_execute_tools(auto)
            .build()
            .unwrap();
        let mut client = Client::new(options).unwrap();
        client.history.push(Message::user("Hello"));
        client.current_stream = Some(block_stream(vec![
            Ok(ContentBlock::Text(TextBlock::new("Partial"))),
            Err(Error::stream("connection reset")),
            Ok(ContentBlock::Text(TextBlock::new("Untrustworthy suffix"))),
        ]));
        if !auto {
            assert!(client.receive().await.unwrap().is_some());
        }
        assert!(
            client
                .receive()
                .await
                .unwrap_err()
                .to_string()
                .contains("connection reset")
        );
        assert!(client.receive().await.unwrap().is_none(), "auto={auto}");
        assert_eq!(client.history().len(), 1, "auto={auto}");
    }
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
                *captured_pre_history.lock().expect("pre-hook capture lock") = Some(event.history);
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
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/v1/chat/completions"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string("data: [DONE]\n\n"),
        )
        .expect(1)
        .mount(&server)
        .await;
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url(format!("{}/v1", server.uri()))
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

    assert!(client.auto_execute_loop().await.unwrap().is_empty());

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
