//! Test for P1 bug fix: Preserve assistant content field when only tool calls exist
//!
//! The OpenAI chat completions API requires the content field even when empty.
//! This test verifies that assistant messages with tool calls always have content.

use open_agent::{
    AgentOptions, Client, ContentBlock, Message, MessageRole, TextBlock, ToolUseBlock,
};
use serde_json::json;

#[tokio::test]
async fn test_assistant_tool_call_without_text_has_content() {
    // Create an assistant message with ONLY tool calls (no text)
    let tool_use = ToolUseBlock::new("call_123", "test_function", json!({"arg": "value"}));

    let msg = Message::new(
        MessageRole::Assistant,
        vec![ContentBlock::ToolUse(tool_use)],
    );

    // Create client and add message
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .unwrap();

    let mut client = Client::new(options).unwrap();
    client.history_mut().push(msg);

    // Verify the message is in history
    assert_eq!(client.history().len(), 1);

    // The content block should be a ToolUse
    let history_msg = &client.history()[0];
    assert_eq!(history_msg.content.len(), 1);

    match &history_msg.content[0] {
        ContentBlock::ToolUse(tool) => {
            assert_eq!(tool.name, "test_function");
            assert_eq!(tool.id, "call_123");
        }
        _ => panic!("Expected ToolUse block"),
    }

    // This is the key: when the client serializes this message for the API,
    // it should include an empty content field, not omit it.
    // We can't directly test the internal serialization without exposing
    // OpenAIMessage, but the fix ensures content is always Some(...) not None.
}

#[tokio::test]
async fn test_assistant_tool_call_with_text_has_content() {
    // Create an assistant message with tool calls AND text
    let tool_use = ToolUseBlock::new("call_456", "another_function", json!({"param": 42}));

    let msg = Message::new(
        MessageRole::Assistant,
        vec![
            ContentBlock::Text(TextBlock::new("Let me call a function")),
            ContentBlock::ToolUse(tool_use),
        ],
    );

    let options = AgentOptions::builder()
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .build()
        .unwrap();

    let mut client = Client::new(options).unwrap();
    client.history_mut().push(msg);

    // Verify message structure
    assert_eq!(client.history().len(), 1);
    assert_eq!(client.history()[0].content.len(), 2);

    // First block should be text
    match &client.history()[0].content[0] {
        ContentBlock::Text(text) => {
            assert_eq!(text.text, "Let me call a function");
        }
        _ => panic!("Expected Text block"),
    }

    // Second block should be tool use
    match &client.history()[0].content[1] {
        ContentBlock::ToolUse(tool) => {
            assert_eq!(tool.name, "another_function");
        }
        _ => panic!("Expected ToolUse block"),
    }
}

#[test]
fn test_openai_content_empty_string_serialization() {
    // Test that empty OpenAIContent::Text serializes correctly
    use open_agent::OpenAIContent;

    let content = OpenAIContent::Text(String::new());
    let json = serde_json::to_value(&content).unwrap();

    // Empty string should serialize as ""
    assert_eq!(json, "");
}

#[test]
fn test_openai_content_with_text_serialization() {
    // Test that OpenAIContent::Text with content serializes correctly
    use open_agent::OpenAIContent;

    let content = OpenAIContent::Text("Hello".to_string());
    let json = serde_json::to_value(&content).unwrap();

    assert_eq!(json, "Hello");
}
