//! Integration tests for the Open Agent SDK
//!
//! These tests verify that different modules work together correctly.

use open_agent::{AgentOptions, Client, ContentBlock, Message, MessageRole};

#[test]
fn test_agent_options_with_tools() {
    use open_agent::tool;

    let calculator = tool("add", "Add two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(serde_json::json!({"result": a + b}))
        });

    let options = AgentOptions::builder()
        .system_prompt("You are a calculator assistant")
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .tool(calculator)
        .build()
        .unwrap();

    assert_eq!(options.tools.len(), 1);
    assert_eq!(options.tools[0].name, "add");
}

#[test]
fn test_agent_options_with_hooks() {
    use open_agent::{HookDecision, Hooks, UserPromptSubmitEvent};

    let hooks = Hooks::new().add_user_prompt_submit(|event: UserPromptSubmitEvent| async move {
        if event.prompt.contains("forbidden") {
            Some(HookDecision::block("Forbidden word detected"))
        } else {
            Some(HookDecision::continue_())
        }
    });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .hooks(hooks)
        .build()
        .unwrap();

    // Just verify it builds successfully with hooks
    assert!(!options.system_prompt.is_empty());
}

#[test]
fn test_message_construction_flow() {
    let user_msg = Message::user("Hello");
    let system_msg = Message::system("You are helpful");
    let assistant_msg = Message::assistant(vec![ContentBlock::Text(open_agent::TextBlock::new(
        "Hi there!",
    ))]);

    assert!(matches!(user_msg.role, MessageRole::User));
    assert!(matches!(system_msg.role, MessageRole::System));
    assert!(matches!(assistant_msg.role, MessageRole::Assistant));
}

#[test]
fn test_client_creation_with_full_config() {
    let options = AgentOptions::builder()
        .system_prompt("Full test")
        .model("test-model")
        .base_url("http://localhost:1234/v1")
        .temperature(0.5)
        .max_turns(10)
        .build()
        .unwrap();

    let client = Client::new(options);
    assert_eq!(client.history().len(), 0);
}

#[test]
fn test_context_management_integration() {
    use open_agent::{estimate_tokens, is_approaching_limit, truncate_messages};

    let messages = vec![
        Message::system("System prompt"),
        Message::user("User message 1"),
        Message::assistant(vec![ContentBlock::Text(open_agent::TextBlock::new(
            "Response 1",
        ))]),
        Message::user("User message 2"),
    ];

    // Test token estimation
    let tokens = estimate_tokens(&messages);
    assert!(tokens > 0);

    // Test truncation
    let truncated = truncate_messages(&messages, 2, true);
    assert_eq!(truncated.len(), 3); // System + last 2

    // Test limit checking
    let approaching = is_approaching_limit(&messages, 1000, 0.8);
    assert!(!approaching); // Small message count shouldn't hit limit
}

#[test]
fn test_retry_config_integration() {
    use open_agent::retry::RetryConfig;
    use std::time::Duration;

    let config = RetryConfig::new()
        .with_max_attempts(5)
        .with_initial_delay(Duration::from_millis(100))
        .with_backoff_multiplier(1.5);

    assert_eq!(config.max_attempts, 5);
    assert_eq!(config.initial_delay, Duration::from_millis(100));
    assert_eq!(config.backoff_multiplier, 1.5);
}

#[tokio::test]
async fn test_tool_execution() {
    use open_agent::tool;

    let multiply = tool("multiply", "Multiply two numbers")
        .param("x", "number")
        .param("y", "number")
        .build(|args| async move {
            let x = args["x"].as_f64().unwrap_or(0.0);
            let y = args["y"].as_f64().unwrap_or(0.0);
            Ok(serde_json::json!({"result": x * y}))
        });

    let result = multiply
        .execute(serde_json::json!({"x": 5.0, "y": 3.0}))
        .await
        .unwrap();

    assert_eq!(result["result"], 15.0);
}

#[test]
fn test_error_handling_integration() {
    use open_agent::{Error, Result};

    fn operation_that_fails() -> Result<()> {
        Err(Error::timeout())
    }

    fn operation_that_succeeds() -> Result<i32> {
        Ok(42)
    }

    assert!(operation_that_fails().is_err());
    assert_eq!(operation_that_succeeds().unwrap(), 42);
}

#[test]
fn test_content_blocks() {
    use open_agent::{TextBlock, ToolUseBlock};

    let text = ContentBlock::Text(TextBlock::new("Hello"));
    let tool_use = ContentBlock::ToolUse(ToolUseBlock::new(
        "call_1",
        "tool_name",
        serde_json::json!({}),
    ));

    match text {
        ContentBlock::Text(t) => assert_eq!(t.text, "Hello"),
        _ => panic!("Expected TextBlock"),
    }

    match tool_use {
        ContentBlock::ToolUse(t) => assert_eq!(t.name, "tool_name"),
        _ => panic!("Expected ToolUseBlock"),
    }
}

#[test]
fn test_builder_pattern_chain() {
    // Test that builder pattern methods can be chained fluently
    let result = AgentOptions::builder()
        .system_prompt("Test")
        .model("model")
        .base_url("http://localhost")
        .temperature(0.7)
        .build();

    assert!(result.is_ok());
    let options = result.unwrap();
    assert_eq!(options.temperature, 0.7);
}
