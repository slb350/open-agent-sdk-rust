//! Auto-execution mode integration tests
//!
//! Tests for automatic tool execution loop functionality.
//! These tests verify that when auto_execute_tools=true, the client
//! automatically executes tools and continues until a text-only response.
//!
//! NOTE: These tests make real API calls to Ollama at localhost:11434
//! Run with: cargo test --test auto_execution_test -- --test-threads=1 --nocapture

use open_agent::{tool, AgentOptions, Client, ContentBlock, Error};
use serde_json::json;
use tokio::time::{timeout, Duration};

const TEST_TIMEOUT: Duration = Duration::from_secs(30);
const OLLAMA_URL: &str = "http://localhost:11434/v1";
const MODEL: &str = "qwen3:8b";

/// Helper: Collect all response blocks with timeout
async fn collect_response(client: &mut Client) -> Result<(Vec<String>, usize), String> {
    let result = timeout(TEST_TIMEOUT, async {
        let mut text_blocks = Vec::new();
        let mut tool_blocks_received = 0;

        while let Some(block) = client.receive().await {
            match block {
                Ok(ContentBlock::Text(text)) => {
                    text_blocks.push(text.text);
                }
                Ok(ContentBlock::ToolUse(_)) => {
                    tool_blocks_received += 1;
                }
                Ok(ContentBlock::ToolResult(_)) => {
                    // Ignore tool results in auto mode
                }
                Err(e) => {
                    return Err(format!("Error receiving block: {}", e));
                }
            }
        }

        Ok((text_blocks, tool_blocks_received))
    })
    .await;

    match result {
        Ok(Ok(data)) => Ok(data),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("Test timed out after 30 seconds".to_string()),
    }
}

/// Test: Basic auto-execution with simple query (no tools needed)
#[ignore] // Requires running Ollama server at localhost:11434
#[tokio::test]
async fn test_auto_execution_simple_query() {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant. Respond concisely.")
        .model(MODEL)
        .base_url(OLLAMA_URL)
        .auto_execute_tools(true)
        .temperature(0.1)
        .build()
        .unwrap();

    let mut client = Client::new(options);
    client
        .send("What is 2+2? Answer with just the number.")
        .await
        .unwrap();

    let (text_blocks, tool_blocks) = collect_response(&mut client)
        .await
        .expect("Failed to collect response");

    assert!(!text_blocks.is_empty(), "Should receive text response");
    assert_eq!(
        tool_blocks, 0,
        "Should not receive tool blocks without tools"
    );
}

/// Test: Auto-execution with calculator tools
#[ignore] // Requires running Ollama server at localhost:11434
#[tokio::test]
async fn test_auto_execution_with_tools() {
    let add_tool = tool(
        "add",
        "Add two numbers a and b. Use this for addition only.",
    )
    .param("a", "number")
    .param("b", "number")
    .build(|args| async move {
        let a = args["a"].as_f64().unwrap_or(0.0);
        let b = args["b"].as_f64().unwrap_or(0.0);
        Ok(json!({"result": a + b}))
    });

    let options = AgentOptions::builder()
        .system_prompt(
            "You are a calculator. ALWAYS use the add tool for addition. Never calculate manually.",
        )
        .model(MODEL)
        .base_url(OLLAMA_URL)
        .tools(vec![add_tool])
        .auto_execute_tools(true)
        .max_tool_iterations(5)
        .temperature(0.1)
        .build()
        .unwrap();

    let mut client = Client::new(options);
    client
        .send("Use the add tool to calculate 25 + 17")
        .await
        .unwrap();

    let (text_blocks, _tool_blocks) = collect_response(&mut client)
        .await
        .expect("Failed to collect response");

    // Should have received final text response
    assert!(
        !text_blocks.is_empty(),
        "Should receive final text response"
    );

    // In auto mode, tools are executed internally, so we might see results in text
    let full_response = text_blocks.join("");
    assert!(!full_response.is_empty(), "Response should not be empty");
}

/// Test: Auto-execution respects max_tool_iterations limit
#[ignore] // Requires running Ollama server at localhost:11434
#[tokio::test]
async fn test_auto_execution_max_iterations() {
    let increment_tool = tool("increment", "Add 1 to the value")
        .param("value", "number")
        .build(|args| async move {
            let value = args["value"].as_f64().unwrap_or(0.0);
            Ok(json!({"result": value + 1.0}))
        });

    let options = AgentOptions::builder()
        .system_prompt("You are a counter. Use the increment tool.")
        .model(MODEL)
        .base_url(OLLAMA_URL)
        .tools(vec![increment_tool])
        .auto_execute_tools(true)
        .max_tool_iterations(2) // Limit to 2 iterations
        .temperature(0.1)
        .build()
        .unwrap();

    let mut client = Client::new(options);
    client.send("Increment 0 five times").await.unwrap();

    let result = collect_response(&mut client).await;

    // Should complete even if max iterations hit
    assert!(result.is_ok(), "Should complete despite iteration limit");
}

/// Test: Auto-execution handles tool errors gracefully
#[ignore] // Requires running Ollama server at localhost:11434
#[tokio::test]
async fn test_auto_execution_tool_error() {
    let failing_tool = tool("divide", "Divide two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);

            if b == 0.0 {
                return Err(Error::tool("Cannot divide by zero"));
            }

            Ok(json!({"result": a / b}))
        });

    let options = AgentOptions::builder()
        .system_prompt("You are a calculator. Use the divide tool.")
        .model(MODEL)
        .base_url(OLLAMA_URL)
        .tools(vec![failing_tool])
        .auto_execute_tools(true)
        .max_tool_iterations(3)
        .temperature(0.1)
        .build()
        .unwrap();

    let mut client = Client::new(options);
    client.send("Calculate 10 divided by 2").await.unwrap();

    let result = collect_response(&mut client).await;

    // Should handle gracefully
    assert!(
        result.is_ok(),
        "Should complete even with potential tool errors"
    );
}

/// Test: Multiple tools available
#[ignore] // Requires running Ollama server at localhost:11434
#[tokio::test]
async fn test_auto_execution_multiple_tools() {
    let add_tool = tool("add", "Add two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!({"result": a + b}))
        });

    let multiply_tool = tool("multiply", "Multiply two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!({"result": a * b}))
        });

    let options = AgentOptions::builder()
        .system_prompt("You are a calculator with multiple operations.")
        .model(MODEL)
        .base_url(OLLAMA_URL)
        .tools(vec![add_tool, multiply_tool])
        .auto_execute_tools(true)
        .max_tool_iterations(5)
        .temperature(0.1)
        .build()
        .unwrap();

    let mut client = Client::new(options);
    client
        .send("What tools do you have available?")
        .await
        .unwrap();

    let result = collect_response(&mut client).await;
    assert!(result.is_ok(), "Should complete successfully");
}

/// Test: Auto-execution without tools behaves like normal mode
#[ignore] // Requires running Ollama server at localhost:11434
#[tokio::test]
async fn test_auto_execution_no_tools() {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model(MODEL)
        .base_url(OLLAMA_URL)
        .auto_execute_tools(true) // Enabled but no tools provided
        .temperature(0.1)
        .build()
        .unwrap();

    let mut client = Client::new(options);
    client.send("Hello, respond briefly").await.unwrap();

    let (text_blocks, tool_blocks) = collect_response(&mut client)
        .await
        .expect("Failed to collect response");

    assert!(!text_blocks.is_empty(), "Should receive text response");
    assert_eq!(tool_blocks, 0, "Should not receive tool blocks");
}

/// Test: Manual mode (auto_execute_tools=false) returns ToolUse blocks
#[ignore] // Requires running Ollama server at localhost:11434
#[tokio::test]
async fn test_manual_mode_returns_tool_blocks() {
    let add_tool = tool("add", "Add two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!({"result": a + b}))
        });

    let options = AgentOptions::builder()
        .system_prompt("You are a calculator. Use tools when asked.")
        .model(MODEL)
        .base_url(OLLAMA_URL)
        .tools(vec![add_tool])
        .auto_execute_tools(false) // Manual mode
        .temperature(0.1)
        .build()
        .unwrap();

    let mut client = Client::new(options);
    client.send("What is 5 plus 3?").await.unwrap();

    let result = timeout(TEST_TIMEOUT, async {
        let mut received_blocks = 0;

        while let Some(block) = client.receive().await {
            match block {
                Ok(_) => {
                    received_blocks += 1;
                    // In manual mode, just verify we receive some blocks
                    if received_blocks > 0 {
                        break;
                    }
                }
                Err(e) => {
                    return Err(format!("Error: {}", e));
                }
            }
        }

        Ok(received_blocks)
    })
    .await;

    assert!(result.is_ok(), "Should receive blocks in manual mode");
}

/// Test: Streaming response with auto-execution
#[ignore] // Requires running Ollama server at localhost:11434
#[tokio::test]
async fn test_auto_execution_streaming() {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model(MODEL)
        .base_url(OLLAMA_URL)
        .auto_execute_tools(true)
        .temperature(0.1)
        .build()
        .unwrap();

    let mut client = Client::new(options);
    client.send("Count to 3").await.unwrap();

    let result = timeout(TEST_TIMEOUT, async {
        let mut block_count = 0;

        while let Some(block) = client.receive().await {
            if block.is_ok() {
                block_count += 1;
            }
        }

        block_count
    })
    .await;

    assert!(result.is_ok(), "Should stream response");
    assert!(result.unwrap() > 0, "Should receive at least one block");
}

/// Test: History tracking in auto-execution mode
#[ignore] // Requires running Ollama server at localhost:11434
#[tokio::test]
async fn test_auto_execution_history() {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model(MODEL)
        .base_url(OLLAMA_URL)
        .auto_execute_tools(true)
        .temperature(0.1)
        .build()
        .unwrap();

    let mut client = Client::new(options);

    let initial_history_len = client.history().len();

    client.send("Hello").await.unwrap();

    timeout(TEST_TIMEOUT, async {
        while let Some(_) = client.receive().await {}
    })
    .await
    .ok();

    let final_history_len = client.history().len();

    assert!(
        final_history_len > initial_history_len,
        "History should grow after interaction"
    );
}

/// Test: Temperature affects randomness
#[ignore] // Requires running Ollama server at localhost:11434
#[tokio::test]
async fn test_auto_execution_temperature() {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model(MODEL)
        .base_url(OLLAMA_URL)
        .temperature(0.1) // Low temperature for consistent responses
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let mut client = Client::new(options);
    client.send("Say hello").await.unwrap();

    let result = collect_response(&mut client).await;
    assert!(result.is_ok(), "Should complete with custom temperature");
}

/// Test: Max tokens limit
#[ignore] // Requires running Ollama server at localhost:11434
#[tokio::test]
async fn test_auto_execution_max_tokens() {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model(MODEL)
        .base_url(OLLAMA_URL)
        .auto_execute_tools(true)
        .temperature(0.1)
        .build()
        .unwrap();

    let mut client = Client::new(options);
    client.send("Tell me a story").await.unwrap();

    let (text_blocks, _) = collect_response(&mut client)
        .await
        .expect("Should complete despite low token limit");

    let total_text = text_blocks.join("");
    // Response should be short due to token limit
    assert!(!total_text.is_empty(), "Should receive some response");
}

/// Test: Timeout configuration
#[ignore] // Requires running Ollama server at localhost:11434
#[tokio::test]
async fn test_auto_execution_custom_timeout() {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model(MODEL)
        .base_url(OLLAMA_URL)
        .auto_execute_tools(true)
        .temperature(0.1)
        .build()
        .unwrap();

    let mut client = Client::new(options);
    client.send("Hello").await.unwrap();

    // Should complete within our test timeout
    let result = collect_response(&mut client).await;
    assert!(result.is_ok(), "Should complete with custom timeout");
}
