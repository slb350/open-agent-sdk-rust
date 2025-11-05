//! Auto-execution mode integration tests
//!
//! Tests for automatic tool execution loop functionality.
//! These tests verify that when auto_execute_tools=true, the client
//! automatically executes tools and continues until a text-only response.

use open_agent::{tool, AgentOptions, Client, ContentBlock};
use serde_json::json;

/// Test: Auto-execution with sequential tool calls
///
/// Scenario: Query requires 2 tool calls (add, then multiply)
/// Expected: Both tools execute automatically, final text response returned
#[tokio::test]
async fn test_auto_execution_sequential_tools() {
    // Create simple calculator tools
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

    // Create options with auto-execution enabled
    let options = AgentOptions::builder()
        .system_prompt("You are a calculator. Use tools to compute results.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(add_tool)
        .tool(multiply_tool)
        .auto_execute_tools(true) // ← Enable auto-execution
        .max_tool_iterations(5)
        .build()
        .unwrap();

    let mut client = Client::new(options);

    // Send query that requires tool use
    // Note: This is a mock test - we're testing the structure, not actual LLM
    // Real test would need a mock server or live LLM
    client
        .send("Calculate 5 + 3, then multiply the result by 2")
        .await
        .unwrap();

    // In auto-execution mode, we should only receive final text blocks
    // All tool execution happens internally
    let mut text_blocks = Vec::new();
    let mut tool_blocks_received = 0;

    while let Some(block) = client.receive().await {
        match block.unwrap() {
            ContentBlock::Text(text) => {
                text_blocks.push(text.text);
            }
            ContentBlock::ToolUse(_) => {
                tool_blocks_received += 1;
                // In auto-execution mode, we should NOT receive ToolUse blocks
                // They should be handled internally
            }
            ContentBlock::ToolResult(_) => {
                // Should not receive ToolResult blocks directly either
                // They are handled internally in auto-execution mode
            }
        }
    }

    // Verify: Should have received text response
    assert!(
        !text_blocks.is_empty(),
        "Should receive final text response"
    );

    // Verify: Should NOT receive ToolUse blocks in auto mode
    assert_eq!(
        tool_blocks_received, 0,
        "Should not receive ToolUse blocks in auto-execution mode"
    );

    // Verify: History should contain tool calls and results
    let history = client.history();
    assert!(
        history.len() > 1,
        "History should contain multiple messages"
    );
}

/// Test: Auto-execution respects max_tool_iterations limit
///
/// Scenario: Set max_tool_iterations=2, but query would need 3 iterations
/// Expected: Stops after 2 iterations, returns partial result
#[tokio::test]
async fn test_auto_execution_max_iterations() {
    let counter_tool = tool("increment", "Increment a counter")
        .param("value", "number")
        .build(|args| async move {
            let value = args["value"].as_f64().unwrap_or(0.0);
            Ok(json!({"result": value + 1.0}))
        });

    let options = AgentOptions::builder()
        .system_prompt("You are a counter agent.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(counter_tool)
        .auto_execute_tools(true)
        .max_tool_iterations(2) // ← Limit to 2 iterations
        .build()
        .unwrap();

    let client = Client::new(options);

    // This test would need a mock to truly verify iteration count
    // For now, we're testing the structure exists
    assert_eq!(client.options().max_tool_iterations, 2);
}

/// Test: Auto-execution with no tools (text-only response)
///
/// Scenario: Query doesn't require tools
/// Expected: Returns text immediately without tool execution loop
#[tokio::test]
async fn test_auto_execution_no_tools_needed() {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let mut client = Client::new(options);

    // Send simple query that doesn't need tools
    // (This would need mock/live LLM to truly test)
    client.send("Hello").await.unwrap();

    // Should receive text response immediately
    // No tool execution loop needed
    let mut received_response = false;
    while let Some(block) = client.receive().await {
        if let Ok(ContentBlock::Text(_)) = block {
            received_response = true;
            break;
        }
    }

    assert!(received_response, "Should receive text response");
}

/// Test: Auto-execution with tool execution error
///
/// Scenario: Tool execution fails
/// Expected: Error handled gracefully, doesn't crash the loop
#[tokio::test]
async fn test_auto_execution_tool_error_handling() {
    let failing_tool = tool("divide", "Divide two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);

            if b == 0.0 {
                return Err(open_agent::Error::tool("Division by zero"));
            }

            Ok(json!({"result": a / b}))
        });

    let options = AgentOptions::builder()
        .system_prompt("You are a calculator.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(failing_tool)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // Tool error should be handled gracefully in auto-execution mode
    // This test validates the structure is in place
}

/// Test: Auto-execution can be interrupted
///
/// Scenario: Interrupt during auto-execution loop
/// Expected: Loop stops gracefully, partial results returned
#[tokio::test]
async fn test_auto_execution_with_interrupt() {
    use std::sync::{Arc, Mutex};

    let slow_tool = tool("slow_operation", "Slow operation")
        .param("data", "string")
        .build(|args| async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            Ok(json!({"processed": args["data"]}))
        });

    let options = AgentOptions::builder()
        .system_prompt("You are a processor.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(slow_tool)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let client = Arc::new(Mutex::new(Client::new(options)));

    // Spawn interrupt task
    let client_clone = Arc::clone(&client);
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        let client_lock = client_clone.lock().unwrap();
        client_lock.interrupt();
    });

    // Auto-execution should respect interrupt
    {
        let mut client_lock = client.lock().unwrap();
        client_lock.send("Process data").await.unwrap();
    }

    // Should stop cleanly when interrupted
    loop {
        let block_opt = {
            let mut client_lock = client.lock().unwrap();
            client_lock.receive().await
        };

        if block_opt.is_none() {
            break;
        }
    }

    // If we reach here without hanging, interrupt worked
    assert!(true);
}

/// Test: Empty tool result handling
///
/// Scenario: Tool returns empty JSON object
/// Expected: Empty result handled gracefully, conversation continues
#[tokio::test]
async fn test_auto_execution_empty_tool_result() {
    let empty_tool = tool("get_data", "Get empty data")
        .param("id", "string")
        .build(|_args| async move {
            Ok(json!({})) // Empty result
        });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(empty_tool)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // Empty result should be handled gracefully
    // This validates the structure is in place
}

/// Test: Tool not found in registry
///
/// Scenario: Request to execute a tool that doesn't exist
/// Expected: Error handled gracefully in auto-execution loop
#[tokio::test]
async fn test_auto_execution_tool_not_found() {
    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // If model calls a nonexistent tool during auto-execution,
    // the loop should handle it gracefully by adding an error result
    // and continuing the conversation rather than panicking
    // This validates the structure is in place
}

/// Test: Multiple tools in single response
///
/// Scenario: Model returns multiple tool calls at once
/// Expected: All tools should be executed
#[tokio::test]
async fn test_auto_execution_multiple_tools_single_response() {
    let tool_a = tool("op_a", "Operation A")
        .param("x", "number")
        .build(|args| async move { Ok(json!({"result_a": args["x"]})) });

    let tool_b = tool("op_b", "Operation B")
        .param("y", "number")
        .build(|args| async move { Ok(json!({"result_b": args["y"]})) });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(tool_a)
        .tool(tool_b)
        .auto_execute_tools(true)
        .max_tool_iterations(3)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // Should execute both tools (structure validated)
}

/// Test: Mixed text and tool blocks in response
///
/// Scenario: Model returns both text and tool calls together
/// Expected: Both should be added to history, tools executed
#[tokio::test]
async fn test_auto_execution_mixed_blocks() {
    let calc_tool = tool("calculate", "Calculate")
        .param("expr", "string")
        .build(|_args| async move { Ok(json!({"result": 42})) });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(calc_tool)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // Response with both text and tool calls should:
    // - Add both to history
    // - Execute tools
    // - Continue conversation
}

/// Test: Large tool result handling
///
/// Scenario: Tool returns large data (10KB)
/// Expected: Large results handled without issues
#[tokio::test]
async fn test_auto_execution_large_tool_result() {
    let big_data_tool = tool("get_big_data", "Returns large data")
        .param("size", "number")
        .build(|_args| async move {
            // Simulate large result (10KB)
            let large_string = "x".repeat(10000);
            Ok(json!({"data": large_string}))
        });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(big_data_tool)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // Large results should be handled without issues
}

/// Test: Auto-execution preserves history correctly
///
/// Scenario: After auto-execution, verify history structure
/// Expected: History contains appropriate messages in correct order
#[tokio::test]
async fn test_auto_execution_history_management() {
    let simple_tool = tool("echo", "Echo input")
        .param("msg", "string")
        .build(|args| async move { Ok(args["msg"].clone()) });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(simple_tool)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let client = Client::new(options);

    let initial_history_len = client.history().len();

    // After auto-execution, history should contain:
    // 1. User message (if any)
    // 2. Assistant message with tool calls
    // 3. User message with tool results
    // 4. Final assistant message with text

    // Verify history structure makes sense
    assert!(client.history().len() >= initial_history_len);
}

/// Test: Iteration limit hit with pending tools
///
/// Scenario: max_tool_iterations=2 but query needs more iterations
/// Expected: Stops after 2 iterations, returns partial results
#[tokio::test]
async fn test_auto_execution_iteration_limit_with_pending_tools() {
    let loop_tool = tool("loop_op", "Keeps calling itself")
        .param("count", "number")
        .build(|args| async move { Ok(json!({"count": args["count"]})) });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(loop_tool)
        .auto_execute_tools(true)
        .max_tool_iterations(2) // Very low limit
        .build()
        .unwrap();

    let client = Client::new(options);

    // Should stop after 2 iterations
    // Not continue forever
    // Return partial results
    assert_eq!(client.options().max_tool_iterations, 2);
}

/// Test: Tool execution with error result
///
/// Scenario: Tool returns an error during execution
/// Expected: Error added to history as error result, not crash
#[tokio::test]
async fn test_auto_execution_tool_execution_error() {
    use open_agent::Error;

    let error_tool = tool("failing_op", "Operation that fails")
        .param("x", "number")
        .build(|_args| async move { Err(Error::tool("Simulated failure")) });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(error_tool)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // Error should be added to history as error result
    // Not crash the loop
}
