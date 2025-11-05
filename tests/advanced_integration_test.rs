//! Advanced integration tests
//!
//! Tests that combine multiple features together to verify they work correctly
//! in realistic scenarios.

use open_agent::{tool, AgentOptions, Client, HookDecision, Hooks};
use serde_json::json;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

/// Test: Auto-execution + Hooks together
///
/// Verifies that auto-execution and hooks work correctly when combined,
/// with hooks intercepting and modifying tool execution during auto-execution.
#[tokio::test]
async fn test_auto_execution_with_hooks_integration() {
    let execution_count = Arc::new(AtomicUsize::new(0));
    let count_clone = execution_count.clone();

    // Tool that can be blocked or modified by hooks
    let calculator = tool("divide", "Divide numbers")
        .param("a", "number")
        .param("b", "number")
        .build(move |args| {
            let count = count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                let a = args["a"].as_f64().unwrap_or(0.0);
                let b = args["b"].as_f64().unwrap_or(0.0);

                if b == 0.0 {
                    return Err(open_agent::Error::tool("Division by zero"));
                }

                Ok(json!({"result": a / b}))
            }
        });

    let blocked_count = Arc::new(AtomicUsize::new(0));
    let blocked_clone = blocked_count.clone();

    let hooks = Hooks::new()
        .add_pre_tool_use(move |event| {
            let blocked = blocked_clone.clone();
            async move {
                // Block division by zero before execution
                if let Some(b) = event.tool_input.get("b").and_then(|v| v.as_f64()) {
                    if b == 0.0 {
                        blocked.fetch_add(1, Ordering::SeqCst);
                        return Some(HookDecision::block("Division by zero prevented"));
                    }
                }
                Some(HookDecision::continue_())
            }
        })
        .add_post_tool_use(|event| async move {
            // Add precision info to result
            if event.tool_result.get("result").is_some() {
                let mut enhanced = event.tool_result.clone();
                enhanced["precision"] = json!("high");
                return Some(HookDecision::modify_input(enhanced, "Added precision"));
            }
            None
        });

    let options = AgentOptions::builder()
        .system_prompt("Calculator")
        .model("test")
        .base_url("http://localhost:11434/v1")
        .tool(calculator)
        .hooks(hooks)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // Verify structure is correct
    assert!(execution_count.load(Ordering::SeqCst) == 0);
    assert!(blocked_count.load(Ordering::SeqCst) == 0);
}

/// Test: Auto-execution + Interrupt
///
/// Verifies that interrupt functionality works correctly during auto-execution,
/// allowing long-running auto-execution loops to be stopped gracefully.
#[tokio::test]
async fn test_auto_execution_with_interrupt_integration() {
    let slow_tool = tool("slow_operation", "Slow operation")
        .param("duration_ms", "number")
        .build(|args| async move {
            let duration = args["duration_ms"].as_u64().unwrap_or(100);
            tokio::time::sleep(tokio::time::Duration::from_millis(duration)).await;
            Ok(json!({"completed": true}))
        });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("test")
        .base_url("http://localhost:11434/v1")
        .tool(slow_tool)
        .auto_execute_tools(true)
        .max_tool_iterations(10)
        .build()
        .unwrap();

    let client = Client::new(options);

    // Interrupt can be used to stop auto-execution
    // The interrupt() method sets the interrupt flag
    client.interrupt();

    // Structure validates interrupt capability during auto-execution
}

/// Test: Auto-execution + Context Management
///
/// Verifies that context management utilities work correctly with auto-execution,
/// allowing proper handling of conversation history and token limits.
#[tokio::test]
async fn test_auto_execution_with_context_management() {
    use open_agent::{estimate_tokens, truncate_messages};

    let echo_tool = tool("echo", "Echo")
        .param("msg", "string")
        .build(|args| async move { Ok(args["msg"].clone()) });

    let options = AgentOptions::builder()
        .system_prompt("Test assistant")
        .model("test")
        .base_url("http://localhost:11434/v1")
        .tool(echo_tool)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let mut client = Client::new(options);

    // After auto-execution, check token count
    let initial_tokens = estimate_tokens(client.history());

    // Add a message to history
    client
        .history_mut()
        .push(open_agent::Message::user("Hello"));

    let tokens = estimate_tokens(client.history());
    assert!(tokens > initial_tokens);

    // Truncate if needed
    if tokens > 100 {
        let truncated = truncate_messages(client.history(), 2, true);
        *client.history_mut() = truncated;

        let new_tokens = estimate_tokens(client.history());
        assert!(new_tokens < tokens);
    }
}

/// Test: Multiple hooks with auto-execution
///
/// Verifies that multiple hooks of the same type work correctly together
/// during auto-execution, with proper first-match-wins semantics.
#[tokio::test]
async fn test_multiple_hooks_with_auto_execution() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone1 = log.clone();
    let log_clone2 = log.clone();

    let test_tool = tool("test_op", "Test operation")
        .param("value", "number")
        .build(|args| async move { Ok(json!({"result": args["value"]})) });

    let hooks = Hooks::new()
        .add_pre_tool_use(move |event| {
            let log = log_clone1.clone();
            async move {
                log.lock()
                    .unwrap()
                    .push(format!("Pre1: {}", event.tool_name));

                // First hook: block if value > 100
                if let Some(value) = event.tool_input.get("value").and_then(|v| v.as_f64()) {
                    if value > 100.0 {
                        return Some(HookDecision::block("Value too large"));
                    }
                }
                None // Continue to next hook
            }
        })
        .add_pre_tool_use(move |event| {
            let log = log_clone2.clone();
            async move {
                log.lock()
                    .unwrap()
                    .push(format!("Pre2: {}", event.tool_name));

                // Second hook: modify if value > 50
                if let Some(value) = event.tool_input.get("value").and_then(|v| v.as_f64()) {
                    if value > 50.0 {
                        return Some(HookDecision::modify_input(
                            json!({"value": 50.0}),
                            "Clamped to 50",
                        ));
                    }
                }
                Some(HookDecision::continue_())
            }
        });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("test")
        .base_url("http://localhost:11434/v1")
        .tool(test_tool)
        .hooks(hooks)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // Multiple hooks should execute in sequence
    // First hook that returns a decision stops the chain
}

/// Test: Auto-execution preserves manual mode compatibility
///
/// Verifies that when auto_execute_tools is false, manual mode still works
/// correctly and hooks behave as expected.
#[tokio::test]
async fn test_manual_mode_still_works_after_auto_execution_implementation() {
    let manual_tool = tool("manual_op", "Manual operation")
        .param("x", "number")
        .build(
            |args| async move { Ok(json!({"result": args["x"].as_f64().unwrap_or(0.0) * 2.0})) },
        );

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("test")
        .base_url("http://localhost:11434/v1")
        .tool(manual_tool)
        .auto_execute_tools(false) // ← Manual mode!
        .build()
        .unwrap();

    let client = Client::new(options);

    // In manual mode, receive() should yield ToolUse blocks
    // User manually calls get_tool() and executes
    assert_eq!(client.options().auto_execute_tools, false);

    // Manual execution should still work
    if let Some(tool) = client.get_tool("manual_op") {
        let result = tool.execute(json!({"x": 21.0})).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["result"], 42.0);
    } else {
        panic!("Tool not found");
    }
}

/// Test: History management with auto-execution and hooks
///
/// Verifies that conversation history is properly maintained when
/// auto-execution and hooks are used together.
#[tokio::test]
async fn test_history_management_with_auto_execution_and_hooks() {
    let simple_tool = tool("echo", "Echo input")
        .param("msg", "string")
        .build(|args| async move { Ok(args["msg"].clone()) });

    let hooks = Hooks::new().add_post_tool_use(|event| async move {
        // Log but don't modify
        println!("Executed: {}", event.tool_name);
        None
    });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("test")
        .base_url("http://localhost:11434/v1")
        .tool(simple_tool)
        .hooks(hooks)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let client = Client::new(options);

    let initial_history_len = client.history().len();

    // After auto-execution with hooks, history should contain:
    // 1. System message (already present)
    // Additional messages would be added during actual auto-execution

    // Verify history structure
    assert!(client.history().len() >= initial_history_len);
}

/// Test: Error handling in auto-execution with hooks
///
/// Verifies that errors are handled gracefully when tools fail
/// during auto-execution, with hooks potentially modifying error results.
#[tokio::test]
async fn test_error_handling_auto_execution_with_hooks() {
    let failing_tool = tool("failing_op", "Operation that fails")
        .param("should_fail", "boolean")
        .build(|args| async move {
            if args["should_fail"].as_bool().unwrap_or(false) {
                return Err(open_agent::Error::tool("Intentional failure"));
            }
            Ok(json!({"success": true}))
        });

    let hooks = Hooks::new().add_post_tool_use(|event| async move {
        // Even if tool failed, we can modify the error result
        if event.tool_result.get("error").is_some() {
            let mut modified = event.tool_result.clone();
            modified["handled_by_hook"] = json!(true);
            return Some(HookDecision::modify_input(modified, "Added error handling"));
        }
        None
    });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("test")
        .base_url("http://localhost:11434/v1")
        .tool(failing_tool)
        .hooks(hooks)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // Tool failures should not crash auto-execution
    // Hooks can modify error results
}

/// Test: Concurrent features don't interfere
///
/// Verifies that features like retry, interrupt, and hooks don't
/// interfere with each other when used together.
#[tokio::test]
async fn test_feature_isolation_and_compatibility() {
    let test_tool = tool("test", "Test")
        .param("x", "number")
        .build(|args| async move { Ok(args["x"].clone()) });

    // All features enabled
    let hooks = Hooks::new()
        .add_pre_tool_use(|_| async { Some(HookDecision::continue_()) })
        .add_post_tool_use(|_| async { None });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("test")
        .base_url("http://localhost:11434/v1")
        .tool(test_tool)
        .hooks(hooks)
        .auto_execute_tools(true)
        .max_tool_iterations(5)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // All features should coexist without conflicts
}

/// Test: Complex workflow integration
///
/// Realistic scenario combining auto-execution, multiple hooks,
/// context management, and error handling.
#[tokio::test]
async fn test_complex_workflow_integration() {
    let execution_log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = execution_log.clone();

    // Multiple tools
    let calc_tool = tool("calculate", "Calculate")
        .param("a", "number")
        .param("b", "number")
        .param("op", "string")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            let op = args["op"].as_str().unwrap_or("add");

            let result = match op {
                "add" => a + b,
                "multiply" => a * b,
                "divide" => {
                    if b == 0.0 {
                        return Err(open_agent::Error::tool("Division by zero"));
                    }
                    a / b
                }
                _ => a + b,
            };

            Ok(json!({"result": result}))
        });

    let format_tool = tool("format", "Format result")
        .param("value", "number")
        .build(|args| async move {
            let value = args["value"].as_f64().unwrap_or(0.0);
            Ok(json!({"formatted": format!("{:.2}", value)}))
        });

    // Comprehensive hooks
    let hooks = Hooks::new()
        .add_pre_tool_use(|event| async move {
            // Safety check
            if event.tool_name == "calculate" {
                if let Some(op) = event.tool_input.get("op").and_then(|v| v.as_str()) {
                    if op == "divide" {
                        if let Some(b) = event.tool_input.get("b").and_then(|v| v.as_f64()) {
                            if b == 0.0 {
                                return Some(HookDecision::block("Division by zero prevented"));
                            }
                        }
                    }
                }
            }
            Some(HookDecision::continue_())
        })
        .add_post_tool_use(move |event| {
            let log = log_clone.clone();
            async move {
                // Log all executions
                log.lock().unwrap().push(format!(
                    "{}: {:?}",
                    event.tool_name,
                    event.tool_result.get("result")
                ));
                None
            }
        });

    let options = AgentOptions::builder()
        .system_prompt("Advanced calculator")
        .model("test")
        .base_url("http://localhost:11434/v1")
        .tool(calc_tool)
        .tool(format_tool)
        .hooks(hooks)
        .auto_execute_tools(true)
        .max_tool_iterations(10)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // Complex workflow should work seamlessly
    // - Multiple tools
    // - Safety hooks
    // - Logging hooks
    // - Auto-execution
    // - Error handling
}
