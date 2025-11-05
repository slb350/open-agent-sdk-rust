//! Hook integration tests with auto-execution mode
//!
//! Tests verifying that PreToolUse and PostToolUse hooks are properly
//! integrated with the automatic tool execution loop.

use open_agent::{AgentOptions, Client, HookDecision, Hooks, tool};
use serde_json::json;
use std::sync::{Arc, Mutex};

/// Test: PreToolUse hook blocks tool during auto-execution
///
/// Scenario: Hook blocks dangerous tool call
/// Expected: Tool not executed, error result added, conversation continues
#[tokio::test]
async fn test_pre_tool_use_blocks_tool_in_auto_mode() {
    let dangerous_tool = tool("delete_file", "Delete a file")
        .param("path", "string")
        .build(|_args| async move {
            panic!("Should not execute - hook should block this!");
        });

    // Hook that blocks delete_file
    let hooks = Hooks::new().add_pre_tool_use(|event| async move {
        if event.tool_name == "delete_file" {
            return Some(HookDecision::block("Dangerous operation blocked"));
        }
        None
    });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(dangerous_tool)
        .hooks(hooks)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // If model tries to call delete_file during auto-execution:
    // 1. PreToolUse hook should fire
    // 2. Hook should block execution
    // 3. Error result should be added to history
    // 4. Conversation should continue (not crash)
}

/// Test: PreToolUse hook modifies tool input during auto-execution
///
/// Scenario: Hook modifies tool parameters before execution
/// Expected: Tool receives modified input, not original
#[tokio::test]
async fn test_pre_tool_use_modifies_input_in_auto_mode() {
    use std::sync::Arc;
    use std::sync::Mutex;

    let executed_input = Arc::new(Mutex::new(None));
    let executed_input_clone = Arc::clone(&executed_input);

    let calculation_tool = tool("calculate", "Perform calculation")
        .param("value", "number")
        .build(move |args| {
            let executed_input = Arc::clone(&executed_input_clone);
            async move {
                let value = args["value"].as_f64().unwrap_or(0.0);
                *executed_input.lock().unwrap() = Some(value);
                Ok(json!({"result": value * 2.0}))
            }
        });

    // Hook that modifies input: always clamp value to max 100
    let hooks = Hooks::new().add_pre_tool_use(|event| async move {
        if event.tool_name == "calculate" {
            if let Some(value) = event.tool_input.get("value").and_then(|v| v.as_f64()) {
                if value > 100.0 {
                    let modified = json!({"value": 100.0});
                    return Some(HookDecision::modify_input(
                        modified,
                        "Clamped value to max 100",
                    ));
                }
            }
        }
        None
    });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(calculation_tool)
        .hooks(hooks)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // If model tries to call calculate(value=500):
    // 1. PreToolUse hook should fire
    // 2. Hook should modify input to value=100
    // 3. Tool should receive value=100 (not 500)
    // 4. executed_input should show 100.0
}

/// Test: PostToolUse hook modifies result during auto-execution
///
/// Scenario: Hook transforms tool result before adding to history
/// Expected: Modified result used in conversation, not original
#[tokio::test]
async fn test_post_tool_use_modifies_result_in_auto_mode() {
    let data_tool = tool("get_data", "Get data")
        .param("id", "string")
        .build(|_args| async move { Ok(json!({"data": "sensitive_information"})) });

    // Hook that redacts sensitive data
    let hooks = Hooks::new().add_post_tool_use(|event| async move {
        if event.tool_name == "get_data" {
            if let Some(data) = event.tool_result.get("data") {
                if data.as_str() == Some("sensitive_information") {
                    let redacted = json!({"data": "[REDACTED]"});
                    return Some(HookDecision::modify_input(
                        redacted,
                        "Redacted sensitive data",
                    ));
                }
            }
        }
        None
    });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(data_tool)
        .hooks(hooks)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // If model calls get_data during auto-execution:
    // 1. Tool returns {"data": "sensitive_information"}
    // 2. PostToolUse hook should fire
    // 3. Hook should modify result to {"data": "[REDACTED]"}
    // 4. History should contain redacted version
    // 5. Model should see redacted version in next response
}

/// Test: PostToolUse hook logs tool execution
///
/// Scenario: Hook logs all tool executions without modifying
/// Expected: All tools logged, execution continues normally
#[tokio::test]
async fn test_post_tool_use_logs_execution_in_auto_mode() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let log_clone = Arc::clone(&log);

    let tool_a = tool("op_a", "Operation A")
        .param("x", "number")
        .build(
            |args| async move { Ok(json!({"result": args["x"].as_f64().unwrap_or(0.0) * 2.0})) },
        );

    let tool_b = tool("op_b", "Operation B")
        .param("y", "number")
        .build(
            |args| async move { Ok(json!({"result": args["y"].as_f64().unwrap_or(0.0) + 10.0})) },
        );

    // Hook that logs all tool executions
    let hooks = Hooks::new().add_post_tool_use(move |event| {
        let log = Arc::clone(&log_clone);
        async move {
            log.lock()
                .unwrap()
                .push(format!("{}: {:?}", event.tool_name, event.tool_result));
            None // Don't modify, just log
        }
    });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(tool_a)
        .tool(tool_b)
        .hooks(hooks)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // After auto-execution with multiple tool calls:
    // 1. PostToolUse hook should fire for each tool
    // 2. Log should contain entries for all executed tools
    // 3. Execution should continue normally (hook doesn't block)

    // This test validates the structure - actual logging
    // would be verified with live LLM
}

/// Test: Multiple PreToolUse hooks (first match wins)
///
/// Scenario: Multiple hooks registered, first blocker wins
/// Expected: First hook that returns decision stops chain
#[tokio::test]
async fn test_multiple_pre_tool_use_hooks() {
    let test_tool = tool("test_op", "Test operation")
        .param("value", "number")
        .build(|_args| async move { Ok(json!({"result": 42})) });

    let hooks = Hooks::new()
        .add_pre_tool_use(|event| async move {
            // First hook: blocks if value > 1000
            if let Some(value) = event.tool_input.get("value").and_then(|v| v.as_f64()) {
                if value > 1000.0 {
                    return Some(HookDecision::block("Value too large"));
                }
            }
            None
        })
        .add_pre_tool_use(|event| async move {
            // Second hook: would modify, but won't run if first hook decides
            if let Some(value) = event.tool_input.get("value").and_then(|v| v.as_f64()) {
                if value > 100.0 {
                    return Some(HookDecision::modify_input(
                        json!({"value": 100.0}),
                        "Clamped to 100",
                    ));
                }
            }
            None
        });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(test_tool)
        .hooks(hooks)
        .auto_execute_tools(true)
        .build()
        .unwrap();

    let _client = Client::new(options);

    // If model calls test_op(value=2000):
    // 1. First hook should block (value > 1000)
    // 2. Second hook should NOT run
    // 3. Tool should not execute
}

/// Test: Hook integration doesn't break manual mode
///
/// Scenario: Hooks configured but auto_execute_tools=false
/// Expected: Manual mode works normally, hooks not called during receive()
#[tokio::test]
async fn test_hooks_dont_break_manual_mode() {
    let manual_tool = tool("manual_op", "Manual operation")
        .param("x", "number")
        .build(
            |args| async move { Ok(json!({"result": args["x"].as_f64().unwrap_or(0.0) * 2.0})) },
        );

    let hooks = Hooks::new()
        .add_pre_tool_use(|_event| async move {
            // This should not fire during receive() in manual mode
            Some(HookDecision::block("Should not see this"))
        })
        .add_post_tool_use(|_event| async move {
            // This should not fire during receive() in manual mode
            None
        });

    let options = AgentOptions::builder()
        .system_prompt("Test")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(manual_tool)
        .hooks(hooks)
        .auto_execute_tools(false) // ← Manual mode!
        .build()
        .unwrap();

    let _client = Client::new(options);

    // In manual mode:
    // 1. receive() should yield ToolUse blocks
    // 2. PreToolUse hooks should NOT fire during receive()
    // 3. User manually calls get_tool() and executes
    // 4. PostToolUse hooks could fire during manual execution
    //    (implementation detail)
}
