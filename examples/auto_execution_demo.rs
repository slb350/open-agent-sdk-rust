//! Auto-Execution Mode Demo
//!
//! Demonstrates the automatic tool execution feature where the SDK
//! automatically executes tool calls and continues the conversation
//! until a text-only response is received.
//!
//! This example shows:
//! - Auto-execution with calculator tools
//! - Automatic iteration through tool calls
//! - Seamless tool result integration
//! - Final text response delivery
//!
//! Usage:
//!   cargo run --example auto_execution_demo
//!
//! Requires:
//!   - Ollama running on localhost:11434
//!   - qwen3:8b model (or change model in code)

use open_agent::{AgentOptions, Client, ContentBlock, tool};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(70));
    println!("AUTO-EXECUTION MODE DEMO");
    println!("{}", "=".repeat(70));
    println!();
    println!("This example demonstrates automatic tool execution where:");
    println!("  - SDK automatically executes tool calls");
    println!("  - Continues conversation until text-only response");
    println!("  - You only receive the final answer, not tool blocks");
    println!();

    // ============================================================================
    // Setup: Create Calculator Tools
    // ============================================================================

    let add_tool = tool("add", "Add two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            let result = a + b;
            println!("  🔧 Tool: add({}, {}) = {}", a, b, result);
            Ok(json!({"result": result}))
        });

    let multiply_tool = tool("multiply", "Multiply two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            let result = a * b;
            println!("  🔧 Tool: multiply({}, {}) = {}", a, b, result);
            Ok(json!({"result": result}))
        });

    let subtract_tool = tool("subtract", "Subtract two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            let result = a - b;
            println!("  🔧 Tool: subtract({}, {}) = {}", a, b, result);
            Ok(json!({"result": result}))
        });

    let divide_tool = tool("divide", "Divide two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            if b == 0.0 {
                return Err(open_agent::Error::tool("Division by zero"));
            }
            let result = a / b;
            println!("  🔧 Tool: divide({}, {}) = {}", a, b, result);
            Ok(json!({"result": result}))
        });

    // ============================================================================
    // Setup: Configure Client with Auto-Execution
    // ============================================================================

    let options = AgentOptions::builder()
        .system_prompt(
            "You are a calculator assistant. Use the provided tools to perform calculations. \
             Show your work step by step.",
        )
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(add_tool)
        .tool(multiply_tool)
        .tool(subtract_tool)
        .tool(divide_tool)
        .auto_execute_tools(true) // ← Enable auto-execution!
        .max_tool_iterations(10) // Allow up to 10 tool calls
        .build()?;

    let mut client = Client::new(options)?;

    println!("{}", "-".repeat(70));

    // ============================================================================
    // Example 1: Simple Calculation
    // ============================================================================

    println!("\n📝 Example 1: Simple Calculation");
    println!("{}", "-".repeat(70));

    let query1 = "Calculate 15 + 27";
    println!("Query: {}", query1);
    println!();

    client.send(query1).await?;

    let mut response = String::new();
    while let Some(block) = client.receive().await? {
        match block {
            ContentBlock::Text(text) => {
                response.push_str(&text.text);
            }
            ContentBlock::ToolUse(_) => {
                // Should NOT receive ToolUse blocks in auto mode!
                println!("⚠️  Unexpected: Received ToolUse block");
            }
            ContentBlock::ToolResult(_) => {
                // Should NOT receive ToolResult blocks either!
                println!("⚠️  Unexpected: Received ToolResult block");
            }
        }
    }

    println!();
    println!("🤖 Assistant: {}", response.trim());

    // ============================================================================
    // Example 2: Multi-Step Calculation
    // ============================================================================

    println!("\n{}", "-".repeat(70));
    println!("\n📝 Example 2: Multi-Step Calculation");
    println!("{}", "-".repeat(70));

    let query2 = "Calculate (10 + 5) * 3";
    println!("Query: {}", query2);
    println!();

    client.send(query2).await?;

    let mut response = String::new();
    while let Some(block) = client.receive().await? {
        match block {
            ContentBlock::Text(text) => {
                response.push_str(&text.text);
            }
            ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) => {
                println!("⚠️  Unexpected: Received tool block in auto mode");
            }
        }
    }

    println!();
    println!("🤖 Assistant: {}", response.trim());

    // ============================================================================
    // Example 3: Complex Multi-Step Expression
    // ============================================================================

    println!("\n{}", "-".repeat(70));
    println!("\n📝 Example 3: Complex Expression");
    println!("{}", "-".repeat(70));

    let query3 = "Calculate (20 - 8) * 3 + 15";
    println!("Query: {}", query3);
    println!();

    client.send(query3).await?;

    let mut response = String::new();
    while let Some(block) = client.receive().await? {
        match block {
            ContentBlock::Text(text) => {
                response.push_str(&text.text);
            }
            ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) => {
                println!("⚠️  Unexpected: Received tool block in auto mode");
            }
        }
    }

    println!();
    println!("🤖 Assistant: {}", response.trim());

    // ============================================================================
    // Show History
    // ============================================================================

    println!("\n{}", "=".repeat(70));
    println!("CONVERSATION HISTORY");
    println!("{}", "=".repeat(70));
    println!();
    println!("Total messages in history: {}", client.history().len());
    println!();
    println!("History shows all tool calls and results that were executed");
    println!("automatically during the conversation.");

    // ============================================================================
    // Summary
    // ============================================================================

    println!("\n{}", "=".repeat(70));
    println!("AUTO-EXECUTION SUMMARY");
    println!("{}", "=".repeat(70));
    println!();
    println!("✅ Key Features Demonstrated:");
    println!("   - Auto-execution enabled with .auto_execute_tools(true)");
    println!("   - Tools executed automatically without manual intervention");
    println!("   - Only final text responses delivered to application");
    println!("   - Tool calls and results tracked in history");
    println!("   - Multi-step calculations handled seamlessly");
    println!();
    println!("📚 Comparison:");
    println!("   Manual Mode:");
    println!("     - Receive ToolUse blocks → Execute tools → Send results");
    println!("     - Application handles tool execution loop");
    println!();
    println!("   Auto Mode:");
    println!("     - SDK handles everything automatically");
    println!("     - Application only receives final text answers");
    println!("     - Much simpler application code!");
    println!();

    Ok(())
}
