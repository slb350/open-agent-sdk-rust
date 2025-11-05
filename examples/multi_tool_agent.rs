//! Multi-Tool Agent Example
//!
//! Demonstrates complex agent with multiple tools, hooks, and auto-execution.
//!
//! This agent can:
//! - Perform calculations
//! - Get current date/time
//! - Convert units
//! - Search (mock)
//! - Format data
//!
//! Shows realistic usage with safety hooks and logging.
//!
//! Usage:
//!   cargo run --example multi_tool_agent
//!
//! Requires:
//!   - Ollama running on localhost:11434
//!   - qwen3:8b model (or adjust model name)

use open_agent::{
    AgentOptions, Client, ContentBlock, HookDecision, Hooks, PostToolUseEvent, PreToolUseEvent,
    tool,
};
use serde_json::json;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(70));
    println!("MULTI-TOOL AGENT DEMO");
    println!("Realistic agent with 5+ tools, hooks, and auto-execution");
    println!("{}", "=".repeat(70));
    println!();

    // Execution log
    let execution_log = Arc::new(Mutex::new(Vec::new()));

    // Tool 1: Calculator
    let calculator = tool("calculate", "Perform arithmetic")
        .param("operation", "string")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let op = args["operation"].as_str().unwrap_or("");
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);

            let result = match op {
                "add" => a + b,
                "subtract" => a - b,
                "multiply" => a * b,
                "divide" => {
                    if b == 0.0 {
                        return Err(open_agent::Error::tool("Division by zero"));
                    }
                    a / b
                }
                _ => return Err(open_agent::Error::tool("Unknown operation")),
            };

            Ok(json!({"result": result, "operation": op}))
        });

    // Tool 2: Get current date/time
    let datetime = tool("get_datetime", "Get current date and time")
        .param("format", "string")
        .build(|args| async move {
            let format = args["format"].as_str().unwrap_or("unix");
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let formatted = match format {
                "unix" => now.to_string(),
                _ => format!("Unix timestamp: {}", now),
            };

            Ok(json!({"datetime": formatted, "format": format}))
        });

    // Tool 3: Unit converter
    let converter = tool("convert_units", "Convert between units")
        .param("value", "number")
        .param("from_unit", "string")
        .param("to_unit", "string")
        .build(|args| async move {
            let value = args["value"].as_f64().unwrap_or(0.0);
            let from = args["from_unit"].as_str().unwrap_or("");
            let to = args["to_unit"].as_str().unwrap_or("");

            // Simple conversion (km/miles, kg/lbs, etc.)
            let result = match (from, to) {
                ("km", "miles") => value * 0.621371,
                ("miles", "km") => value / 0.621371,
                ("kg", "lbs") => value * 2.20462,
                ("lbs", "kg") => value / 2.20462,
                ("celsius", "fahrenheit") => (value * 9.0 / 5.0) + 32.0,
                ("fahrenheit", "celsius") => (value - 32.0) * 5.0 / 9.0,
                _ => return Err(open_agent::Error::tool("Unsupported conversion")),
            };

            Ok(json!({
                "result": result,
                "from": from,
                "to": to,
                "original_value": value
            }))
        });

    // Tool 4: Search (mock)
    let search = tool("search", "Search for information")
        .param("query", "string")
        .build(|args| async move {
            let query = args["query"].as_str().unwrap_or("");

            // Mock results
            Ok(json!({
                "results": [
                    {"title": format!("Result for: {}", query), "relevance": 0.95},
                    {"title": "Related information", "relevance": 0.80},
                ],
                "query": query
            }))
        });

    // Tool 5: Data formatter
    let formatter = tool("format_data", "Format data as table/list")
        .param("data", "object")
        .param("format", "string")
        .build(|args| async move {
            let format = args["format"].as_str().unwrap_or("list");

            Ok(json!({
                "formatted": format!("Data formatted as: {}", format),
                "format": format
            }))
        });

    // Set up hooks
    let log_clone = execution_log.clone();
    let hooks = Hooks::new()
        .add_pre_tool_use(|event: PreToolUseEvent| async move {
            println!("🔍 PreToolUse: {}", event.tool_name);

            // Safety check: block dangerous operations
            if event.tool_name == "delete" || event.tool_name == "modify_system" {
                println!("   🛑 BLOCKED: Dangerous operation");
                return Some(HookDecision::block("Safety policy violation"));
            }

            // Validation: ensure division by zero is caught
            if event.tool_name == "calculate" {
                if let Some(op) = event.tool_input.get("operation").and_then(|v| v.as_str()) {
                    if op == "divide" {
                        if let Some(b) = event.tool_input.get("b").and_then(|v| v.as_f64()) {
                            if b == 0.0 {
                                println!("   🛑 BLOCKED: Division by zero");
                                return Some(HookDecision::block("Division by zero prevented"));
                            }
                        }
                    }
                }
            }

            println!("   ✅ Allowed");
            Some(HookDecision::continue_())
        })
        .add_post_tool_use(move |event: PostToolUseEvent| {
            let log = log_clone.clone();
            async move {
                println!("✨ PostToolUse: {} completed", event.tool_name);

                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();

                // Log execution
                log.lock().unwrap().push(format!(
                    "[{}] {} -> {}",
                    timestamp,
                    event.tool_name,
                    serde_json::to_string(&event.tool_result).unwrap_or_default()
                ));

                // Add metadata to all results
                if let Some(result_obj) = event.tool_result.as_object() {
                    let mut enhanced = result_obj.clone();
                    enhanced.insert(
                        "_metadata".to_string(),
                        json!({
                            "tool": event.tool_name,
                            "executed_at": timestamp,
                        }),
                    );
                    return Some(HookDecision::modify_input(
                        json!(enhanced),
                        "Added metadata",
                    ));
                }

                None
            }
        });

    // Create agent with all tools
    let options = AgentOptions::builder()
        .system_prompt(
            "You are a helpful multi-tool assistant. You have access to: \
             calculator, datetime, unit converter, search, and data formatter. \
             Use these tools to help answer questions. \
             Always use tools for calculations and conversions.",
        )
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tool(calculator)
        .tool(datetime)
        .tool(converter)
        .tool(search)
        .tool(formatter)
        .hooks(hooks)
        .auto_execute_tools(true)
        .max_tool_iterations(10)
        .temperature(0.3)
        .build()?;

    let mut client = Client::new(options);

    println!("Agent configured with:");
    println!("- 5 tools: calculator, datetime, converter, search, formatter");
    println!("- Auto-execution: ENABLED");
    println!("- Hooks: PreToolUse (safety), PostToolUse (logging)");
    println!();
    println!("{}", "-".repeat(70));
    println!();

    // Example queries
    let queries = [
        "What's 15 kilometers in miles?",
        "Calculate 25 divided by 5, then multiply by 3",
        "What's the current Unix timestamp?",
        "Search for information about Rust programming",
    ];

    for (i, query) in queries.iter().enumerate() {
        println!("Query {}: {}", i + 1, query);
        println!();

        client.send(query).await?;

        while let Some(block) = client.receive().await {
            match block? {
                ContentBlock::Text(text) => {
                    println!("Assistant: {}", text.text);
                }
                ContentBlock::ToolUse(tool_use) => {
                    // This shouldn't happen in auto-execution mode
                    println!(
                        "⚠️  Unexpected: ToolUse block in auto mode: {}",
                        tool_use.name
                    );
                }
                ContentBlock::ToolResult(tool_result) => {
                    // This shouldn't happen in auto-execution mode
                    println!(
                        "⚠️  Unexpected: ToolResult block in auto mode: {}",
                        tool_result.tool_use_id
                    );
                }
            }
        }

        println!();
        println!("{}", "-".repeat(70));
        println!();
    }

    // Show execution log
    println!("Execution Log:");
    println!("{}", "=".repeat(70));
    for entry in execution_log.lock().unwrap().iter() {
        println!("{}", entry);
    }
    println!("{}", "=".repeat(70));
    println!();

    println!("Conversation History: {} messages", client.history().len());
    println!();

    // Show what the history looks like
    println!("History breakdown:");
    for (i, msg) in client.history().iter().enumerate() {
        println!(
            "  {}. {:?} - {} content blocks",
            i + 1,
            msg.role,
            msg.content.len()
        );
    }
    println!();

    println!("{}", "=".repeat(70));
    println!("DEMO COMPLETE");
    println!("{}", "=".repeat(70));
    println!();
    println!("Key Features Demonstrated:");
    println!("✅ Multiple specialized tools working together");
    println!("✅ Auto-execution mode (tools called automatically)");
    println!("✅ PreToolUse hooks (safety validation)");
    println!("✅ PostToolUse hooks (logging and metadata)");
    println!("✅ Error handling (division by zero prevention)");
    println!("✅ Comprehensive conversation history tracking");
    println!();
    println!("This example shows a production-ready agent configuration!");
    println!("{}", "=".repeat(70));

    Ok(())
}
