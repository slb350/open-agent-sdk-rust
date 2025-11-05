//! Calculator with Tools Example
//!
//! Demonstrates using tools to give local LLMs function calling abilities.
//! Shows manual tool execution pattern.

use open_agent::{AgentOptions, Client, ContentBlock, tool};
use serde_json::json;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define calculator tools
    let add_tool = tool("add", "Add two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!({"result": a + b}))
        });

    let subtract_tool = tool("subtract", "Subtract two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!({"result": a - b}))
        });

    let multiply_tool = tool("multiply", "Multiply two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!({"result": a * b}))
        });

    let divide_tool = tool("divide", "Divide two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            if b == 0.0 {
                return Ok(json!({"error": "Cannot divide by zero"}));
            }
            Ok(json!({"result": a / b}))
        });

    // Create tool registry for lookup
    let mut tool_registry = HashMap::new();
    tool_registry.insert("add".to_string(), add_tool.clone());
    tool_registry.insert("subtract".to_string(), subtract_tool.clone());
    tool_registry.insert("multiply".to_string(), multiply_tool.clone());
    tool_registry.insert("divide".to_string(), divide_tool.clone());

    // Configure agent with tools
    let options = AgentOptions::builder()
        .system_prompt(
            "You are a helpful calculator assistant. \
             Use the provided tools to perform calculations. \
             Always show your work and explain the result.",
        )
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .tools(vec![add_tool, subtract_tool, multiply_tool, divide_tool])
        .auto_execute_tools(false) // Manual mode for this example
        .max_turns(5)
        .temperature(0.1)
        .build()?;

    // Example calculations
    let queries = vec![
        "What is 25 plus 17?",
        "Calculate 144 divided by 12",
        "What's 7 times 8, then add 5?",
    ];

    println!("{}", "=".repeat(70));
    println!("CALCULATOR TOOLS EXAMPLE");
    println!("{}", "=".repeat(70));
    println!();

    for query in queries {
        println!("User: {}", query);
        println!("{}", "-".repeat(50));

        let mut client = Client::new(options.clone());
        client.send(query).await?;

        // Process responses and handle tool calls
        while let Some(block) = client.receive().await? {
            match block {
                ContentBlock::Text(text) => {
                    if !text.text.trim().is_empty() {
                        println!("Assistant: {}", text.text);
                    }
                }
                ContentBlock::ToolUse(tool_use) => {
                    println!("🔧 Tool call: {}", tool_use.name);
                    println!("   Arguments: {}", tool_use.input);

                    // Look up and execute tool
                    if let Some(tool) = tool_registry.get(&tool_use.name) {
                        match tool.execute(tool_use.input.clone()).await {
                            Ok(result) => {
                                println!("   Result: {}", result);

                                // Add result to conversation
                                client.add_tool_result(&tool_use.id, result);

                                // Continue conversation to get assistant's response
                                client.send("").await?;
                            }
                            Err(e) => {
                                eprintln!("   Error executing tool: {}", e);
                            }
                        }
                    } else {
                        eprintln!("   Tool not found: {}", tool_use.name);
                    }
                }
                ContentBlock::ToolResult(_) => {
                    // Tool results handled manually in this example
                }
            }
        }

        println!();
    }

    println!("{}", "=".repeat(70));
    println!("Example complete!");

    Ok(())
}
