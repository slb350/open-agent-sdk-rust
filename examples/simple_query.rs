//! Simple query example
//!
//! Demonstrates basic usage of the query function

use futures::StreamExt;
use open_agent::{query, AgentOptions, ContentBlock};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure the agent
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .temperature(0.7)
        .max_tokens(500)
        .build()?;

    println!("Sending query to model...\n");

    // Send query and stream response
    let mut stream = query("What's the capital of France? Please be brief.", &options).await?;

    print!("Response: ");

    while let Some(block) = stream.next().await {
        match block? {
            ContentBlock::Text(text) => {
                print!("{}", text.text);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            ContentBlock::ToolUse(tool) => {
                println!("\nTool called: {} (id: {})", tool.name, tool.id);
                println!("Arguments: {}", tool.input);
            }
            ContentBlock::ToolResult(_) => {
                // Tool results not expected in simple query
            }
        }
    }

    println!("\n\nQuery complete!");

    Ok(())
}
