//! Simple query example
//!
//! Demonstrates basic usage of the query function, including reading the terminating
//! `StreamEvent::Finish` to learn why generation stopped.

use futures::StreamExt;
use open_agent::{AgentOptions, ContentBlock, FinishReason, StreamEvent, query};

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

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Block(ContentBlock::Text(text)) => {
                print!("{}", text.text);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            StreamEvent::Block(ContentBlock::ToolUse(tool)) => {
                println!("\nTool called: {} (id: {})", tool.name(), tool.id());
                println!("Arguments: {}", tool.input());
            }
            // The stream always ends with exactly one Finish event. `Length` means the
            // response was cut off at the token cap rather than finished — worth knowing
            // before treating a partial answer as the model's final word.
            StreamEvent::Finish(FinishReason::Length) => {
                println!("\n[truncated at the token cap]");
            }
            StreamEvent::Finish(reason) => {
                println!("\n[stopped: {reason}]");
            }
            _ => {}
        }
    }

    println!("\nQuery complete!");

    Ok(())
}
