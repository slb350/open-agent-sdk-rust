//! Anthropic messages endpoint
//!
//! The same single-turn query as `simple_query.rs`, against an endpoint that speaks the
//! Anthropic messages protocol instead of OpenAI chat completions. The only difference at
//! the call site is `.protocol(ApiProtocol::Anthropic)`; the SDK then posts to
//! `{base_url}/messages` with `x-api-key` and `anthropic-version`, and decodes the
//! Anthropic streaming vocabulary into the same `StreamEvent`s.
//!
//! Run with:
//!
//! ```bash
//! ANTHROPIC_API_KEY=sk-... cargo run --example anthropic_query
//! ```

use futures::StreamExt;
use open_agent::{AgentOptions, ApiProtocol, ContentBlock, StreamEvent, query};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| "set ANTHROPIC_API_KEY to run this example")?;

    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant")
        .model("claude-sonnet-5")
        .base_url("https://api.anthropic.com/v1")
        .protocol(ApiProtocol::Anthropic)
        .api_key(api_key)
        // Optional on Anthropic itself, required by some Anthropic-shaped third-party
        // endpoints. Unset means the field is omitted, so the server decides.
        .max_tokens(500)
        // Thinking blocks arrive on the reasoning channel and never enter content or
        // conversation history. They appear only when the endpoint emits them: the SDK
        // sends no `thinking` request field, and current Anthropic models omit the
        // summary by default.
        .include_reasoning(true)
        .build()?;

    println!("Sending query to {}...\n", options.model());

    let mut stream = query("What's the capital of France? Please be brief.", &options).await?;

    print!("Response: ");

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Block(ContentBlock::Text(text)) => {
                print!("{}", text.text);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            StreamEvent::Reasoning(thinking) => {
                print!("\n[thinking] {thinking}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            // Anthropic stop reasons map onto the same `FinishReason` the OpenAI protocol
            // produces, so `Length` means a truncation on either wire format.
            StreamEvent::Finish(reason) => {
                println!("\n[stopped: {reason}]");
            }
            _ => {}
        }
    }

    println!("\nQuery complete!");

    Ok(())
}
