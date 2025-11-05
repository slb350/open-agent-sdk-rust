//! Context Management Examples
//!
//! This demonstrates manual history management patterns using the context utilities.
//!
//! The SDK provides low-level helpers (estimate_tokens, truncate_messages) but does NOT
//! automatically manage context. You decide when and how to manage history based on your
//! domain-specific needs.
//!
//! Patterns demonstrated:
//! 1. Stateless agents (recommended for single-task agents)
//! 2. Manual truncation at natural breakpoints
//! 3. Token budget monitoring with periodic checks

use open_agent::{
    AgentOptions, Client, ContentBlock, estimate_tokens, is_approaching_limit, truncate_messages,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(70));
    println!("CONTEXT MANAGEMENT EXAMPLES");
    println!("{}", "=".repeat(70));
    println!();

    // Run each pattern
    pattern_1_stateless().await?;
    pattern_2_manual_truncation().await?;
    pattern_3_token_monitoring().await?;

    Ok(())
}

// ============================================================================
// Pattern 1: Stateless Agents (Recommended)
// ============================================================================
// Best for: Single-task agents (copy editor, code formatter, etc.)

async fn pattern_1_stateless() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Pattern 1: Stateless Agents ===");
    println!("Best for: Single-task agents with no context needed");
    println!();

    let options = AgentOptions::builder()
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .system_prompt("You are a helpful assistant")
        .build()?;

    // Process each task independently
    let tasks = vec!["Explain Rust", "Explain Python", "Explain JavaScript"];

    for task in tasks {
        // Fresh client for each task - no history accumulation
        let mut client = Client::new(options.clone());
        client.send(task).await?;

        let mut response = String::new();
        while let Some(block) = client.receive().await? {
            if let ContentBlock::Text(text) = block {
                response.push_str(&text.text);
            }
        }

        println!("Task: {}", task);
        println!("Response length: {} chars", response.len());
        println!("History size: {} messages", client.history().len());
        println!("Estimated tokens: {}", estimate_tokens(client.history()));
        println!();
    }
    // Client dropped, fresh context for next task

    println!("{}", "-".repeat(70));
    println!();

    Ok(())
}

// ============================================================================
// Pattern 2: Manual Truncation at Natural Breakpoints
// ============================================================================
// Best for: Multi-turn conversations with clear task boundaries

async fn pattern_2_manual_truncation() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Pattern 2: Manual Truncation ===");
    println!("Best for: Multi-turn conversations with task boundaries");
    println!();

    let options = AgentOptions::builder()
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .system_prompt("You are a helpful coding assistant")
        .max_turns(10)
        .build()?;

    let mut client = Client::new(options);

    // Task 1: Code analysis (simplified - no actual API call for demo)
    println!("Task 1: Adding messages to history...");
    client
        .send("Analyze this: def add(a, b): return a + b")
        .await?;
    // Simulate processing
    while client.receive().await?.is_some() {
        // Process messages
    }
    println!("After task 1: {} messages", client.history().len());

    // Task 2: Write tests (simplified)
    println!("\nTask 2: Adding more messages...");
    client.send("Write unit tests for the add function").await?;
    while client.receive().await?.is_some() {
        // Process messages
    }
    println!("After task 2: {} messages", client.history().len());

    // Truncate after completing major milestone
    println!("\nTruncating history (keeping last 3 messages)...");
    let truncated = truncate_messages(client.history(), 3, true);
    *client.history_mut() = truncated;
    println!("After truncation: {} messages", client.history().len());

    println!();
    println!("{}", "-".repeat(70));
    println!();

    Ok(())
}

// ============================================================================
// Pattern 3: Token Budget Monitoring
// ============================================================================
// Best for: Long-running conversations with token limits

async fn pattern_3_token_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Pattern 3: Token Budget Monitoring ===");
    println!("Best for: Long-running conversations with token limits");
    println!();

    let options = AgentOptions::builder()
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .system_prompt("You are a helpful assistant")
        .build()?;

    let mut client = Client::new(options);

    // Simulate multiple interactions
    let interactions = [
        "What is Rust?",
        "Explain ownership",
        "What are lifetimes?",
        "How do traits work?",
        "Explain async/await",
    ];

    let token_limit = 1000; // Example limit (much lower for demo)
    let margin = 0.8; // 80% threshold

    for (i, prompt) in interactions.iter().enumerate() {
        println!("Interaction {}: {}", i + 1, prompt);

        // Check token budget before sending
        let current_tokens = estimate_tokens(client.history());
        println!("  Current tokens: {}", current_tokens);

        if is_approaching_limit(client.history(), token_limit, margin) {
            println!("  ⚠️  Approaching token limit! Truncating...");
            let truncated = truncate_messages(client.history(), 3, true);
            *client.history_mut() = truncated;
            println!(
                "  After truncation: {} tokens",
                estimate_tokens(client.history())
            );
        }

        // Send message (simplified - no actual API call for demo)
        client.send(prompt).await?;

        println!();
    }

    println!("Final history size: {} messages", client.history().len());
    println!("Final token count: {}", estimate_tokens(client.history()));

    println!();
    println!("{}", "-".repeat(70));
    println!();

    Ok(())
}
