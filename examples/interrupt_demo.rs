//! Interrupt Capability Demo
//!
//! This example demonstrates how to use the interrupt() capability to cancel
//! long-running operations:
//!
//! 1. Timeout-based interruption
//! 2. Conditional interruption (based on content)
//! 3. Concurrent interruption (simulated cancel button)
//! 4. Interrupt and retry
//!
//! Note: This example uses Ollama. Make sure Ollama is running at
//! http://localhost:11434 with a model loaded before running.

use open_agent::{AgentOptions, Client, ContentBlock};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio::time::timeout;

// ============================================================================
// Example 1: Timeout-based Interruption
// ============================================================================
async fn timeout_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("Example 1: Timeout-based Interruption");
    println!("{}", "=".repeat(60));
    println!();

    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant. Be verbose in your responses.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .temperature(0.7)
        .build()?;

    let mut client = Client::new(options)?;
    client
        .send("Write a detailed 1000-word essay about quantum computing")
        .await?;

    println!("Starting to receive response (will timeout after 3 seconds)...\n");

    let mut response_text = String::new();

    // Collect messages with timeout
    let result = timeout(Duration::from_secs(3), async {
        while let Some(block) = client.receive().await? {
            if let ContentBlock::Text(text) = block {
                print!("{}", text.text);
                response_text.push_str(&text.text);
            }
        }
        Ok::<_, Box<dyn std::error::Error>>(())
    })
    .await;

    match result {
        Ok(_) => {
            println!("\n\nResponse completed within timeout");
        }
        Err(_) => {
            // Timeout occurred - interrupt
            client.interrupt();
            println!("\n\n⚠️  Operation timed out and was interrupted!");
            println!("Received {} characters before timeout", response_text.len());
            println!("History preserved: {} messages\n", client.history().len());
        }
    }

    Ok(())
}

// ============================================================================
// Example 2: Conditional Interruption
// ============================================================================
async fn conditional_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("Example 2: Conditional Interruption");
    println!("{}", "=".repeat(60));
    println!();

    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .temperature(0.7)
        .build()?;

    let mut client = Client::new(options)?;
    client.send("Tell me about machine learning").await?;

    println!("Receiving response (will stop if we see 'neural network')...\n");

    let mut full_text = String::new();
    while let Some(block) = client.receive().await? {
        if let ContentBlock::Text(text) = block {
            print!("{}", text.text);
            full_text.push_str(&text.text);

            // Interrupt if we see a specific keyword
            if full_text.to_lowercase().contains("neural network") {
                client.interrupt();
                println!("\n\n⚠️  Found keyword 'neural network' - interrupting!");
                break;
            }
        }
    }

    println!(
        "\nReceived {} characters before interrupt\n",
        full_text.len()
    );

    Ok(())
}

// ============================================================================
// Example 3: Concurrent Interruption
// ============================================================================
fn spawn_interrupt_task(interrupt_handle: Arc<AtomicBool>, delay: Duration) -> JoinHandle<()> {
    tokio::spawn(async move {
        tokio::time::sleep(delay).await;
        println!("\n\n🛑 User clicked cancel button!");
        interrupt_handle.store(true, Ordering::SeqCst);
    })
}

async fn concurrent_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("Example 3: Concurrent Interruption (Simulated User Cancel)");
    println!("{}", "=".repeat(60));
    println!();

    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .temperature(0.7)
        .build()?;

    let mut client = Client::new(options)?;
    client
        .send("Explain artificial intelligence in detail")
        .await?;

    println!("Receiving response...\n");

    // Share only the atomic interrupt flag with the simulated cancel button.
    // The receive loop retains exclusive ownership of the client.
    let cancel_handle = spawn_interrupt_task(client.interrupt_handle(), Duration::from_secs(2));

    let mut full_text = String::new();
    while let Some(block) = client.receive().await? {
        if let ContentBlock::Text(text) = block {
            print!("{}", text.text);
            full_text.push_str(&text.text);
            tokio::time::sleep(Duration::from_millis(50)).await; // Simulate processing
        }
    }

    // Wait for cancel task to complete
    let _ = cancel_handle.await;

    println!(
        "\n\nReceived {} characters before interrupt\n",
        full_text.len()
    );

    Ok(())
}

// ============================================================================
// Example 4: Interrupt and Retry
// ============================================================================
async fn retry_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("Example 4: Interrupt and Retry");
    println!("{}", "=".repeat(60));
    println!();

    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .temperature(0.7)
        .build()?;

    let mut client = Client::new(options)?;

    // First query - will be interrupted
    println!("First query (will be interrupted)...\n");
    client
        .send("Tell me everything about the history of computing")
        .await?;

    let mut count = 0;
    while let Some(block) = client.receive().await? {
        if let ContentBlock::Text(_) = block {
            count += 1;
            if count == 3 {
                println!("\n⚠️  Oops, that was too broad. Interrupting...\n");
                client.interrupt();
                break;
            }
        }
    }

    // Retry with more specific query
    println!("Retrying with more specific query...\n");
    client
        .send("Tell me about Alan Turing in 2 sentences")
        .await?;

    while let Some(block) = client.receive().await? {
        if let ContentBlock::Text(text) = block {
            print!("{}", text.text);
        }
    }

    println!("\n\nSuccess! Query completed after retry.\n");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[tokio::test]
    async fn concurrent_cancel_does_not_require_locking_the_client() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();
        let client = Client::new(options).unwrap();
        let interrupt_handle = client.interrupt_handle();

        let cancel_task =
            spawn_interrupt_task(Arc::clone(&interrupt_handle), Duration::from_millis(1));

        timeout(Duration::from_millis(100), cancel_task)
            .await
            .expect("cancel task deadlocked")
            .expect("cancel task panicked");

        assert!(interrupt_handle.load(Ordering::SeqCst));
        assert!(client.history().is_empty());
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(60));
    println!("INTERRUPT CAPABILITY DEMO");
    println!("{}", "=".repeat(60));
    println!();

    println!("This demo requires Ollama running at http://localhost:11434");
    println!("with a model loaded (e.g., qwen3:8b)\n");

    println!("Running examples...\n");

    // Run each example
    if let Err(e) = timeout_example().await {
        eprintln!("Timeout example error: {}", e);
    }
    tokio::time::sleep(Duration::from_secs(1)).await;

    if let Err(e) = conditional_example().await {
        eprintln!("Conditional example error: {}", e);
    }
    tokio::time::sleep(Duration::from_secs(1)).await;

    if let Err(e) = concurrent_example().await {
        eprintln!("Concurrent example error: {}", e);
    }
    tokio::time::sleep(Duration::from_secs(1)).await;

    if let Err(e) = retry_example().await {
        eprintln!("Retry example error: {}", e);
    }

    println!("{}", "=".repeat(60));
    println!("All examples completed!");
    println!("{}", "=".repeat(60));

    Ok(())
}
