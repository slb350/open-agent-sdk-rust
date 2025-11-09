//! Advanced Patterns - Retry Logic and Concurrent Requests
//!
//! This example demonstrates advanced patterns for production use:
//! 1. Retry with exponential backoff for handling transient failures
//! 2. Concurrent request handling for parallel queries
//! 3. Error handling strategies
//!
//! Usage:
//!     cargo run --example advanced_patterns

use futures::stream::{FuturesUnordered, StreamExt};
use open_agent::retry::{RetryConfig, retry_with_backoff, retry_with_backoff_conditional};
use open_agent::{AgentOptions, Client, ContentBlock};
use std::time::Duration;
use tokio::time::Instant;

// ============================================================================
// Example 1: Retry with Exponential Backoff
// ============================================================================
async fn retry_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("Example 1: Retry with Exponential Backoff");
    println!("{}", "=".repeat(60));
    println!();

    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .temperature(0.7)
        .timeout(5) // Short timeout to demonstrate retries
        .build()?;

    // Configure retry behavior
    let retry_config = RetryConfig::new()
        .with_max_attempts(3)
        .with_initial_delay(Duration::from_secs(1))
        .with_backoff_multiplier(2.0)
        .with_jitter_factor(0.1);

    println!("🔄 Attempting query with retry (up to 3 attempts)...\n");

    let start = Instant::now();
    let result = retry_with_backoff(retry_config, || async {
        let mut client = Client::new(options.clone())?;
        client.send("What is 2+2?").await?;

        let mut response = String::new();
        while let Some(block) = client.receive().await? {
            if let ContentBlock::Text(text) = block {
                response.push_str(&text.text);
            }
        }

        Ok::<_, open_agent::Error>(response)
    })
    .await;

    match result {
        Ok(response) => {
            println!("✅ Success after {:?}", start.elapsed());
            println!("Response: {}\n", response.trim());
        }
        Err(e) => {
            println!("❌ Failed after all retries: {}\n", e);
        }
    }

    Ok(())
}

// ============================================================================
// Example 2: Conditional Retry (Only Retry Transient Errors)
// ============================================================================
async fn conditional_retry_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("Example 2: Conditional Retry (Transient Errors Only)");
    println!("{}", "=".repeat(60));
    println!();

    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .temperature(0.7)
        .build()?;

    let retry_config = RetryConfig::new().with_max_attempts(3);

    println!("🔄 Using conditional retry (only retries network/server errors)...\n");

    let start = Instant::now();
    let result = retry_with_backoff_conditional(retry_config, || async {
        let mut client = Client::new(options.clone())?;
        client
            .send("Explain quantum computing in one sentence")
            .await?;

        let mut response = String::new();
        while let Some(block) = client.receive().await? {
            if let ContentBlock::Text(text) = block {
                response.push_str(&text.text);
            }
        }

        Ok::<_, open_agent::Error>(response)
    })
    .await;

    match result {
        Ok(response) => {
            println!("✅ Success after {:?}", start.elapsed());
            println!("Response: {}\n", response.trim());
        }
        Err(e) => {
            println!("❌ Failed: {}\n", e);
        }
    }

    Ok(())
}

// ============================================================================
// Example 3: Concurrent Requests
// ============================================================================
async fn concurrent_requests_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("Example 3: Concurrent Requests");
    println!("{}", "=".repeat(60));
    println!();

    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .temperature(0.7)
        .build()?;

    let questions = [
        "What is 5+5?",
        "What is the capital of France?",
        "Name a primary color",
    ];

    println!("🚀 Running {} queries in parallel...\n", questions.len());

    let start = Instant::now();

    // Create concurrent futures using FuturesUnordered (doesn't require Sync)
    let mut futures = FuturesUnordered::new();
    for (i, question) in questions.iter().enumerate() {
        let options_clone = options.clone();
        let question_owned = question.to_string();

        let future = async move {
            let mut client = Client::new(options_clone)?;
            client.send(&question_owned).await?;

            let mut response = String::new();
            while let Some(block) = client.receive().await? {
                if let ContentBlock::Text(text) = block {
                    response.push_str(&text.text);
                }
            }

            Ok::<(usize, String, String), open_agent::Error>((i, question_owned, response))
        };

        futures.push(future);
    }

    // Collect all results
    let mut results = Vec::new();
    while let Some(result) = futures.next().await {
        results.push(result);
    }

    let elapsed = start.elapsed();

    println!("✅ All queries completed in {:?}\n", elapsed);

    // Print results
    for result in results {
        match result {
            Ok((i, question, response)) => {
                println!("Query {}: {}", i + 1, question);
                println!("Response: {}", response.trim());
                println!();
            }
            Err(e) => {
                println!("❌ Query failed: {}\n", e);
            }
        }
    }

    Ok(())
}

// ============================================================================
// Example 4: Concurrent with Retry
// ============================================================================
async fn concurrent_with_retry_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("Example 4: Concurrent Requests with Retry");
    println!("{}", "=".repeat(60));
    println!();

    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .temperature(0.7)
        .build()?;

    let questions = ["What is 10+10?", "What is the capital of Japan?"];

    let retry_config = RetryConfig::new()
        .with_max_attempts(2)
        .with_initial_delay(Duration::from_millis(500));

    println!(
        "🚀 Running {} queries in parallel with retry...\n",
        questions.len()
    );

    let start = Instant::now();

    // Create concurrent futures with retry using FuturesUnordered
    let mut futures = FuturesUnordered::new();
    for (i, question) in questions.iter().enumerate() {
        let options_clone = options.clone();
        let question_owned = question.to_string();
        let retry_config_clone = retry_config.clone();

        let future = async move {
            // Wrap the operation in retry logic
            let result = retry_with_backoff(retry_config_clone, || async {
                let mut client = Client::new(options_clone.clone())?;
                client.send(&question_owned).await?;

                let mut response = String::new();
                while let Some(block) = client.receive().await? {
                    if let ContentBlock::Text(text) = block {
                        response.push_str(&text.text);
                    }
                }

                Ok::<String, open_agent::Error>(response)
            })
            .await;

            result.map(|response| (i, question_owned, response))
        };

        futures.push(future);
    }

    // Collect all results
    let mut results = Vec::new();
    while let Some(result) = futures.next().await {
        results.push(result);
    }

    let elapsed = start.elapsed();

    println!("✅ All queries completed in {:?}\n", elapsed);

    // Print results
    for result in results {
        match result {
            Ok((i, question, response)) => {
                println!("Query {}: {}", i + 1, question);
                println!("Response: {}", response.trim());
                println!();
            }
            Err(e) => {
                println!("❌ Query failed after retries: {}\n", e);
            }
        }
    }

    Ok(())
}

// ============================================================================
// Example 5: Rate Limiting with Semaphore
// ============================================================================
async fn rate_limiting_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("Example 5: Rate Limiting with Semaphore");
    println!("{}", "=".repeat(60));
    println!();

    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .temperature(0.7)
        .build()?;

    let questions = [
        "What is 1+1?",
        "What is 2+2?",
        "What is 3+3?",
        "What is 4+4?",
        "What is 5+5?",
    ];

    // Limit to 2 concurrent requests at a time
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(2));

    println!(
        "🚦 Running {} queries with max 2 concurrent...\n",
        questions.len()
    );

    let start = Instant::now();

    // Create futures with rate limiting using FuturesUnordered
    let mut futures = FuturesUnordered::new();
    for (i, question) in questions.iter().enumerate() {
        let options_clone = options.clone();
        let question_owned = question.to_string();
        let semaphore_clone = semaphore.clone();

        let future = async move {
            // Acquire permit before executing
            let _permit = semaphore_clone.acquire().await.unwrap();
            println!("  [Starting Query {}]", i + 1);

            let mut client = Client::new(options_clone)?;
            client.send(&question_owned).await?;

            let mut response = String::new();
            while let Some(block) = client.receive().await? {
                if let ContentBlock::Text(text) = block {
                    response.push_str(&text.text);
                }
            }

            println!("  [Completed Query {}]", i + 1);

            Ok::<(usize, String, String), open_agent::Error>((i, question_owned, response))
            // Permit is automatically released when _permit drops
        };

        futures.push(future);
    }

    // Collect all results
    let mut results = Vec::new();
    while let Some(result) = futures.next().await {
        results.push(result);
    }

    let elapsed = start.elapsed();

    println!("\n✅ All queries completed in {:?}\n", elapsed);

    // Print results
    for result in results {
        match result {
            Ok((i, question, response)) => {
                println!("Query {}: {} => {}", i + 1, question, response.trim());
            }
            Err(e) => {
                println!("❌ Query failed: {}\n", e);
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(60));
    println!("ADVANCED PATTERNS DEMO");
    println!("{}", "=".repeat(60));
    println!();

    println!("This demo requires Ollama running at http://localhost:11434");
    println!("with a model loaded (e.g., qwen3:8b)\n");

    // Run examples
    if let Err(e) = retry_example().await {
        eprintln!("Retry example error: {}", e);
    }
    tokio::time::sleep(Duration::from_secs(1)).await;

    if let Err(e) = conditional_retry_example().await {
        eprintln!("Conditional retry example error: {}", e);
    }
    tokio::time::sleep(Duration::from_secs(1)).await;

    if let Err(e) = concurrent_requests_example().await {
        eprintln!("Concurrent requests example error: {}", e);
    }
    tokio::time::sleep(Duration::from_secs(1)).await;

    if let Err(e) = concurrent_with_retry_example().await {
        eprintln!("Concurrent with retry example error: {}", e);
    }
    tokio::time::sleep(Duration::from_secs(1)).await;

    if let Err(e) = rate_limiting_example().await {
        eprintln!("Rate limiting example error: {}", e);
    }

    println!("\n{}", "=".repeat(60));
    println!("All examples completed!");
    println!("{}", "=".repeat(60));

    Ok(())
}
