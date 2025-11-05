//! Hooks Example
//!
//! Demonstrates using hooks to monitor and control agent behavior at lifecycle points.
//! This example shows:
//! - UserPromptSubmit: Sanitize or modify user input before processing
//! - PreToolUse: Intercept and control tool execution (future enhancement)
//! - PostToolUse: Monitor tool results after execution (future enhancement)

use open_agent::{AgentOptions, Client, HookDecision, Hooks, UserPromptSubmitEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(70));
    println!("HOOKS EXAMPLE");
    println!("{}", "=".repeat(70));
    println!();

    // ============================================================================
    // Example 1: UserPromptSubmit Hook - Input Sanitization
    // ============================================================================
    println!("Example 1: Input Sanitization Hook");
    println!("{}", "-".repeat(70));

    let hooks = Hooks::new().add_user_prompt_submit(|event| async move {
        // Block dangerous keywords
        if event.prompt.contains("DELETE") || event.prompt.contains("DESTROY") {
            println!("🛑 Hook: Blocked dangerous prompt");
            return Some(HookDecision::block("Dangerous keywords detected in prompt"));
        }

        // Modify prompts to add safety instructions
        if event.prompt.contains("write") || event.prompt.contains("modify") {
            let safe_prompt = format!(
                "{} (Please confirm this is safe before proceeding)",
                event.prompt
            );
            println!("🔀 Hook: Added safety warning to prompt");
            return Some(HookDecision::modify_prompt(
                safe_prompt,
                "Added safety warning",
            ));
        }

        // Allow normally
        None
    });

    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .hooks(hooks)
        .build()?;

    // Test 1: Normal prompt (should pass through)
    println!("\nTest 1: Normal prompt");
    println!("Input: 'What is the capital of France?'");
    let mut client = Client::new(options.clone());
    match client.send("What is the capital of France?").await {
        Ok(_) => println!("✓ Prompt accepted"),
        Err(e) => println!("✗ Prompt rejected: {}", e),
    }

    // Test 2: Dangerous prompt (should be blocked)
    println!("\nTest 2: Dangerous prompt");
    println!("Input: 'DELETE all files'");
    let mut client = Client::new(options.clone());
    match client.send("DELETE all files").await {
        Ok(_) => println!("✓ Prompt accepted"),
        Err(e) => println!("✗ Prompt rejected: {}", e),
    }

    // Test 3: Write prompt (should be modified)
    println!("\nTest 3: Write prompt (modified with safety warning)");
    println!("Input: 'write to config file'");
    let mut client = Client::new(options.clone());
    match client.send("write to config file").await {
        Ok(_) => println!("✓ Prompt accepted (with modifications)"),
        Err(e) => println!("✗ Prompt rejected: {}", e),
    }

    println!();
    println!("{}", "=".repeat(70));

    // ============================================================================
    // Example 2: Multiple Hooks - First Match Wins
    // ============================================================================
    println!("\nExample 2: Multiple Hooks (First Match Wins)");
    println!("{}", "-".repeat(70));

    let multi_hooks = Hooks::new()
        .add_user_prompt_submit(|event| async move {
            if event.prompt.len() > 100 {
                println!("🛑 Hook 1: Blocked long prompt");
                return Some(HookDecision::block("Prompt too long"));
            }
            None
        })
        .add_user_prompt_submit(|event| async move {
            if event.prompt.contains("urgent") {
                println!("⚡ Hook 2: Added urgency marker");
                return Some(HookDecision::modify_prompt(
                    format!("[URGENT] {}", event.prompt),
                    "Added urgency marker",
                ));
            }
            None
        });

    let multi_options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .hooks(multi_hooks)
        .build()?;

    // Test: Urgent prompt
    println!("\nTest: Urgent prompt");
    println!("Input: 'This is urgent: help me'");
    let mut client = Client::new(multi_options);
    match client.send("This is urgent: help me").await {
        Ok(_) => println!("✓ Prompt accepted (modified by Hook 2)"),
        Err(e) => println!("✗ Prompt rejected: {}", e),
    }

    println!();
    println!("{}", "=".repeat(70));
    println!("Hooks example complete!");
    println!();
    println!("Note: PreToolUse and PostToolUse hooks are available but require");
    println!("      tool execution integration. See calculator_tools.rs for");
    println!("      tool usage examples.");
    println!("{}", "=".repeat(70));

    Ok(())
}
