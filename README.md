# Open Agent SDK (Rust)

> Build production-ready AI agents in Rust using your own hardware

**What you can build:**

- **Copy editors** that analyze manuscripts and track writing patterns
- **Git commit generators** that write meaningful commit messages
- **Market analyzers** that research competitors and summarize findings
- **Code reviewers**, **data analysts**, **research assistants**, and more

**Why local?**

- **No API costs** - use your hardware, not OpenAI's
- **Privacy** - your data never leaves your machine
- **Control** - pick your model (Qwen, Llama, Mistral, etc.)

**How fast?**
From zero to working agent in under 5 minutes. Rust-native performance (zero-cost abstractions, no GC), fearless concurrency, and production-ready quality with 85+ tests.

[![Crates.io](https://img.shields.io/crates/v/open-agent-sdk.svg)](https://crates.io/crates/open-agent-sdk)
[![Documentation](https://docs.rs/open-agent-sdk/badge.svg)](https://docs.rs/open-agent-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Open Agent SDK (Rust) provides a clean, streaming API for working with OpenAI-compatible local model servers. 100% feature parity with the Python SDK—complete with streaming, tool call aggregation, hooks, and automatic tool execution—built on Tokio for high-performance async I/O.

## Supported Providers

### Supported (OpenAI-Compatible Endpoints)

- **LM Studio** - `http://localhost:1234/v1`
- **Ollama** - `http://localhost:11434/v1`
- **llama.cpp server** - OpenAI-compatible mode
- **vLLM** - OpenAI-compatible API
- **Text Generation WebUI** - OpenAI extension
- **Any OpenAI-compatible local endpoint**
- **Local gateways proxying cloud models** - e.g., Ollama or custom gateways that route to cloud providers

### Not Supported (Use Official SDKs)

- **Claude/OpenAI direct** - Use their official SDKs, unless you proxy through a local OpenAI-compatible gateway
- **Cloud provider SDKs** - Bedrock, Vertex, Azure, etc. (proxied via local gateway is fine)

## Quick Start

### Installation

```toml
[dependencies]
open-agent-sdk = "0.1.0"
tokio = { version = "1", features = ["full"] }
futures = "0.3"
serde_json = "1.0"
```

For development:

```bash
git clone https://github.com/slb350/open-agent-sdk-rust.git
cd open-agent-sdk-rust
cargo build
```

### Simple Query (LM Studio)

```rust
use open_agent::{query, AgentOptions, ContentBlock};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = AgentOptions::builder()
        .system_prompt("You are a professional copy editor")
        .model("qwen2.5-32b-instruct")
        .base_url("http://localhost:1234/v1")
        .temperature(0.1)
        .build()?;

    let mut stream = query("Analyze this text...", &options).await?;

    while let Some(block) = stream.next().await {
        match block? {
            ContentBlock::Text(text) => print!("{}", text.text),
            _ => {}
        }
    }

    Ok(())
}
```

### Multi-Turn Conversation (Ollama)

```rust
use open_agent::{Client, AgentOptions, ContentBlock};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .build()?;

    let mut client = Client::new(options)?;

    client.send("What's the capital of France?").await?;

    while let Some(block) = client.receive().await {
        match block? {
            ContentBlock::Text(text) => {
                println!("Assistant: {}", text.text);
            }
            ContentBlock::ToolUse(tool_use) => {
                println!("Tool used: {}", tool_use.name);
                // Execute tool and add result
                // client.add_tool_result(&tool_use.id, result, Some(&tool_use.name));
            }
            _ => {}
        }
    }

    Ok(())
}
```

### Function Calling with Tools

Define tools using the builder pattern for clean, type-safe function calling:

```rust
use open_agent::{tool, Client, AgentOptions, ContentBlock};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define tools
    let add_tool = tool("add", "Add two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!({"result": a + b}))
        });

    // Enable automatic tool execution (recommended)
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant with access to tools.")
        .model("qwen2.5-32b-instruct")
        .base_url("http://localhost:1234/v1")
        .tool(add_tool)
        .auto_execute_tools(true)      // Tools execute automatically
        .max_tool_iterations(10)       // Safety limit for tool loops
        .build()?;

    let mut client = Client::new(options)?;
    client.send("What's 25 + 17?").await?;

    // Simply iterate - tools execute automatically!
    while let Some(block) = client.receive().await {
        match block? {
            ContentBlock::Text(text) => {
                println!("Response: {}", text.text);
            }
            _ => {}
        }
    }

    Ok(())
}
```

### Advanced: Manual Tool Execution

For custom execution logic or result interception:

```rust
// Disable auto-execution
let options = AgentOptions::builder()
    .system_prompt("You are a helpful assistant with access to tools.")
    .model("qwen2.5-32b-instruct")
    .base_url("http://localhost:1234/v1")
    .tool(add_tool.clone())
    .auto_execute_tools(false)  // Manual mode
    .build()?;

let mut client = Client::new(options)?;
client.send("What's 25 + 17?").await?;

while let Some(block) = client.receive().await {
    match block? {
        ContentBlock::ToolUse(tool_use) => {
            // You execute the tool manually
            let result = add_tool.execute(tool_use.input).await?;

            // Return result to agent
            client.add_tool_result(&tool_use.id, result, Some(&tool_use.name));

            // Continue conversation
            client.send("").await?;
        }
        ContentBlock::Text(text) => {
            println!("{}", text.text);
        }
        _ => {}
    }
}
```

**Key Features:**

- **Automatic execution** - Tools run automatically with safety limits
- **Type-safe schemas** - Automatic JSON schema generation from parameters
- **OpenAI-compatible** - Works with any OpenAI function calling endpoint
- **Clean builder API** - Fluent API for tool definition
- **Hook integration** - PreToolUse/PostToolUse hooks work in both modes

See `examples/calculator_tools.rs` and `examples/auto_execution_demo.rs` for complete examples.

## Context Management

Local models have fixed context windows (typically 8k-32k tokens). The SDK provides utilities for manual history management—no silent mutations, you stay in control.

### Token Estimation & Truncation

```rust
use open_agent::{Client, AgentOptions, estimate_tokens, truncate_messages};

let mut client = Client::new(options)?;

// Long conversation...
for i in 0..50 {
    client.send(&format!("Question {}", i)).await?;
    while let Some(block) = client.receive().await {
        // Process blocks
    }
}

// Check token usage
let tokens = estimate_tokens(client.history());
println!("Context size: ~{} tokens", tokens);

// Manually truncate when needed
if tokens > 28000 {
    let truncated = truncate_messages(client.history(), 10, true);
    *client.history_mut() = truncated;
}
```

### Recommended Patterns

**1. Stateless Agents** (Best for single-task agents):

```rust
// Process each task independently - no history accumulation
for task in tasks {
    let mut client = Client::new(options.clone());
    client.send(&task).await?;
    // Client dropped, fresh context for next task
}
```

**2. Manual Truncation** (At natural breakpoints):

```rust
use open_agent::truncate_messages;

let mut client = Client::new(options)?;
for task in tasks {
    client.send(&task).await?;
    // Truncate after each major task
    let truncated = truncate_messages(client.history(), 5, false);
    *client.history_mut() = truncated;
}
```

**3. External Memory** (RAG-lite for research agents):

```rust
// Store important facts in database, keep conversation context small
let mut database = HashMap::new();
let mut client = Client::new(options)?;

client.send("Research topic X").await?;
// Save response to database
database.insert("topic_x", extract_facts(&response));

// Clear history, query database when needed
let truncated = truncate_messages(client.history(), 0, false);
*client.history_mut() = truncated;
```

### Why Manual?

The SDK **intentionally** does not auto-compact history because:

- **Domain-specific needs**: Copy editors need different strategies than research agents
- **Token accuracy varies**: Each model family has different tokenizers
- **Risk of breaking context**: Silently removing messages could break tool chains
- **Natural limits exist**: Compaction doesn't bypass model context windows

See `examples/context_management.rs` for complete patterns and usage.

## Lifecycle Hooks

Monitor and control agent behavior at key execution points with zero-cost Rust hooks.

### Quick Example

```rust
use open_agent::{
    AgentOptions, Client, Hooks,
    PreToolUseEvent, PostToolUseEvent,
    HookDecision,
};

// Security gate - block dangerous operations
let hooks = Hooks::new()
    .add_pre_tool_use(|event| async move {
        if event.tool_name == "delete_file" {
            return Some(HookDecision::block("Delete operations require approval"));
        }
        Some(HookDecision::continue_())
    })
    .add_post_tool_use(|event| async move {
        // Audit logger - track all tool executions
        println!("Tool executed: {} -> {:?}", event.tool_name, event.tool_result);
        None
    });

// Register hooks in AgentOptions
let options = AgentOptions::builder()
    .system_prompt("You are a helpful assistant")
    .model("qwen2.5-32b-instruct")
    .base_url("http://localhost:1234/v1")
    .hooks(hooks)
    .build()?;

let mut client = Client::new(options)?;
```

### Hook Types

**PreToolUse** - Fires before tool execution

- **Block operations**: Return `Some(HookDecision::block(reason))`
- **Modify inputs**: Return `Some(HookDecision::modify_input(json!({}), reason))`
- **Allow**: Return `Some(HookDecision::continue_())`

**PostToolUse** - Fires after tool result added to history

- **Observational** (tool already executed)
- Use for audit logging, metrics, result validation
- Return `None` or `Some(HookDecision::...)`

**UserPromptSubmit** - Fires before sending prompt to API

- **Block prompts**: Return `Some(HookDecision::block(reason))`
- **Modify prompts**: Return `Some(HookDecision::modify_prompt(text, reason))`
- **Allow**: Return `Some(HookDecision::continue_())`

### Common Patterns

#### Pattern 1: Redirect to Sandbox

```rust
hooks.add_pre_tool_use(|event| async move {
    if event.tool_name == "file_writer" {
        let path = event.tool_input.get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if !path.starts_with("/tmp/") {
            let safe_path = format!("/tmp/sandbox/{}", path.trim_start_matches('/'));
            let mut modified = event.tool_input.clone();
            modified["path"] = json!(safe_path);
            return Some(HookDecision::modify_input(modified, "Redirected to sandbox"));
        }
    }
    Some(HookDecision::continue_())
})
```

#### Pattern 2: Compliance Audit Log

```rust
let audit_log = Arc::new(Mutex::new(Vec::new()));
let log_clone = audit_log.clone();

hooks.add_post_tool_use(move |event| {
    let log = log_clone.clone();
    async move {
        log.lock().unwrap().push(format!(
            "[{}] {} -> {:?}",
            chrono::Utc::now(),
            event.tool_name,
            event.tool_result
        ));
        None
    }
})
```

### Hook Execution Flow

- Hooks run **sequentially** in the order registered
- **First non-None decision wins** (short-circuit behavior)
- Hooks run **inline on async runtime** (spawn tasks for heavy work)
- Works with both **Client** and **query()** function

See `examples/hooks_example.rs` and `examples/multi_tool_agent.rs` for comprehensive patterns.

## Interrupt Capability

Cancel long-running operations cleanly without corrupting client state. Perfect for timeouts, user cancellations, or conditional interruptions.

### Interrupt Quick Example

```rust
use open_agent::{Client, AgentOptions};
use tokio::time::{timeout, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model("qwen2.5-32b-instruct")
        .base_url("http://localhost:1234/v1")
        .build()?;

    let mut client = Client::new(options)?;
    client.send("Write a detailed 1000-word essay...").await?;

    // Timeout after 5 seconds
    match timeout(Duration::from_secs(5), async {
        while let Some(block) = client.receive().await {
            // Process blocks
        }
    }).await {
        Ok(_) => println!("Completed"),
        Err(_) => {
            client.interrupt();  // Clean cancellation
            println!("Operation timed out!");
        }
    }

    // Client is still usable after interrupt
    client.send("Short question?").await?;
    // Continue using client...

    Ok(())
}
```

### Common Interrupt Patterns

#### 1. Conditional Interruption

```rust
let mut full_text = String::new();
while let Some(block) = client.receive().await {
    if let ContentBlock::Text(text) = block? {
        full_text.push_str(&text.text);
        if full_text.contains("error") {
            client.interrupt();
            break;
        }
    }
}
```

#### 2. Concurrent Cancellation

```rust
use tokio::select;

let stream_task = async {
    while let Some(block) = client.receive().await {
        // Process blocks
    }
};

let cancel_task = async {
    tokio::time::sleep(Duration::from_secs(2)).await;
    client.interrupt();
};

tokio::select! {
    _ = stream_task => println!("Completed"),
    _ = cancel_task => println!("Cancelled"),
}
```

### How It Works

When you call `client.interrupt()`:

1. **Active stream closure** - HTTP stream closed immediately (not just a flag)
2. **Clean state** - Client remains in valid state for reuse
3. **Partial output** - Text blocks flushed to history, incomplete tools skipped
4. **Idempotent** - Safe to call multiple times
5. **Thread-safe** - Can be called from separate async tasks

See `examples/interrupt_demo.rs` for comprehensive patterns.

## Practical Examples

We've included production-ready agents that demonstrate real-world usage:

### Git Commit Agent

**[examples/git_commit_agent.rs](examples/git_commit_agent.rs)**

Analyzes your staged git changes and writes professional commit messages following conventional commit format.

```bash
# Stage your changes
git add .

# Run the agent
cargo run --example git_commit_agent

# Output:
# Found staged changes in 3 file(s)
# Analyzing changes and generating commit message...
#
# Suggested commit message:
# feat(auth): Add OAuth2 integration with refresh tokens
#
# - Implement token refresh mechanism
# - Add secure cookie storage for tokens
# - Update login flow to support OAuth2 providers
```

**Features:**

- Analyzes diff to determine commit type (feat/fix/docs/etc)
- Writes clear, descriptive commit messages
- Follows conventional commit standards

### Log Analyzer Agent

**[examples/log_analyzer_agent.rs](examples/log_analyzer_agent.rs)**

Intelligently analyzes application logs to identify patterns, errors, and provide actionable insights.

```bash
# Analyze a log file
cargo run --example log_analyzer_agent -- /var/log/app.log
```

**Features:**

- Automatic error pattern detection
- Time-based analysis (peak error times)
- Root cause suggestions
- Supports multiple log formats

### Why These Examples?

These agents demonstrate:

- **Practical Value**: Solve real problems developers face daily
- **Tool Integration**: Show how to integrate with system commands (git, file I/O)
- **Structured Output**: Parse and format LLM responses for actionable results
- **Privacy-First**: Keep your code and logs local while getting AI assistance

## Why Not Just Use OpenAI Client?

**Without open-agent-sdk** (raw reqwest):

```rust
use reqwest::Client;

let client = Client::new();
let response = client
    .post("http://localhost:1234/v1/chat/completions")
    .json(&json!({
        "model": "qwen2.5-32b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": true
    }))
    .send()
    .await?;

// Complex parsing of SSE chunks
// Extract delta content
// Handle tool calls manually
// Track conversation state yourself
```

**With open-agent-sdk**:

```rust
use open_agent::{query, AgentOptions};

let options = AgentOptions::builder()
    .system_prompt(system_prompt)
    .model("qwen2.5-32b-instruct")
    .base_url("http://localhost:1234/v1")
    .build()?;

let mut stream = query(user_prompt, &options).await?;
// Clean message types (TextBlock, ToolUseBlock)
// Automatic streaming and tool call handling
```

**Value**: Familiar patterns + Less boilerplate + Rust performance

## Why Rust?

**Performance**: Zero-cost abstractions mean no runtime overhead. Streaming responses with Tokio delivers throughput comparable to C/C++ while maintaining memory safety.

**Safety**: Compile-time guarantees prevent data races, null pointer dereferences, and buffer overflows. Your agents won't crash from memory issues.

**Concurrency**: Fearless concurrency with `async`/`await` lets you run multiple agents or handle hundreds of concurrent requests without fear of race conditions.

**Production Ready**: Strong type system catches bugs at compile time. Comprehensive error handling with `Result` types. No surprises in production.

**Small Binaries**: Standalone executables under 10MB. Deploy anywhere without runtime dependencies.

## API Reference

### AgentOptions

```rust
AgentOptions::builder()
    .system_prompt(str)                  // System prompt
    .model(str)                          // Model name (required)
    .base_url(str)                       // OpenAI-compatible endpoint (required)
    .tool(Tool)                          // Add tools for function calling
    .hooks(Hooks)                        // Lifecycle hooks for monitoring/control
    .auto_execute_tools(bool)            // Enable automatic tool execution
    .max_tool_iterations(usize)          // Max tool calls per query in auto mode
    .max_tokens(Option<u32>)             // Tokens to generate (None = provider default)
    .temperature(f32)                    // Sampling temperature
    .timeout(u64)                        // Request timeout in seconds
    .api_key(str)                        // API key (default: "not-needed")
    .build()?
```

### query()

Simple single-turn query function.

```rust
pub async fn query(prompt: &str, options: &AgentOptions)
    -> Result<ContentStream>
```

Returns a stream yielding `ContentBlock` items.

### Client

Multi-turn conversation client with tool monitoring.

```rust
let mut client = Client::new(options)?;
client.send(prompt).await?;

while let Some(block) = client.receive().await {
    // Process ContentBlock items
}
```

### Message Types

- `ContentBlock::Text(TextBlock)` - Text content from model
- `ContentBlock::ToolUse(ToolUseBlock)` - Tool calls from model
- `ContentBlock::ToolResult(ToolResultBlock)` - Tool execution results

### Tool System

```rust
use open_agent::tool;

let my_tool = tool("name", "description")
    .param("param_name", "type")
    .build(|args| async move {
        // Tool implementation
        Ok(json!({"result": value}))
    });
```

## Recommended Models

**Local models** (LM Studio, Ollama, llama.cpp):

- **GPT-OSS-120B** - Best in class for speed and quality
- **Qwen 3 30B** - Excellent instruction following, good for most tasks
- **GPT-OSS-20B** - Solid all-around performance
- **Mistral 7B** - Fast and efficient for simple agents

**Cloud-proxied via local gateway**:

- **kimi-k2:1t-cloud** - Tested and working via Ollama gateway
- **deepseek-v3.1:671b-cloud** - High-quality reasoning model
- **qwen3-coder:480b-cloud** - Code-focused models

## Project Structure

```text
open-agent-sdk-rust/
├── src/
│   ├── client.rs          # query() and Client implementation
│   ├── config.rs          # Configuration builder
│   ├── context.rs         # Token estimation and truncation
│   ├── error.rs           # Error types
│   ├── hooks.rs           # Lifecycle hooks
│   ├── lib.rs             # Public exports
│   ├── retry.rs           # Retry logic with exponential backoff
│   ├── tools.rs           # Tool system
│   ├── types.rs           # Core types (AgentOptions, ContentBlock, etc.)
│   └── utils.rs           # SSE parsing and tool call aggregation
├── examples/
│   ├── simple_query.rs              # Basic streaming query
│   ├── calculator_tools.rs          # Function calling (manual mode)
│   ├── auto_execution_demo.rs       # Automatic tool execution
│   ├── multi_tool_agent.rs          # Production agent with 5 tools and hooks
│   ├── hooks_example.rs             # Lifecycle hooks patterns
│   ├── context_management.rs        # Context management patterns
│   ├── interrupt_demo.rs            # Interrupt capability patterns
│   ├── git_commit_agent.rs          # Production: Git commit generator
│   ├── log_analyzer_agent.rs        # Production: Log analyzer
│   └── advanced_patterns.rs         # Retry logic and concurrent requests
├── tests/
│   ├── integration_tests.rs
│   ├── hooks_integration_test.rs    # Hooks integration tests
│   ├── auto_execution_test.rs       # Auto-execution tests
│   └── advanced_integration_test.rs # Advanced integration tests
├── Cargo.toml
└── README.md
```

## Examples

### Production Agents

- **`git_commit_agent.rs`** – Analyzes git diffs and writes professional commit messages
- **`log_analyzer_agent.rs`** – Parses logs, finds patterns, suggests fixes
- **`multi_tool_agent.rs`** – Complete production setup with 5 tools, hooks, and auto-execution

### Core SDK Usage

- `simple_query.rs` – Minimal streaming query (simplest quickstart)
- `calculator_tools.rs` – Manual tool execution pattern
- `auto_execution_demo.rs` – Automatic tool execution pattern
- `hooks_example.rs` – Lifecycle hooks patterns (security gates, audit logging)
- `context_management.rs` – Manual history management patterns
- `interrupt_demo.rs` – Interrupt capability patterns (timeout, conditional, concurrent)
- `advanced_patterns.rs` – Retry logic and concurrent request handling

## Documentation

- [API Documentation](https://docs.rs/open-agent-sdk)
- [Python SDK](https://github.com/slb350/open-agent-sdk) - Reference implementation
- [Examples](examples/) - Comprehensive usage examples

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_agent_options_builder
```

**Test Coverage:**

- 57 unit tests across 10 modules
- 28 integration tests
  - 6 hooks integration tests
  - 13 auto-execution tests
  - 9 advanced integration tests

## Requirements

- Rust 1.85+
- Tokio 1.0+ (async runtime)
- serde, serde_json (serialization)
- reqwest (HTTP client)
- futures (async streams)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Rust port of [open-agent-sdk](https://github.com/slb350/open-agent-sdk) Python library
- API design inspired by claude-agent-sdk
- Built for local/open-source LLM enthusiasts

---

**Status**: v0.1.0 Published - 100% feature parity with Python SDK, production-ready

Star this repo if you're building AI agents with local models in Rust!
