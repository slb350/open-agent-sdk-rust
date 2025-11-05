# Open Agent SDK (Rust)

> Build production-ready AI agents in Rust using your own hardware

[![Crates.io](https://img.shields.io/crates/v/open-agent-sdk.svg)](https://crates.io/crates/open-agent-sdk)
[![Documentation](https://docs.rs/open-agent-sdk/badge.svg)](https://docs.rs/open-agent-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Rust implementation of the Open Agent SDK, providing a clean, streaming API for working with OpenAI-compatible local model servers like LMStudio, Ollama, llama.cpp, and vLLM.

## ✨ Features

- **🚀 Production Ready**: 100% feature parity with Python SDK, 85+ comprehensive tests
- **⚡ Streaming First**: Built on Tokio for high-performance async I/O
- **🤖 Auto-Execution Mode**: Automatic tool calling with seamless iteration
- **🛠️ Tool System**: Function calling with automatic schema generation
- **🪝 Lifecycle Hooks**: Intercept and modify prompts, tool calls, and responses
- **🔄 Retry Logic**: Exponential backoff with jitter for transient failures
- **🎯 Context Management**: Token estimation and smart message truncation
- **⏸️ Interrupt Support**: Cancel long-running operations gracefully
- **📊 10 Examples**: From simple queries to production agents
- **🧪 Comprehensive Tests**: 57 unit tests + 28 integration tests

## 📦 Installation

```toml
[dependencies]
open-agent-sdk = "0.1.0"
tokio = { version = "1", features = ["full"] }
```

## 🚀 Quick Start

### Simple Query

```rust
use open_agent::{query, AgentOptions, ContentBlock};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant")
        .model("qwen2.5-32b-instruct")
        .base_url("http://localhost:1234/v1")
        .build()?;

    let mut stream = query("What's the capital of France?", &options).await?;

    while let Some(block) = stream.next().await {
        match block? {
            ContentBlock::Text(text) => print!("{}", text.text),
            _ => {}
        }
    }

    Ok(())
}
```

### Multi-Turn Conversation with Client

```rust
use open_agent::{Client, AgentOptions, ContentBlock};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant")
        .model("qwen2.5-32b-instruct")
        .base_url("http://localhost:1234/v1")
        .build()?;

    let mut client = Client::new(options);

    // First message
    client.send("Hello!").await?;
    while let Some(block) = client.receive().await {
        if let ContentBlock::Text(text) = block? {
            print!("{}", text.text);
        }
    }

    // Continue conversation (client maintains history)
    client.send("Tell me a joke").await?;
    while let Some(block) = client.receive().await {
        if let ContentBlock::Text(text) = block? {
            print!("{}", text.text);
        }
    }

    Ok(())
}
```

### Function Calling with Tools

```rust
use open_agent::{tool, Client, AgentOptions, ContentBlock};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define a tool
    let add_tool = tool("add", "Add two numbers")
        .param("a", "number", "First number")
        .param("b", "number", "Second number")
        .handler(|args| {
            Box::pin(async move {
                let a = args["a"].as_f64().unwrap_or(0.0);
                let b = args["b"].as_f64().unwrap_or(0.0);
                Ok(json!({"result": a + b}))
            })
        })
        .build();

    // Configure agent with tools
    let options = AgentOptions::builder()
        .system_prompt("You are a calculator assistant")
        .model("qwen2.5-32b-instruct")
        .base_url("http://localhost:1234/v1")
        .tool(add_tool.clone())
        .build()?;

    let mut client = Client::new(options);
    client.send("What is 25 plus 17?").await?;

    // Handle tool calls
    while let Some(block) = client.receive().await {
        match block? {
            ContentBlock::ToolUse(tool_use) => {
                println!("Calling tool: {}", tool_use.name);
                let result = add_tool.execute(tool_use.input).await?;
                client.add_tool_result(&tool_use.id, result, Some(&tool_use.name));
            }
            ContentBlock::Text(text) => {
                println!("Assistant: {}", text.text);
            }
        }
    }

    Ok(())
}
```

### Auto-Execution Mode

Let the SDK handle tool execution automatically - perfect for production agents:

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

    let multiply_tool = tool("multiply", "Multiply two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!({"result": a * b}))
        });

    // Enable auto-execution
    let options = AgentOptions::builder()
        .system_prompt("You are a calculator assistant")
        .model("qwen2.5-32b-instruct")
        .base_url("http://localhost:1234/v1")
        .tool(add_tool)
        .tool(multiply_tool)
        .auto_execute_tools(true)  // ← Enable auto-execution
        .max_tool_iterations(10)
        .build()?;

    let mut client = Client::new(options);

    // Tools are executed automatically - you only get the final answer!
    client.send("What's (5 + 3) * 2?").await?;

    while let Some(block) = client.receive().await {
        if let ContentBlock::Text(text) = block? {
            println!("{}", text.text);  // Just prints: "16"
        }
    }

    Ok(())
}
```

**What happens behind the scenes:**
1. Model calls `add(5, 3)` → Returns 8
2. Model calls `multiply(8, 2)` → Returns 16
3. Model responds with final text: "16"
4. You receive only the final text response!

### Retry with Exponential Backoff

```rust
use open_agent::{Client, AgentOptions};
use open_agent::retry::{retry_with_backoff, RetryConfig};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = AgentOptions::builder()
        .model("qwen2.5-32b-instruct")
        .base_url("http://localhost:1234/v1")
        .build()?;

    let retry_config = RetryConfig::new()
        .with_max_attempts(3)
        .with_initial_delay(Duration::from_secs(1))
        .with_backoff_multiplier(2.0);

    let response = retry_with_backoff(retry_config, || async {
        let mut client = Client::new(options.clone());
        client.send("What is 2+2?").await?;

        let mut result = String::new();
        while let Some(block) = client.receive().await {
            if let open_agent::ContentBlock::Text(text) = block? {
                result.push_str(&text.text);
            }
        }
        Ok::<_, open_agent::Error>(result)
    }).await?;

    println!("Response: {}", response);
    Ok(())
}
```

## 📚 Examples

The SDK includes 10 comprehensive examples:

1. **simple_query** - Basic streaming query
2. **calculator_tools** - Function calling with tools (manual mode)
3. **auto_execution_demo** - ✨ Automatic tool execution demo
4. **multi_tool_agent** - ✨ Production agent with 5 tools, hooks, and auto-execution
5. **hooks_example** - Input sanitization and prompt modification
6. **context_management** - Token estimation and message truncation
7. **interrupt_demo** - Canceling operations gracefully
8. **git_commit_agent** - Production agent that writes commit messages
9. **log_analyzer_agent** - Production agent for log analysis
10. **advanced_patterns** - Retry logic and concurrent requests

Run examples with:

```bash
cargo run --example simple_query
cargo run --example auto_execution_demo
cargo run --example multi_tool_agent
```

## 🏗️ Architecture

The SDK is built around a few core concepts:

- **AgentOptions**: Configuration builder for model settings, tools, hooks
- **Client**: Stateful conversation manager with message history
- **Tool**: Function definitions that the LLM can call
- **Hooks**: Lifecycle events for monitoring and control
- **Retry**: Utilities for handling transient failures

## 🧪 Testing

The SDK has extensive test coverage:

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
- Tests for error conditions, edge cases, and serialization
- All features working together validated

## 🤝 Compatible Servers

Works with any OpenAI-compatible API:

- **LMStudio** - `http://localhost:1234/v1`
- **Ollama** - `http://localhost:11434/v1`
- **llama.cpp** - `http://localhost:8080/v1`
- **vLLM** - `http://localhost:8000/v1`
- **LocalAI** - Custom endpoint
- **OpenAI** - Direct API access

## 📖 Documentation

- [API Documentation](https://docs.rs/open-agent-sdk)
- [Python SDK](https://github.com/slb350/open-agent-sdk) - Reference implementation
- [Examples](examples/) - Comprehensive usage examples

## 🤝 Contributing

This is a reference implementation developed in a separate branch. Contributions welcome!

##  License

MIT License - see LICENSE for details

##  Acknowledgments

Rust port of [open-agent-sdk](https://github.com/slb350/open-agent-sdk) Python library.

## 💡 Why Rust?

- **Performance**: Zero-cost abstractions, no GC pauses
- **Safety**: Memory safety without runtime overhead
- **Concurrency**: Fearless concurrency with compile-time guarantees
- **Production Ready**: Type safety, error handling, comprehensive testing
- **Small Binaries**: Standalone executables under 10MB

---

**Made with ❤️ for developers who want to run AI agents on their own hardware**
