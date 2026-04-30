# Open Agent SDK (Rust)

## Project Description

A Rust SDK (v0.6.3) for building AI agents with local or cloud LLMs via OpenAI-compatible endpoints. Feature parity with the Python SDK — streaming, tools, hooks, auto-execution, vision, context management, and interrupt capability. Published to crates.io as `open-agent-sdk`.

## Repository Structure

```
open-agent-sdk-rust/
├── src/
│   ├── lib.rs           # Public API, re-exports, prelude module
│   ├── client.rs        # query() function + Client struct (streaming, multi-turn)
│   ├── types.rs         # AgentOptions builder, ContentBlock, AssistantMessage, ImageBlock
│   ├── tools.rs         # Tool definition, schema conversion, registry
│   ├── hooks.rs         # Lifecycle hooks (PreToolUse, PostToolUse, UserPromptSubmit)
│   ├── context.rs       # Token estimation + message truncation utilities
│   ├── config.rs        # Environment variable config helpers
│   ├── error.rs         # Error types
│   ├── retry.rs         # Retry policy (pub module)
│   └── utils.rs         # Shared internal utilities
├── examples/            # Runnable examples
│   ├── simple_query.rs
│   ├── calculator_tools.rs
│   ├── auto_execution_demo.rs
│   ├── multi_tool_agent.rs
│   ├── hooks_example.rs
│   ├── context_management.rs
│   ├── interrupt_demo.rs
│   ├── git_commit_agent.rs
│   ├── log_analyzer_agent.rs
│   ├── advanced_patterns.rs
│   ├── vision_example.rs
│   ├── vision_api_demo.rs
│   └── test_tool_serialization.rs
├── tests/               # Integration tests
│   ├── integration_tests.rs
│   ├── advanced_integration_test.rs
│   ├── auto_execution_test.rs
│   ├── backward_compatibility_test.rs
│   ├── client_image_serialization_test.rs
│   ├── debug_logging_test.rs
│   ├── defensive_validation_test.rs
│   ├── edge_cases_test.rs
│   ├── hooks_integration_test.rs
│   ├── image_serialization_test.rs
│   ├── security_bypass_test.rs
│   ├── send_message_test.rs
│   └── tool_call_content_test.rs
├── Cargo.toml           # version = "0.6.3"
├── Cargo.lock
└── CHANGELOG.md
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Rust 1.85+ |
| **Async** | Tokio |
| **HTTP** | reqwest 0.12.28+ |
| **Serialization** | serde / serde_json |
| **Streams** | futures 0.3.32+ |
| **Logging** | log 0.4.29+ |

## Common Commands

```bash
# Build
cargo build

# Run tests (unit + integration + doctests)
cargo test

# Run with output
cargo test -- --nocapture

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt

# Run an example
cargo run --example simple_query
cargo run --example git_commit_agent

# Build with version metadata
cargo build --release

# Security audit
cargo audit
```

## Public API (prelude)

```rust
use open_agent::prelude::*;
// or explicitly:
use open_agent::{query, Client, AgentOptions, ContentBlock};
use open_agent::types::{TextBlock, ToolUseBlock, ImageBlock};
use open_agent::tools::Tool;
use open_agent::hooks::{PreToolUseEvent, PostToolUseEvent, HookDecision};
```

## Key Features

### Streaming Query
```rust
let options = AgentOptions::builder()
    .system_prompt("You are a helpful assistant")
    .model("qwen2.5-32b-instruct")
    .base_url("http://localhost:1234/v1")
    .temperature(0.1)
    .build()?;

let mut stream = query("Hello", &options).await?;
while let Some(block) = stream.next().await {
    match block? {
        ContentBlock::Text(t) => print!("{}", t.text),
        _ => {}
    }
}
```

### Automatic Tool Execution
```rust
let options = AgentOptions::builder()
    .tools(vec![my_tool])
    .auto_execute_tools(true)
    .max_tool_iterations(10)
    .build()?;
```

### Multimodal Vision
```rust
// From URL
let img = ImageBlock::from_url("https://...", "image/jpeg");
// From file
let img = ImageBlock::from_file("screenshot.png").await?;
// From base64 (with validation: char set, length, padding, MIME injection prevention)
let img = ImageBlock::from_base64(data, "image/png");
```

### Interrupt
```rust
// From a separate Tokio task:
client.interrupt().await;
```

### Context Management
```rust
use open_agent::context::{estimate_tokens, truncate_messages};
```

## AgentOptions Builder

| Method | Default | Description |
|--------|---------|-------------|
| `.model(s)` | required | Model name |
| `.base_url(s)` | required | OpenAI-compatible endpoint |
| `.system_prompt(s)` | `""` | System message |
| `.max_turns(n)` | `1` | Max conversation turns |
| `.temperature(f)` | `None` | Sampling temperature |
| `.max_tokens(n)` | `None` | Max output tokens |
| `.tools(vec)` | `[]` | Tool definitions |
| `.auto_execute_tools(b)` | `false` | Auto-execute tools |
| `.max_tool_iterations(n)` | `5` | Safety limit for tool loops |
| `.timeout_secs(n)` | `60` | HTTP timeout |
| `.api_key(s)` | `"not-needed"` | API key |

## Supported Providers

All OpenAI-compatible endpoints:
- LM Studio: `http://localhost:1234/v1`
- Ollama: `http://localhost:11434/v1`
- llama.cpp server (OpenAI mode)
- vLLM, Text Generation WebUI
- Any local gateway proxying cloud models

## Development Rules

- TDD: Write failing tests first, implement, refactor
- All tests must pass before committing: `cargo test`
- Run `cargo clippy -- -D warnings` before committing — zero warnings required
- Run `cargo fmt` before committing
- `auto_execute_tools` defaults to `false` (backwards compat with manual mode)
- Manual mode `receive()` must commit assistant messages to history on natural stream EOF
- Partial output must be discarded on stream errors, interrupts, and abandoned streams
- `clear_history()` must also clear the manual buffer (prevent stale replay)
- `ImageBlock::from_base64` validates: char set, length (multiple of 4), padding (max 2 `=`), MIME injection (no semicolons, newlines, commas)
- Warn (via `log::warn!`) for base64 payloads exceeding 10MB
- `cargo audit` must pass — update deps promptly when RustSec advisories fire

## Testing Notes

- Integration tests in `tests/` require a running local model server unless mocked
- `backward_compatibility_test.rs` guards against breaking API changes
- `defensive_validation_test.rs` and `security_bypass_test.rs` test ImageBlock validation
- `send_message_test.rs` covers the manual-mode history bug (v0.6.2 fix)
