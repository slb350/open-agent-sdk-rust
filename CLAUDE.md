# Open Agent SDK (Rust)

## Project Description

A lightweight Rust SDK (v0.6.7 source) for building AI agents with local or cloud LLMs via OpenAI-compatible endpoints. Rust port of the Python open-agent-sdk. Published to crates.io as `open-agent-sdk`.

## Repository Structure

```text
open-agent-sdk-rust/
├── src/
│   ├── client.rs      # query() function + Client struct (streaming, tool loop, send_message)
│   ├── config.rs      # Provider enum, get_base_url(), get_model() helpers
│   ├── context.rs     # estimate_tokens(), is_approaching_limit(), truncate_messages()
│   ├── error.rs       # Error type and Result alias
│   ├── hooks.rs       # Hooks system (PreToolUse, PostToolUse, UserPromptSubmit, HookDecision)
│   ├── lib.rs         # Public exports and prelude module
│   ├── retry.rs       # RetryConfig, retry_with_backoff, retry_with_backoff_conditional, is_retryable_error
│   ├── tools.rs       # tool() builder + Tool struct
│   ├── types.rs       # AgentOptions builder, ContentBlock, ImageBlock, Message, MessageRole, etc.
│   └── utils.rs       # SSE parser, ToolCallAggregator, OpenAI wire types
├── examples/
│   ├── simple_query.rs              # Minimal streaming query
│   ├── calculator_tools.rs          # Manual tool execution
│   ├── auto_execution_demo.rs       # Automatic tool execution
│   ├── multi_tool_agent.rs          # Production agent: 5 tools + hooks
│   ├── hooks_example.rs             # Hook patterns (security gate, audit log)
│   ├── context_management.rs        # History management patterns
│   ├── interrupt_demo.rs            # Timeout and cancellation patterns
│   ├── git_commit_agent.rs          # Production: git diff → commit message
│   ├── log_analyzer_agent.rs        # Production: log file analysis
│   ├── advanced_patterns.rs         # Retry logic, concurrent requests
│   ├── vision_example.rs            # Multimodal: URLs, local files, base64
│   ├── vision_api_demo.rs           # Vision API walkthrough
│   └── test_tool_serialization.rs   # Tool call serialization verification
├── benches/
│   └── performance.rs               # Criterion benchmarks
├── tests/
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
│   ├── package_manifest_test.rs     # Package exclusion coverage (CLAUDE.md, .markdownlint.json)
│   ├── security_bypass_test.rs
│   ├── send_message_test.rs         # Manual-mode history regression (v0.6.2)
│   └── tool_call_content_test.rs
├── .github/
│   ├── dependabot.yml               # Grouped weekly Cargo dependency updates
│   └── workflows/
│       ├── ci.yml                   # CI pipeline (build, test, clippy, fmt)
│       └── scheduled-audit.yml      # Scheduled dependency audit
├── .markdownlint.json               # Markdown lint rules (disable MD013, allow duplicate sibling headings)
├── Cargo.toml
├── Cargo.lock
├── CHANGELOG.md
└── README.md
```

## Tech Stack

| Component | Technology |
| ----------- | ------------ |
| **Language** | Rust 1.85+ |
| **Async runtime** | Tokio 1.50+ |
| **HTTP** | reqwest 0.12.28+ |
| **Serialization** | serde, serde_json 1.0 |
| **Streams** | futures 0.3.32+ |
| **Error handling** | anyhow 1.0.103+ |
| **Logging** | log 0.4.29+ |
| **Benchmarks** | criterion |
| **Automation** | Dependabot (grouped weekly Cargo updates) |

## Common Commands

```bash
# Build
cargo build

# Run tests (all: unit + integration + doctests)
cargo test

# Run tests with output
cargo test -- --nocapture

# Run a specific test
cargo test test_agent_options_builder

# Run an example
cargo run --example git_commit_agent
cargo run --example vision_example

# Run benchmarks
cargo bench

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt
```

## Public API

### AgentOptions (builder pattern)

```rust
use open_agent::AgentOptions;

let options = AgentOptions::builder()
    .system_prompt("You are a helpful assistant")  // required
    .model("qwen2.5-32b-instruct")                 // required
    .base_url("http://localhost:1234/v1")           // required
    .tool(my_tool)                                  // add one tool
    .tools(vec![tool_a, tool_b])                    // add multiple tools
    .hooks(hooks)                                   // lifecycle hooks
    .auto_execute_tools(true)                       // auto-execute (default: false)
    .max_tool_iterations(10)                        // safety limit (auto mode)
    .max_tokens(4096)                               // tokens to generate
    .max_turns(1)                                   // conversation turns
    .temperature(0.7)                               // sampling temperature
    .timeout(60)                                    // request timeout (seconds)
    .api_key("not-needed")                          // API key (default: "not-needed")
    .build()?;
```

### query() — single-turn

```rust
use open_agent::{query, ContentBlock};
use futures::StreamExt;

let mut stream = query("prompt", &options).await?;
while let Some(block) = stream.next().await {
    if let ContentBlock::Text(t) = block? { print!("{}", t.text); }
}
```

### Client — multi-turn

```rust
use open_agent::{Client, ContentBlock};

let mut client = Client::new(options)?;
client.send("What's 2 + 2?").await?;

while let Some(block) = client.receive().await? {
    match block {
        ContentBlock::Text(t) => println!("{}", t.text),
        ContentBlock::ToolUse(tu) => { /* handle or let auto-execute handle it */ }
        _ => {}
    }
}

// Additional Client methods
client.send_message(msg).await?;   // send a pre-built Message (e.g. with images)
client.clear_history();            // reset to system prompt only
client.options();                  // access AgentOptions
client.get_tool("name");           // look up a registered tool
client.interrupt();                // cancel in-flight stream (idempotent)
client.interrupt_handle();         // Arc<AtomicBool> for cross-task cancellation
client.history();                  // &[Message] — read conversation history
client.history_mut();              // &mut Vec<Message> — mutate (e.g. truncate)
```

For cancellation from another task, share `client.interrupt_handle()` and set the
atomic flag. Never place `Client` behind a synchronous mutex whose guard can be
held across `receive().await`; that pattern deadlocks the cancellation task.

### tool() builder

```rust
use open_agent::tool;
use serde_json::json;

let my_tool = tool("add", "Add two numbers")
    .param("a", "number")
    .param("b", "number")
    .build(|args| async move {
        let result = args["a"].as_f64().unwrap_or(0.0) + args["b"].as_f64().unwrap_or(0.0);
        Ok(json!({"result": result}))
    });

// Full JSON Schema control:
let my_tool = tool("name", "desc")
    .schema(json!({ "type": "object", "properties": { "x": { "type": "number" } }, "required": ["x"] }))
    .build(|args| async move { Ok(json!({})) });
```

### Hooks

```rust
use open_agent::{Hooks, HookDecision};

let hooks = Hooks::new()
    .add_pre_tool_use(|event| async move {
        if event.tool_name == "dangerous_op" {
            return Some(HookDecision::block("requires approval"));
        }
        Some(HookDecision::continue_())
    })
    .add_post_tool_use(|event| async move {
        println!("ran: {} -> {:?}", event.tool_name, event.tool_result);
        None
    })
    .add_user_prompt_submit(|event| async move {
        // Modify or block the prompt before it's sent
        Some(HookDecision::continue_())
    });
```

`HookDecision` variants: `continue_()`, `block(reason)`, `modify_input(json, reason)`, `modify_prompt(text, reason)`.

Hooks run sequentially; first non-None decision wins. Hook name constants: `HOOK_PRE_TOOL_USE`, `HOOK_POST_TOOL_USE`, `HOOK_USER_PROMPT_SUBMIT`.

### Multimodal Vision

```rust
use open_agent::{Message, ContentBlock, ImageBlock, ImageDetail};

// From URL
let msg = Message::user_with_image("Describe this", "https://example.com/photo.jpg")?;

// From local file (auto-encodes to base64; supports jpg, png, gif, webp, bmp, svg)
let msg = Message::new(MessageRole::User, vec![
    ContentBlock::Text(TextBlock::new("Describe this photo")),
    ContentBlock::Image(ImageBlock::from_file_path("/path/to/photo.jpg")?),
]);

// From base64
let msg = Message::user_with_base64_image("Describe this", base64_data, "image/png")?;

// Control detail level
let msg = Message::user_with_image_detail("Analyze", url, ImageDetail::High)?;
```

`ImageBlock` sources: `from_url(url)`, `from_file_path(path)`, `from_base64(data, mime)`. Detail levels: `Low`, `High`, `Auto`.

### Context Management

```rust
use open_agent::{estimate_tokens, is_approaching_limit, truncate_messages};

let tokens = estimate_tokens(client.history());
if is_approaching_limit(client.history(), 32000, 0.8) {
    let truncated = truncate_messages(client.history(), 10, true); // keep last 10, preserve system
    *client.history_mut() = truncated;
}
```

Context management is **opt-in** — the SDK never silently mutates history.

### Retry

```rust
use open_agent::retry::{RetryConfig, retry_with_backoff, retry_with_backoff_conditional, is_retryable_error};

let config = RetryConfig::default()  // 3 attempts, exponential backoff
    .max_attempts(5)
    .initial_delay_ms(100)
    .max_delay_ms(5000)
    .backoff_multiplier(2.0);

let result = retry_with_backoff(config, || async { some_op().await }).await?;
```

### Provider Helpers

```rust
use open_agent::{Provider, get_base_url, get_model};

let url = get_base_url(Some(Provider::LMStudio), None);  // http://localhost:1234/v1
let url = get_base_url(Some(Provider::Ollama), None);    // http://localhost:11434/v1
let model = get_model(None, true);                        // reads OPEN_AGENT_MODEL env var
```

### Error and Result Types

```rust
use open_agent::{Error, Result};
```

### Prelude

```rust
use open_agent::prelude::*;  // imports the most commonly used types
```

## Supported Providers

All OpenAI-compatible endpoints:

- LM Studio: `http://localhost:1234/v1`
- Ollama: `http://localhost:11434/v1`
- llama.cpp server (OpenAI-compatible mode)
- vLLM, Text Generation WebUI
- Local gateways proxying cloud models (e.g. kimi-k2, deepseek, qwen3-coder via Ollama)

## Test Coverage

- 119 unit tests (lib + inline)
- 80 active integration tests across 14 test files (12 additional `#[ignore]`d by default)
- 151 active doctests (17 additional doctests are `ignore`d)

Total: 350 active tests.

```bash
cargo test              # run all (unit + integration + doctests)
cargo test --lib        # unit tests only
cargo test --test '*'   # integration tests only
cargo test --doc        # doctests only
```

## Development Rules

- **TDD**: Write failing tests first, implement, refactor, commit
- **All tests must pass** before committing — `cargo test` must be green
- **Clippy clean**: `cargo clippy -- -D warnings` must pass with zero warnings
- **Formatted**: `cargo fmt --check` must pass
- **No breaking changes** to `AgentOptions` field order or builder method signatures
- **`auto_execute_tools` defaults to `false`** (backwards compatibility)
- **Context management is opt-in** — never silently mutate history
- **`add_tool_result()` is sync** — no await needed
- Commit format: `type(scope): description` (feat, fix, docs, test, refactor, chore)
- Dependency updates: Dependabot runs weekly (grouped Cargo updates) — resolve security advisories promptly
- Reqwest compatibility: keep reqwest on 0.12.x while `Error::Http` publicly wraps `reqwest::Error`; upgrading reqwest requires a documented v0.7.0 release
- Base64 safety: use base64 0.23 without its default `simd-unsafe` feature unless a measured need justifies enabling it
- GitHub Actions: pin every third-party action to an immutable full commit SHA with a version comment; Dependabot maintains the pins
- Workflow permissions: default to `contents: read` and grant additional permissions only to the job that requires them
- Audit workflows: set `denyWarnings: true` and `createIssues: false` so vulnerabilities, yanked crates, unmaintained crates, and unsoundness warnings fail CI without requiring issue-write permissions
- PR benchmarks: compare Criterion results directly against the base commit with a shared target directory; do not restore the obsolete `boa-dev/criterion-compare-action`
- Coverage reports: retain Tarpaulin XML with the latest compatible, immutable-SHA-pinned `actions/upload-artifact` release (currently v7) and fail CI when the report is missing

## Security Advisories (v0.6.5)

RUSTSEC-2026-0190 and RUSTSEC-2026-0204 resolved:

- `anyhow` raised to `1.0.103` minimum
- `reqwest` pinned to `0.12.28`
- `log` raised to `0.4.29`
- `futures` raised to `0.3.32`

## Current Version

**v0.6.7 source** — Rust 1.85-compatible production dependency updates,
immutable and least-privilege GitHub Actions, first-party coverage artifacts,
and direct Criterion base/head benchmark comparison.

Features: multimodal vision (URLs, local files, base64), manual-mode history fix (v0.6.2 regression), retry module, interrupt capability, lifecycle hooks, automatic tool execution, context management utilities, provider helpers, prelude module, OpenAI wire type exports.
