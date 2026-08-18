# Open Agent SDK (Rust)

## Project Description

A Rust SDK (v0.7.1) for building AI agents with local or cloud LLMs via OpenAI-compatible endpoints. Mirrors the Python open-agent-sdk API shape. Published to crates.io as `open-agent-sdk`.

## Repository Structure

```
open-agent-sdk-rust/
├── src/
│   ├── lib.rs           # Public API exports
│   ├── client.rs        # Client struct, query() function, streaming loop
│   ├── types.rs         # AgentOptions, TextBlock, ToolUseBlock, ToolResultBlock, AssistantMessage, Error
│   ├── tools.rs         # Tool trait, tool! macro
│   ├── hooks.rs         # Hook system (PreToolUse, PostToolUse, UserPromptSubmit)
│   └── utils.rs         # ToolCallAggregator (SSE parsing + flush()), format helpers
├── tests/               # Integration tests (113 tests across 22 files; wiremock HTTP mocks)
├── benches/             # Criterion benchmarks
├── examples/            # Runnable examples
├── .githooks/           # Pre-commit hook (fmt check, clippy, test, mutants --in-diff)
├── .github/
│   └── workflows/
│       └── ci.yml       # 9-job CI: fmt, clippy, msrv, test-linux, test-macos,
│                        #           security, docs, coverage (tarpaulin LLVM), benchmarks
├── Cargo.toml           # crate metadata, version = "0.7.1"
├── Cargo.lock
└── CHANGELOG.md
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Rust (edition 2024, MSRV 1.85) |
| **Async runtime** | tokio 1.50 |
| **HTTP client** | reqwest 0.12 |
| **SSE parsing** | eventsource-stream 0.2 |
| **Serialization** | serde / serde_json |
| **Error handling** | thiserror 2.0, anyhow 1.0 |
| **Async traits** | async-trait 0.1 |
| **Random** | rand 0.10 |
| **Logging** | log 0.4 |
| **Base64** | base64 0.23 |
| **Futures** | futures 0.3 |
| **Test mocking** | wiremock =0.6.4 (pinned — let-chains on MSRV 1.85) |
| **Coverage** | cargo-tarpaulin (LLVM mode) |
| **Mutation testing** | cargo-mutants |
| **Benchmarks** | criterion 0.7 |

## Common Commands

```bash
# Build
cargo build

# Run tests (390 active: 124 unit + 113 integration + 153 doctests)
cargo test

# Lint
cargo clippy -- -D warnings

# Format (check only — used in CI and pre-commit)
cargo fmt --check

# Format (apply)
cargo fmt

# Mutation testing (zero survivors required)
cargo mutants

# Mutation testing for changed lines only (pre-commit subset)
cargo mutants --in-diff

# Run an example
cargo run --example simple_tool

# Benchmarks
cargo bench

# Check MSRV
cargo +1.85 test

# Security audit
cargo audit
```

## Pre-Commit Hook

Install by running once after cloning:

```bash
git config core.hooksPath .githooks
```

The hook runs: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`, and `cargo mutants --in-diff` (mutation testing on changed lines). All must pass before a commit is allowed.

## Public API

```rust
use open_agent_sdk::{
    Client,            // Multi-turn conversation client
    query,             // Simple single-turn query
    AgentOptions,      // Configuration struct
    TextBlock,         // Text content block
    ToolUseBlock,      // Tool call request block
    ToolResultBlock,   // Tool result to feed back
    AssistantMessage,  // Complete response wrapper
    Tool,              // Tool definition trait
    tool,              // tool! macro for ergonomic tool definitions
    PreToolUseEvent,   // Hook: before tool execution
    PostToolUseEvent,  // Hook: after tool execution (observation-only)
    UserPromptSubmitEvent, // Hook: before user input processed
    HookDecision,      // Hook return type (continue/block/modify)
    Error,             // SDK error type
};
```

## Error Type

`Error` is a non-exhaustive enum. Key variants:

| Variant | Description |
|---------|-------------|
| `Api { status: Option<u16>, message: String }` | HTTP/API error (struct variant since v0.7.0) |
| `Http(reqwest::Error)` | Network-level error |
| `Serialization(serde_json::Error)` | JSON parse/serialize error |
| `InvalidInput(String)` | Bad caller-provided input |
| `ToolExecution(String)` | Error from tool handler |
| `Interrupted` | `client.interrupt()` was called |

**Note**: `Error::Api` is a struct variant (not a tuple variant). Pattern match with `Error::Api { status, message }`.

## AgentOptions Fields

| Field | Default | Description |
|-------|---------|-------------|
| `system_prompt` | required | System instructions |
| `model` | required | Model name |
| `base_url` | required | OpenAI-compatible endpoint |
| `tools` | `vec![]` | Tool definitions |
| `hooks` | `None` | Lifecycle hooks |
| `auto_execute_tools` | `false` | Auto-execute tools |
| `max_tool_iterations` | `5` | Safety limit for tool loops |
| `max_turns` | `1` | Max conversation turns |
| `max_tokens` | `None` | Max output tokens — **omit to let the server decide** |
| `temperature` | `0.7` | Sampling temperature |
| `timeout_secs` | `60` | HTTP timeout (seconds) |
| `api_key` | `"not-needed"` | API key |

**Breaking change from 0.6.x**: `max_tokens` no longer defaults to `4096`. Omitting the field allows the server to apply its own default. Set explicitly if you need a specific limit.

## Upgrading from 0.6.x

Two breaking changes in 0.7.0:

1. **`Error::Api` is now a struct variant**:
   ```rust
   // Before (0.6.x):
   Error::Api(status, message)
   // After (0.7.x):
   Error::Api { status, message }
   ```

2. **`max_tokens` no longer defaults to 4096**:
   ```rust
   // Before: omitting max_tokens silently sent 4096
   // After: omitting max_tokens sends nothing (server decides)
   // If you relied on the 4096 default, add:
   AgentOptions { max_tokens: Some(4096), .. }
   ```

## Testing

**390 active tests** total:
- **124 unit tests** — in `src/` modules via `#[cfg(test)]`
- **113 integration tests** — in `tests/` using wiremock HTTP mocks (22 test files)
- **153 doctests** — inline examples in doc comments

Run the full suite: `cargo test`

### Mutation Testing

Mutation testing is **mandatory**. Zero survivors are required before merging. The CI `test-linux` job runs `cargo mutants` on every PR. The pre-commit hook runs `cargo mutants --in-diff` (changed lines only) for fast local feedback.

```bash
# Full mutation test (slow — run before PR)
cargo mutants

# Fast subset (changed lines — pre-commit)
cargo mutants --in-diff
```

### wiremock Version

`wiremock` is pinned to `=0.6.4` (exact version). Later versions use let-chains which require Rust > 1.85 (our MSRV). Do not upgrade wiremock without first verifying the new MSRV.

## CI Pipeline (9 jobs)

| Job | What it checks |
|-----|---------------|
| `fmt` | `cargo fmt --check` |
| `clippy` | `cargo clippy -- -D warnings` |
| `msrv` | `cargo +1.85 test` (MSRV gate) |
| `test-linux` | Full test suite + mutation testing on Linux |
| `test-macos` | Full test suite on macOS |
| `security` | `cargo audit` |
| `docs` | `cargo doc --no-deps` (no warnings) |
| `coverage` | tarpaulin LLVM coverage report |
| `benchmarks` | Criterion benchmarks (no regression gate yet) |

CI runs on GitHub Actions. Gitea is a passive mirror — do not open PRs or trigger CI there.

## Key Features

### Streaming API
```rust
let mut stream = client.query("prompt").await?;
while let Some(msg) = stream.next().await {
    for block in msg?.content {
        if let Block::Text(t) = block {
            println!("{}", t.text);
        }
    }
}
```

### SSE Robustness
`ToolCallAggregator::flush()` drains any partially-buffered tool calls at end-of-stream, preventing silent data loss when providers omit the final SSE flush event.

### Retryable Status Codes
The client retries on: 408, 429, 500, 502, 503, 504, 529.

### Tool Use
```rust
let my_tool = tool!("my_tool", "Description", async |args: MyArgs| {
    Ok(serde_json::json!({ "result": "..." }))
});

let options = AgentOptions {
    tools: vec![my_tool],
    auto_execute_tools: true,
    max_tool_iterations: 10,
    ..
};
```

### Hooks
```rust
async fn pre_tool_hook(event: &PreToolUseEvent) -> HookDecision {
    if event.tool_name == "dangerous_op" {
        return HookDecision::block("Not allowed");
    }
    HookDecision::continue_()
}
```

`PostToolUseEvent` handlers are **observation-only** — return values are ignored. Use `PreToolUseEvent` to intercept or block tool calls.

## MSRV Policy

Minimum Supported Rust Version: **1.85**. Do not use language features or dependency versions that require a newer Rust. Verify with `cargo +1.85 test` before merging.

## Development Rules

- **TDD**: Write failing tests first, implement, refactor, commit.
- **Mutation testing**: Zero survivors required. Run `cargo mutants --in-diff` before committing, full `cargo mutants` before a PR.
- **No breaking changes** without a CHANGELOG entry and version bump (semver).
- **Commit format**: `type(scope): description` — types: feat, fix, docs, test, refactor, chore, perf.
- **All CI jobs must pass** before merging to main.
- **MSRV**: Never break 1.85 compatibility. Check before adding or upgrading any dependency.
- **Error handling**: All errors must be propagated with context. No `.unwrap()` in library code.
- **Run pre-commit**: `git config core.hooksPath .githooks` installs the hook; run `just check` or the hook manually before each commit.

## Supported Providers

All OpenAI-compatible endpoints:
- LM Studio: `http://localhost:1234/v1`
- Ollama: `http://localhost:11434/v1`
- llama.cpp (OpenAI mode)
- vLLM, Text Generation WebUI
- Any local or cloud gateway with OpenAI-compatible chat completions
