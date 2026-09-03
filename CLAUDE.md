# Open Agent SDK (Rust)

## Project Description

A production-ready Rust SDK (v0.11.2) for building AI agents over two wire protocols: OpenAI
chat completions (LM Studio, Ollama, llama.cpp, vLLM, OpenRouter) and Anthropic messages.
Features incremental streaming, tools, hooks, retry logic, custom headers, redirect rejection,
and comprehensive examples. Published to crates.io as `open-agent-sdk`.

Crate name: `open_agent` (lib target). MSRV: Rust 1.85 (edition 2024).

## Repository Structure

```
open-agent-sdk-rust/
├── src/
│   ├── lib.rs           # Public API exports + doctests
│   ├── client.rs        # query() function + Client struct (facade)
│   ├── client/          # Client implementation submodules
│   ├── config.rs        # AgentOptions + config helpers (facade)
│   ├── context.rs       # Token estimation + truncation utilities (opt-in)
│   ├── error.rs         # AgentError enum (thiserror)
│   ├── hooks.rs         # Hooks system facade
│   ├── hooks/           # PreToolUse, PostToolUse, UserPromptSubmit submodules
│   ├── retry.rs         # Retry policy facade
│   ├── retry/           # Exponential backoff + jitter submodules
│   ├── tools.rs         # Tool trait + #[tool] derive support (facade)
│   ├── tools/           # Tool registration and dispatch submodules
│   ├── types.rs         # Core message/block types (facade)
│   ├── types/           # TextBlock, ToolUseBlock, ToolResultBlock, AssistantMessage submodules
│   ├── utils.rs         # Shared utilities (facade)
│   └── utils/           # Protocol formatters, stream helpers submodules
├── tests/               # 31 integration test files (183 integration tests)
│   ├── common/          # Shared test helpers
│   ├── support/         # Wiremock fixtures and stubs
│   ├── integration_tests.rs
│   ├── advanced_integration_test.rs
│   ├── anthropic_protocol_test.rs
│   ├── auto_execution_test.rs
│   ├── backward_compatibility_test.rs
│   ├── ci_workflow_policy_test.rs
│   ├── client_image_serialization_test.rs
│   ├── config_env_test.rs
│   ├── context_estimation_test.rs
│   ├── custom_headers_test.rs
│   ├── debug_logging_test.rs
│   ├── defensive_validation_test.rs
│   ├── edge_cases_test.rs
│   ├── hooks_history_snapshot_test.rs
│   ├── hooks_integration_test.rs
│   ├── image_serialization_test.rs
│   ├── mutation_ci_scope_test.rs       # Validates CI mutation scoping logic
│   ├── mutation_scripts_test.rs        # Validates mutation script behavior (excluded from crate publish)
│   ├── mutation_transport_scripts_test.rs
│   ├── package_manifest_test.rs
│   ├── redirect_policy_test.rs         # Loopback proof redirect target is never contacted
│   ├── regression_finish_reason_test.rs
│   ├── regression_incremental_streaming_test.rs
│   ├── regression_max_tokens_test.rs
│   ├── regression_reasoning_channel_test.rs
│   ├── regression_retry_classification_test.rs
│   ├── regression_stream_flush_test.rs
│   ├── security_bypass_test.rs
│   ├── send_message_test.rs
│   ├── source_file_size_test.rs
│   └── tool_call_content_test.rs
├── examples/            # 15 runnable examples
│   ├── simple_query.rs
│   ├── calculator_tools.rs
│   ├── hooks_example.rs
│   ├── context_management.rs
│   ├── interrupt_demo.rs
│   ├── git_commit_agent.rs
│   ├── log_analyzer_agent.rs
│   ├── advanced_patterns.rs
│   ├── auto_execution_demo.rs
│   ├── multi_tool_agent.rs
│   ├── vision_example.rs
│   ├── vision_api_demo.rs
│   ├── test_tool_serialization.rs
│   └── anthropic_query.rs
├── scripts/             # Mutation testing scripts (not published to crates.io)
│   ├── mutants-ci-scope.sh
│   ├── mutants-common.sh
│   ├── mutants-run.sh
│   ├── mutants-remote.sh
│   └── mutants-staged.sh
├── benches/
│   └── performance.rs   # criterion benchmark (bench = false in lib profile)
├── .githooks/           # Pre-commit hooks (install with git config core.hooksPath .githooks)
├── .github/             # CI workflows
├── Cargo.toml
├── Cargo.lock
└── CHANGELOG.md
```

## Tech Stack

| Component | Crate |
|-----------|-------|
| **Async runtime** | tokio 1.50 (features = ["full"]) |
| **Async streams** | tokio-stream 0.1 |
| **HTTP client** | reqwest 0.12.28 (json + stream features) |
| **Serialization** | serde 1.0 + serde_json 1.0 |
| **Error handling** | thiserror 2.0 + anyhow 1.0 |
| **Async utilities** | futures 0.3 + async-trait 0.1 |
| **SSE streaming** | eventsource-stream 0.2 |
| **Retry jitter** | rand 0.10 |
| **Logging** | log 0.4 |
| **Image encoding** | base64 0.23 (no default-features, std only) |
| **Test HTTP mocks** | wiremock =0.6.4 (pinned — 0.6.5 requires let-chains, breaks MSRV 1.85) |
| **Virtual clock** | tokio test-util feature (for exact backoff timing in retry tests) |
| **Benchmarks** | criterion 0.7 |
| **Mutation testing** | cargo-mutants 27.1.0 |

## Common Commands

```bash
# Build
cargo build

# Run tests (all: unit + integration + doctests)
cargo test

# Run only unit tests
cargo test --lib

# Run only integration tests
cargo test --test '*'

# Run a specific integration test file
cargo test --test redirect_policy_test

# Run doctests only
cargo test --doc

# Format
cargo fmt

# Lint (zero warnings policy)
cargo clippy --all-targets --all-features -- -D warnings

# Run an example
cargo run --example simple_query
cargo run --example anthropic_query

# Benchmark
cargo bench

# Install pre-commit hook (runs fmt + clippy + tests + mutants --in-diff)
git config core.hooksPath .githooks

# Mutation testing (full sweep)
bash scripts/mutants-run.sh

# Mutation testing (staged diff only)
bash scripts/mutants-staged.sh
```

## Key Features

### Two Wire Protocols
- **OpenAI chat completions**: LM Studio, Ollama, llama.cpp, vLLM, OpenRouter, any compatible endpoint
- **Anthropic messages API**: Direct Anthropic API access with native message types

### Incremental Streaming (v0.10.0 — breaking)
Stream content block-by-block as it arrives. Breaking change: `query()` now returns
`impl Stream<Item = Result<AssistantMessage, AgentError>>` — callers iterate the stream
rather than awaiting a single message.

### Tool Use
Register tools as async Rust functions; the SDK handles tool-call round-trips automatically
when `auto_execute_tools = true` (with a `max_tool_iterations` safety cap).

### Hooks
Lifecycle callbacks: `PreToolUse` (can block/modify), `PostToolUse` (observation only),
`UserPromptSubmit` (inspect/modify incoming prompt). `PostToolUse` return values are ignored.

### Custom Headers (v0.11.0)
Pass arbitrary HTTP headers per request (e.g. `X-Request-ID`, per-call auth tokens).

### Redirect Rejection (v0.11.2 — security)
Model requests never follow HTTP redirects, including same-origin `30x` responses.
Credentials (custom headers, API keys) are sent only to the configured origin. Any redirect
response surfaces through the existing API-status error path.

### Context Management (opt-in)
`estimate_tokens()` uses character-based approximation; no external tokenizer dependency.
Install tiktoken via Python interop only if precision is needed for multilingual content.
Context truncation is explicit — the SDK never silently mutates conversation history.

### Retry Module
Exponential backoff with jitter for transient HTTP/provider errors. Virtual clock support
via tokio `test-util` allows deterministic timing assertions without real sleeps.

### Vision Support
Encode images as base64 data URIs and include them in messages for multimodal models.

## AgentOptions Fields

| Field | Default | Description |
|-------|---------|-------------|
| `system_prompt` | required | System instructions |
| `model` | required | Model identifier (provider-specific) |
| `base_url` | required | OpenAI-compatible endpoint URL |
| `protocol` | `OpenAI` | `Protocol::OpenAI` or `Protocol::Anthropic` |
| `tools` | `[]` | Tool registrations |
| `hooks` | `None` | Hook handlers |
| `auto_execute_tools` | `false` | Auto tool-call loop (off by default for backwards compat) |
| `max_tool_iterations` | `5` | Safety cap on tool-call rounds |
| `max_turns` | `1` | Max conversation turns |
| `max_tokens` | `4096` | Max output tokens |
| `temperature` | `0.7` | Sampling temperature |
| `timeout_secs` | `60` | HTTP request timeout |
| `api_key` | `"not-needed"` | API key (local servers don't require one) |
| `custom_headers` | `None` | Extra HTTP headers sent with every request |

## Test Counts (v0.11.2)

| Category | Count |
|----------|-------|
| Unit tests | 248 |
| Integration tests | 183 (31 files) |
| Doctests | 164 |
| **Total** | **595** |

All 595 tests must pass before committing.

## Development Rules

- **TDD**: Write a failing test first, implement, refactor, commit
- **Zero warnings**: `cargo clippy --all-targets --all-features -- -D warnings` must be clean
- **All 595 tests must pass** before committing — run `cargo test` to verify
- **Install pre-commit hook** once after cloning: `git config core.hooksPath .githooks`
  (runs fmt, clippy, full test suite, and `cargo mutants --in-diff` on staged changes)
- **No backwards-incompatible changes** to `AgentOptions` field defaults without a major version bump
- **`auto_execute_tools = false` is the default** — manual mode must remain the default
- **`PostToolUse` is observation-only** — hook return values are ignored; use `PreToolUse` to intercept or block
- **Context management is opt-in** — never silently truncate conversation history
- **Redirect rejection is a security control** — do not add redirect-follow modes or config flags

## Mutation Testing

Mutation testing uses `cargo-mutants` 27.1.0.

**CI scoping policy** (unreleased, next version):
- Added/modified/deleted inline tests → run mutants only for the owning source files
- Integration test changes, fixtures, snapshots, and ambiguous mappings → full sweep
- Production-only revisions → skip mutation entirely
- Manual dispatch + 15th of each month → always sweep full tree
- Failed runs retain bounded evidence for the following day's autonomous repair PR

The `mutation_ci_scope_test.rs` integration test validates the scoping logic.
Scripts in `scripts/` are excluded from the published crate.

## Crate Publish Exclusions

The following are excluded from the published crate (they reference repo-only files):
- `.Codex/`, `.githooks/`, `.github/`, `scripts/`
- `tests/ci_workflow_policy_test.rs`
- `tests/mutation_scripts_test.rs`
- `tests/mutation_transport_scripts_test.rs`
- `tests/fixtures/`, `tests/support/`
- `mutants.out/`

## Key Design Decisions

### Incremental Streaming (v0.10.0)
`query()` returns a stream of `AssistantMessage` values, one per content block, instead of
buffering the full response. Breaking change accepted in v0.10.0 to enable low-latency UX.

### Finish Reasons
Streamed finish-reason metadata is preserved on `AssistantMessage` so callers can distinguish
stop, length, tool_calls, and content_filter outcomes without parsing raw SSE.

### Redirect Rejection (v0.11.2)
reqwest's default redirect policy is overridden to reject all redirects at the HTTP client
level. This prevents credential forwarding to unintended origins. The same-origin case is
also covered — a server that redirects within its own domain cannot steal headers.

### Custom Headers (v0.11.0)
Per-request custom headers allow API gateway tokens, request tracing, and provider-specific
extensions without wrapping the client. Headers are merged with the SDK's own headers;
callers cannot override `Content-Type` or `Authorization` set by the protocol layer.

### wiremock Version Pin
`wiremock` is pinned to exactly `=0.6.4` because 0.6.5 uses let-chains syntax which requires
Rust ≥ 1.88, breaking the 1.85 MSRV. Do not upgrade until the MSRV is raised.

### tokio test-util
`tokio = { features = ["test-util"] }` in dev-dependencies provides a paused virtual clock
(`tokio::time::pause()`) for retry tests. This lets tests assert exact backoff timing without
real sleeps or flaky tolerances.

### Manual Context Management
The SDK never silently mutates conversation history. `context.rs` provides utilities for
callers who want token budgeting, but truncation is always opt-in and explicit.
