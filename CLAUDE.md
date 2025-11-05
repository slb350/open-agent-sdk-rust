# Open Agent SDK - Rust

## Quick Reference

### Crates.io Package
- **Package Name**: `open-agent-sdk` (when published)
- **Install**: `cargo add open-agent-sdk`
- **Import**: `use open_agent::{query, AgentOptions, Client};`
- **Crates.io URL**: https://crates.io/crates/open-agent-sdk (not yet published)
- **Docs.rs**: https://docs.rs/open-agent-sdk (after publication)

### Git Remotes
- **GitHub**: `git@github.com:slb350/open-agent-sdk-rust.git`
- **Gitea**: `ssh://steve@192.168.1.14:22/steve/open-agent-sdk-rust.git`

### Pushing to GitHub
```bash
# Use the specific SSH key for GitHub
GIT_SSH_COMMAND='ssh -i ~/.ssh/github_any_agent' git push github main

# Or configure permanently for this repo:
git config core.sshCommand "ssh -i ~/.ssh/github_any_agent"
```

---

## Development Workflow

### Running Tests

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_agent_options_builder

# Run integration tests only
cargo test --test hooks_integration_test
cargo test --test auto_execution_test
cargo test --test advanced_integration_test
```

### Running Examples

```bash
# Basic examples
cargo run --example simple_query
cargo run --example calculator_tools
cargo run --example hooks_example

# Advanced examples
cargo run --example auto_execution_demo
cargo run --example multi_tool_agent
cargo run --example advanced_patterns

# Production examples
cargo run --example git_commit_agent
cargo run --example log_analyzer_agent
```

### Code Quality

```bash
# Format code
cargo fmt

# Run clippy
cargo clippy

# Build docs
cargo doc --no-deps --open
```

---

## Release Process

Complete checklist for releasing a new version (e.g., v0.2.0):

### 1. Development on Feature Branch

```bash
# Create and work on feature branch
git checkout -b feature-name

# Make your changes, then update documentation
```

**Update these files:**
- ✅ `CHANGELOG.md` - Add new version entry at top (create if doesn't exist)
- ✅ `README.md` - Update examples, API reference if needed
- ✅ `Cargo.toml` - Bump version number
- ✅ Examples - Update if API changed

**Verify quality:**
```bash
# Run all tests
cargo test

# Run clippy
cargo clippy -- -D warnings

# Check formatting
cargo fmt -- --check

# Build documentation
cargo doc --no-deps
```

**Push feature branch:**
```bash
# Push to Gitea (origin) for backup/review
git add .
git commit -m "feat: descriptive message"
git push origin feature-name
```

### 2. Merge to Main

```bash
# Switch to main and merge
git checkout main
git merge feature-name --no-ff

# Push to Gitea
git push origin main
```

### 3. Build and Release

**Build package:**
```bash
# Clean build
cargo clean
cargo build --release

# Verify package contents
cargo package --list

# Dry run publish
cargo publish --dry-run
```

**Push to GitHub:**
```bash
# Push main branch to public GitHub repo
GIT_SSH_COMMAND='ssh -i ~/.ssh/github_any_agent' git push github main
```

**Create and push git tag:**
```bash
# Create version tag
git tag v0.2.0

# Push tag to both remotes
git push origin v0.2.0
GIT_SSH_COMMAND='ssh -i ~/.ssh/github_any_agent' git push github v0.2.0
```

**Publish to crates.io:**
```bash
# First time only: login to crates.io
cargo login

# Publish
cargo publish
```

**Create GitHub Release:**
```bash
# Create release from tag with CHANGELOG content
gh release create v0.2.0 \
  --title "v0.2.0 - Feature Name" \
  --notes "Copy the relevant section from CHANGELOG.md here"
```

### 4. Verify Release

```bash
# Check crates.io
open https://crates.io/crates/open-agent-sdk/0.2.0

# Check docs.rs
open https://docs.rs/open-agent-sdk/0.2.0

# Check GitHub tag
open https://github.com/slb350/open-agent-sdk-rust/releases
```

### 5. Cleanup (Optional)

```bash
# Delete feature branch locally
git branch -d feature-name

# Delete feature branch on Gitea
git push origin --delete feature-name
```

---

## Project Overview

**Goal**: Rust implementation of the Open Agent SDK with 100% feature parity with the Python version.

**Status**: ✅ 100% feature parity achieved

### Features

- ✅ Streaming query and multi-turn client
- ✅ Tool system with function calling
- ✅ Lifecycle hooks (PreToolUse, PostToolUse, UserPromptSubmit)
- ✅ Auto-execution mode with automatic tool calling
- ✅ Context management (token estimation, truncation)
- ✅ Interrupt capability
- ✅ Retry logic with exponential backoff
- ✅ 85+ comprehensive tests
- ✅ 10 production examples

### Architecture

```
┌─────────────────────────────────────┐
│   Your Application / Agent Code     │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  Open Agent SDK (Rust)              │
│  - query() function                 │
│  - Client struct (multi-turn)       │
│  - AgentOptions builder             │
│  - Tool system & hooks              │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      reqwest HTTP Client            │
│  (async streaming)                  │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  Local Model Servers                │
│  - LM Studio (localhost:1234)       │
│  - Ollama (localhost:11434)         │
│  - llama.cpp server                 │
│  - vLLM, etc.                       │
└─────────────────────────────────────┘
```

---

## Testing Strategy

### Test Coverage

- **57 Unit Tests**: Core functionality across 10 modules
- **28 Integration Tests**:
  - 6 hooks integration tests
  - 13 auto-execution tests
  - 9 advanced integration tests

### Running Tests Against Live Servers

```bash
# Start Ollama
ollama serve

# Run examples that test against Ollama
cargo run --example simple_query
cargo run --example auto_execution_demo

# Test against network endpoint
cargo run --example test_llamacpp  # if example exists
```

---

## Common Issues

### Compilation Errors

**Missing `ContentBlock::ToolResult` pattern match:**
- When adding new `ContentBlock` variants, update all match statements
- Check examples: `simple_query.rs`, `calculator_tools.rs`, etc.

**Thread safety issues with `tokio::spawn`:**
- `Client` contains `dyn Stream + Send` which is not `Sync`
- Use `FuturesUnordered` instead of `tokio::spawn` for concurrency
- See `advanced_patterns.rs` for examples

---

## Phase 4: Performance & Polish (Optional)

See `PHASE4_PLAN.md` for:
- Benchmark suite with `criterion`
- CI/CD with GitHub Actions
- Publishing to crates.io

---

## Why Rust?

- **Performance**: Zero-cost abstractions, no GC pauses
- **Safety**: Memory safety without runtime overhead
- **Concurrency**: Fearless concurrency with compile-time guarantees
- **Production Ready**: Type safety, error handling, comprehensive testing
- **Small Binaries**: Standalone executables under 10MB

---

## Contributing

This is a reference implementation developed alongside the Python SDK. Contributions welcome!

---

## License

MIT License - see LICENSE for details

---

## Acknowledgments

Rust port of [open-agent-sdk](https://github.com/slb350/open-agent-sdk) Python library.

---

**Made with ❤️ for developers who want to run AI agents on their own hardware**
