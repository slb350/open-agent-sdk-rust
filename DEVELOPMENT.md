# Development Guide

## Prerequisites

- Rust 1.70+ (2021 edition)
- Internet connection for initial build (to download dependencies)
- A running OpenAI-compatible model server (LMStudio, Ollama, etc.)

## Getting Started

### Initial Setup

```bash
# Clone or navigate to the rust-open-agent-sdk directory
cd rust-open-agent-sdk

# Build the project (downloads dependencies on first run)
cargo build

# Run tests (67 total tests)
cargo test

# Run tests with output
cargo test -- --nocapture

# Build in release mode (optimized)
cargo build --release
```

### Running Examples

```bash
# Make sure you have a model server running
# For example, Ollama on http://localhost:11434

# Run the simple query example
cargo run --example simple_query

# Run production examples
cargo run --example git_commit_agent
cargo run --example log_analyzer_agent /path/to/log/file

# Run advanced patterns demo
cargo run --example advanced_patterns

# With environment variables
OPEN_AGENT_MODEL=qwen3:8b \
OPEN_AGENT_BASE_URL=http://localhost:11434/v1 \
cargo run --example simple_query
```

## Project Structure

```
rust-open-agent-sdk/
├── Cargo.toml              # Project configuration and dependencies
├── src/
│   ├── lib.rs              # Main library entry point
│   ├── client.rs           # Client and query function (5 tests)
│   ├── types.rs            # Core types and data structures (14 tests)
│   ├── config.rs           # Configuration helpers (4 tests)
│   ├── error.rs            # Error types (10 tests)
│   ├── tools.rs            # Tool system (5 tests)
│   ├── hooks.rs            # Lifecycle hooks (4 tests)
│   ├── context.rs          # Context management (7 tests)
│   ├── retry.rs            # Retry utilities (6 tests)
│   └── utils.rs            # Streaming utilities (2 tests)
├── examples/
│   ├── simple_query.rs               # Basic usage
│   ├── calculator_tools.rs           # Tool system demo
│   ├── hooks_example.rs              # Hooks and validation
│   ├── context_management.rs         # Token estimation
│   ├── interrupt_demo.rs             # Cancellation patterns
│   ├── git_commit_agent.rs           # Production example
│   ├── log_analyzer_agent.rs         # Production example
│   └── advanced_patterns.rs          # Retry & concurrency
├── tests/
│   └── integration_tests.rs          # Integration tests (10 tests)
├── README.md                          # Project overview
├── DEVELOPMENT.md                     # This file
└── LICENSE                            # MIT license
```

## Current Status: Phase 7 Complete (90% Feature Parity)

### Implemented Features

✅ **Core System**
- AgentOptions with builder pattern
- Client for multi-turn conversations
- Streaming response handling (SSE)
- Comprehensive error handling
- Configuration helpers (Provider shortcuts)

✅ **Tool System**
- Tool definition with builder pattern
- Automatic JSON Schema generation
- Manual tool execution
- Tool result handling

✅ **Lifecycle Hooks**
- UserPromptSubmit hook (integrated)
- PreToolUse hook (defined)
- PostToolUse hook (defined)
- HookDecision for control flow
- Async hook handlers

✅ **Context Management**
- Token estimation (character-based)
- Message truncation with system preservation
- Context limit checking
- Manual history management

✅ **Interrupt Capability**
- Atomic interrupt flag
- Safe concurrent access
- Idempotent operation
- Stream cleanup on interrupt

✅ **Advanced Features**
- Retry logic with exponential backoff
- Configurable backoff and jitter
- Conditional retry (transient errors only)
- Concurrent request patterns
- Rate limiting with semaphores

✅ **Production Examples**
- Git commit agent (~400 LOC)
- Log analyzer agent (~350 LOC)
- 8 comprehensive examples total

✅ **Testing**
- 67 total tests (57 unit + 10 integration)
- 100% module coverage
- Error conditions tested
- Serialization validated

## Development Workflow

### Adding a New Feature

1. **Write tests first** (TDD approach)
   ```bash
   # Add tests to appropriate module
   vim src/my_module.rs

   # Run tests to see them fail
   cargo test
   ```

2. **Implement the feature**
   ```bash
   # Implement functionality
   cargo build
   ```

3. **Verify tests pass**
   ```bash
   cargo test
   cargo test -- --nocapture  # With output
   ```

4. **Add documentation**
   ```rust
   /// Comprehensive doc comments
   ///
   /// # Examples
   ///
   /// ```rust
   /// // Usage example
   /// ```
   pub fn my_function() {}
   ```

5. **Run clippy for linting**
   ```bash
   cargo clippy
   cargo clippy -- -W clippy::all
   ```

6. **Format code**
   ```bash
   cargo fmt
   ```

### Running Specific Tests

```bash
# Run all tests
cargo test

# Run tests for specific module
cargo test --lib context

# Run specific test
cargo test test_agent_options_builder

# Run integration tests only
cargo test --test integration_tests

# Run doc tests
cargo test --doc
```

### Building Documentation

```bash
# Build and open docs locally
cargo doc --open

# Build docs for all dependencies
cargo doc --open --no-deps
```

## Code Style Guidelines

### Rust Idioms

- **Use builder patterns** for complex configuration
- **Prefer `?` operator** for error propagation
- **Use `Arc` for shared ownership**, `Mutex` for interior mutability
- **Async functions** with `async fn` and `await`
- **Pin<Box<dyn Future>>** for boxed futures
- **Result<T, Error>** for fallible operations

### Error Handling

```rust
// Use custom Error type with thiserror
use crate::{Error, Result};

pub fn operation() -> Result<T> {
    // Use ? for propagation
    let value = fallible_call()?;

    // Create custom errors
    if !valid {
        return Err(Error::invalid_input("reason"));
    }

    Ok(value)
}
```

### Async Patterns

```rust
// Async functions
pub async fn query(prompt: &str) -> Result<T> {
    let response = http_client.post(url)
        .send()
        .await?;
    Ok(response)
}

// Streaming with Pin<Box<dyn Stream>>
use futures::stream::Stream;

pub fn stream() -> Pin<Box<dyn Stream<Item = Result<T>> + Send>> {
    Box::pin(futures::stream::iter(items))
}
```

### Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_operation() {
        let result = operation();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_async_operation() {
        let result = async_operation().await;
        assert!(result.is_ok());
    }
}
```

## Common Tasks

### Adding a New Example

1. Create file in `examples/` directory
2. Add `[[example]]` entry to `Cargo.toml`
3. Include comprehensive documentation
4. Test with: `cargo run --example your_example`

### Adding a New Module

1. Create file in `src/` directory
2. Add `mod module_name;` to `src/lib.rs`
3. Export public API: `pub use module_name::{...};`
4. Add tests at bottom of module
5. Update documentation

### Benchmarking (Optional)

```bash
# Add to Cargo.toml:
# [dev-dependencies]
# criterion = "0.5"

# Create benches/ directory
# Run benchmarks
cargo bench
```

## Debugging

### Enable Logging

```rust
use tracing::{info, debug, error};
use tracing_subscriber;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    info!("Starting application");
    debug!("Debug details");
}
```

### Common Issues

**Issue: Borrow checker errors with async closures**
```rust
// Problem: Can't capture mutable reference
let mut count = 0;
let closure = || async { count += 1; };  // Error!

// Solution: Use Arc<Atomic> or Arc<Mutex>
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
let count = Arc::new(AtomicUsize::new(0));
let count_clone = count.clone();
let closure = move || async move {
    count_clone.fetch_add(1, Ordering::SeqCst);
};
```

**Issue: Send + Sync bounds on async functions**
```rust
// Ensure futures are Send
pub fn returns_future() -> Pin<Box<dyn Future<Output = ()> + Send>> {
    Box::pin(async move {
        // Implementation
    })
}
```

## Performance Optimization

### Release Builds

```bash
# Build with optimizations
cargo build --release

# Binary location
target/release/your_binary

# Strip symbols for smaller binary
strip target/release/your_binary
```

### Profile-Guided Optimization

```bash
# Build with PGO
RUSTFLAGS="-C profile-generate=/tmp/pgo-data" cargo build --release
# Run workload
target/release/your_binary
# Rebuild with profile data
RUSTFLAGS="-C profile-use=/tmp/pgo-data" cargo build --release
```

## CI/CD Integration

The project is ready for CI/CD with GitHub Actions or similar:

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - run: cargo test --all-features
      - run: cargo clippy -- -D warnings
      - run: cargo fmt -- --check
```

## Publishing to Crates.io

### Pre-publish Checklist

- [ ] All tests passing (`cargo test`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Code formatted (`cargo fmt`)
- [ ] Documentation complete (`cargo doc`)
- [ ] README.md updated
- [ ] Cargo.toml metadata complete
- [ ] Version bumped appropriately

### Publishing Steps

```bash
# Dry run
cargo publish --dry-run

# Publish for real
cargo publish

# After publishing, tag the release
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

## Getting Help

- **Documentation**: https://docs.rs/open-agent-sdk
- **Repository**: https://github.com/slb350/open-agent-sdk
- **Issues**: https://github.com/slb350/open-agent-sdk/issues
- **Rust Book**: https://doc.rust-lang.org/book/
- **Async Book**: https://rust-lang.github.io/async-book/

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE for details
