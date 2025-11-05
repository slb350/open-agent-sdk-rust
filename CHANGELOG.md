# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-15

### Added

#### Core Features
- **Streaming API**: Single-query `query()` function with async streaming responses
- **Multi-turn Client**: Stateful `Client` for conversation history management
- **Tool System**: Full function calling support with `tool()` builder
  - Type-safe parameter definitions
  - Async tool execution
  - Automatic tool result handling
- **Auto-execution Mode**: Automatic tool calling loop (`auto_execute_tools` option)
  - Configurable iteration limits
  - Transparent tool execution
  - Error handling and recovery

#### Advanced Features
- **Lifecycle Hooks**: Three extension points for custom logic
  - `PreToolUse`: Intercept before tool execution
  - `PostToolUse`: Process tool results
  - `UserPromptSubmit`: Transform user prompts before sending
- **Context Management**: Utilities for token budget management
  - `estimate_tokens()`: Approximate token counting
  - `truncate_messages()`: Smart message history truncation
  - `is_approaching_limit()`: Token budget monitoring
- **Interrupt Capability**: Cancel long-running operations via `client.interrupt()`
- **Retry Logic**: Exponential backoff with jitter
  - Configurable max retries and delays
  - Automatic retry on transient failures
  - Detailed error context

#### Configuration
- **AgentOptions Builder**: Fluent API for configuration
  - System prompts
  - Model selection
  - Temperature and sampling parameters
  - Token limits and turn limits
  - Base URL for local servers

#### Quality & Documentation
- **85+ Comprehensive Tests**:
  - 57 unit tests across 10 modules
  - 28 integration tests (hooks, auto-execution, advanced patterns)
  - Full test coverage for all features
- **10 Production Examples**:
  - `simple_query.rs` - Basic usage
  - `calculator_tools.rs` - Tool system demo
  - `hooks_example.rs` - Lifecycle hooks
  - `context_management.rs` - Token management patterns
  - `interrupt_demo.rs` - Interrupt capability
  - `git_commit_agent.rs` - Real-world agent (Git commits)
  - `log_analyzer_agent.rs` - Real-world agent (log analysis)
  - `advanced_patterns.rs` - Concurrent operations
  - `auto_execution_demo.rs` - Auto-execution patterns
  - `multi_tool_agent.rs` - Multiple tool coordination
- **CI/CD Pipeline**: GitHub Actions with 8 parallel jobs
  - Formatting (rustfmt)
  - Linting (clippy)
  - Matrix testing (Ubuntu + macOS × stable + beta)
  - MSRV verification (Rust 1.83)
  - Security audit (cargo-audit)
  - Documentation validation
  - Code coverage (tarpaulin + Codecov)
  - Benchmark comparison (PR only)
- **Performance Benchmarks**: Criterion-based benchmark suite
  - Token estimation benchmarks
  - Message truncation performance
  - Tool execution overhead

#### Documentation
- Comprehensive API documentation with examples
- Crate-level quick start guide
- Module-level documentation
- Doc tests for all public APIs

### Technical Details
- **Rust Edition**: 2021
- **MSRV**: 1.83.0
- **License**: MIT
- **Platform Support**: Linux, macOS, Windows
- **Async Runtime**: Tokio
- **HTTP Client**: reqwest with streaming support

### Compatibility
- Works with any OpenAI-compatible API server:
  - LM Studio (localhost:1234)
  - Ollama (localhost:11434)
  - llama.cpp server
  - vLLM
  - Any other OpenAI-compatible endpoint

[0.1.0]: https://github.com/slb350/open-agent-sdk-rust/releases/tag/v0.1.0
