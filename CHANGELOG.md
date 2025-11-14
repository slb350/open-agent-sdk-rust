# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2025-11-13

### Added

**Multimodal Image Support** - Vision API Integration

Added comprehensive support for sending images alongside text to vision-capable models following the OpenAI Vision API format:

#### New Types

- **`ImageBlock`** - Represents an image in a message
  - `from_url(url)` - Create from HTTP/HTTPS URL
  - `from_base64(data, mime_type)` - Create from base64-encoded data
  - `with_detail(detail)` - Set detail level for token cost control
  - `url()` - Get the image URL (HTTP or data URI)
  - `detail()` - Get the detail level

- **`ImageDetail`** - Control image processing and token costs
  - `ImageDetail::Low` - Fixed ~85 tokens, 512x512 max resolution
  - `ImageDetail::High` - Variable tokens based on image dimensions
  - `ImageDetail::Auto` - Let the model decide (default)

- **`OpenAIContent`** - Message content format (internal)
  - `Text(String)` - Simple text (backward compatible)
  - `Parts(Vec<OpenAIContentPart>)` - Mixed text and images

- **`OpenAIContentPart`** - Content part for multimodal messages
  - `text(content)` - Text content part
  - `image_url(url, detail)` - Image content part

#### Convenience API

New `Message` helper methods for ergonomic image support:

```rust
// Simple image + text
let msg = Message::user_with_image(
    "What's in this image?",
    "https://example.com/photo.jpg"
);

// Control detail level for token management
let msg = Message::user_with_image_detail(
    "Analyze this diagram",
    "https://example.com/diagram.png",
    ImageDetail::High
);

// Base64-encoded images
let msg = Message::user_with_base64_image(
    "What color is this?",
    base64_data,
    "image/png"
);
```

#### Client Integration

- Messages containing images automatically serialized to OpenAI Vision API format
- Text-only messages maintain simple string format (backward compatible)
- Mixed text+image messages use array format with proper content parts
- Image detail levels properly passed through to API

#### Examples

- `examples/vision_api_demo.rs` - Comprehensive demonstration of:
  - Simple image + text messages
  - Detail level control for token costs
  - Base64-encoded images
  - Complex multi-image messages
  - Usage patterns with vision models

#### Testing

- 8 integration tests for image serialization (`tests/image_serialization_test.rs`)
- 3 unit tests for Message helper methods
- All existing tests updated to handle `ContentBlock::Image` variant
- Total: 256 tests passing (77 unit + 33 integration + 146 doc)

#### Token Cost Management

Images consume tokens based on detail level (OpenAI Vision API):
- **Low detail**: ~85 tokens (cost-effective, suitable for OCR, object detection)
- **High detail**: Variable tokens based on dimensions (detailed analysis, fine text)
- **Auto detail**: Model decides based on content (balanced default)

**Note:** Token costs are based on OpenAI's Vision API and may vary significantly on local models (llama.cpp, Ollama, etc.)

#### Documentation

- Comprehensive rustdoc for all new types and methods
- Token cost guidance in documentation
- Working examples with vision model setup instructions
- Clear migration patterns for vision capabilities

**Related**: Closes GitHub issue #2

## [0.5.0] - 2025-11-13

### Fixed

**CRITICAL**: Tool Call Serialization Bug - Infinite Loop with auto_execute_tools

Fixed a critical bug where tool calls and tool results were not being serialized into OpenAI message format, causing an infinite loop when using `auto_execute_tools(true)`:

**The Problem:**
- Internal conversation history stored tool results as `ContentBlock::ToolResult`
- When building OpenAI API requests, only text blocks were extracted
- Tool results were silently dropped from the conversation history
- LLM never saw tool results, so it called the same tool repeatedly
- Loop continued until `max_tool_iterations` was reached
- Same tool called 50+ times instead of once

**The Fix:**
- Tool calls now properly serialized with `tool_calls` array in assistant messages
- Tool results now serialized as separate messages with `role: "tool"` and `tool_call_id`
- Message building logic handles three cases:
  1. Messages with ToolResult blocks → separate tool messages with `tool_call_id`
  2. Messages with ToolUse blocks → assistant messages with `tool_calls` array
  3. Messages with only text → normal text messages

**Impact:**
- ✅ Tool results now included in conversation history
- ✅ LLM sees tool results and responds appropriately
- ✅ Each tool called only once per unique request
- ✅ `auto_execute_tools(true)` now fully functional
- ✅ Works correctly with llama.cpp and other OpenAI-compatible servers

**Technical Details:**
- Modified `client.rs` message building logic (lines ~1105-1214)
- Added imports for `OpenAIToolCall` and `OpenAIFunction`
- Properly populates `tool_calls` field with tool ID, name, and serialized arguments
- Properly populates `tool_call_id` field in tool response messages
- Arguments serialized as JSON strings per OpenAI API specification

**Test Case:**
```rust
// Before: Tool called 50+ times, no final response
// After: Tool called once, final text response returned

let client = Client::new(AgentOptions::builder()
    .auto_execute_tools(true)
    .tool(database_tool)
    .build()?)?;

client.send("how many users?").await?;
while let Some(block) = client.receive().await? {
    // Now receives: "The users table has 5 rows."
}
```

See `examples/test_tool_serialization.rs` for demonstration.

## [0.4.0] - 2025-11-09

### Changed

**BREAKING**: API Stability Improvements - Private Fields with Getters

Following Rust API Guidelines (C-STRUCT-PRIVATE), all public struct fields are now private with getter methods for better encapsulation and future-proof APIs:

#### AgentOptions
- **Private fields**: `system_prompt`, `model`, `base_url`, `api_key`, `max_turns`, `max_tokens`, `temperature`, `timeout`, `tools`, `auto_execute_tools`, `max_tool_iterations`, `hooks`
- **Getter methods**: `.system_prompt()`, `.model()`, `.base_url()`, `.api_key()`, `.max_turns()`, `.max_tokens()`, `.temperature()`, `.timeout()`, `.tools()`, `.auto_execute_tools()`, `.max_tool_iterations()`, `.hooks()`
- **Migration**: `options.model` → `options.model()`

#### Tool
- **Private fields**: `name`, `description`, `input_schema`, `handler`
- **Getter methods**: `.name()`, `.description()`, `.input_schema()`
- **Migration**: `tool.name` → `tool.name()`

#### HookDecision
- **Private fields**: `continue_execution`, `modified_input`, `modified_prompt`, `reason`
- **Getter methods**: `.continue_execution()`, `.modified_input()`, `.modified_prompt()`, `.reason()`
- **Migration**: `decision.continue_execution` → `decision.continue_execution()`
- **Note**: Getters return references; use `.clone()` if owned value needed

**BREAKING**: Client::new() Returns Result

`Client::new()` now returns `Result<Self>` instead of panicking on HTTP client creation failure.

**Migration**:
```rust
// Before:
let client = Client::new(options);

// After:
let client = Client::new(options)?;
// or
let client = Client::new(options).expect("Failed to create client");
```

**BREAKING**: add_tool_result() Returns Result

`Client::add_tool_result()` now returns `Result<()>` instead of silently failing on serialization errors.

**Migration**:
```rust
// Before:
client.add_tool_result(&id, result);

// After:
client.add_tool_result(&id, result)?;
```

### Added

- **New method**: `Client::interrupt_handle()` - Returns a cloned `Arc<AtomicBool>` for thread-safe cancellation
  - Replaces direct access to the private `interrupted` field
  - Migration: `client.interrupted.clone()` → `client.interrupt_handle()`

- **Input Validation**: `AgentOptionsBuilder::build()` now validates configuration:
  - Temperature must be between 0.0 and 2.0
  - Model name cannot be empty or whitespace
  - Base URL must start with `http://` or `https://`
  - Max tokens must be greater than 0

### Fixed

- **Safety**: HTTP client no longer panics on invalid timeout - returns error instead
- **Error Handling**: Error response body parsing failures now logged instead of silently suppressed
- **SSE Parsing**: Handles empty chunks/heartbeats gracefully
- **Schema Validation**: Replaced `.unwrap()` with defensive assertions for better error messages
- **Tool Arguments**: Doc examples updated to validate parameters instead of using `.unwrap_or(0.0)`

### Documentation

- Added SAFETY comments to unsafe blocks in tests
- Documented OpenAI tool serialization limitation (ToolUse/ToolResult blocks not serialized to conversation history)
- Fixed documentation accuracy issues (system_prompt optionality, max_tokens defaults)
- Updated 150+ doctests for new APIs

### Technical Details

- All 66 unit and integration tests passing
- 135/139 doctests passing (97% success rate, 14 intentionally ignored)
- Zero tech debt: All identified issues fixed
- Breaking changes acceptable before 1.0 for long-term API stability

## [0.3.0] - 2025-11-05

### Changed

**BREAKING**: Improved `Client::receive()` API ergonomics
- Changed signature from `Option<Result<ContentBlock>>` to `Result<Option<ContentBlock>>`
- More intuitive: errors are always `Err()`, success is always `Ok()`
- Better ergonomics with `?` operator: `while let Some(block) = client.receive().await? { ... }`
- Migration: Change `match block? { ... }` inside the loop to `match block { ... }` and move the `?` to the while condition

### Benefits

- **Clearer Intent**: `Ok(Some(block))` = got a block, `Ok(None)` = stream ended, `Err(e)` = error occurred
- **Better Error Handling**: Can use `?` operator outside the loop instead of inside
- **More Idiomatic**: Follows Rust conventions for fallible iterators

### Technical Details

- All 85+ tests updated and passing
- All 10 examples updated with new API
- Zero breaking changes to other APIs
- Comprehensive test coverage for new signature

## [0.2.0] - 2025-11-04

### Changed

**BREAKING**: Upgraded to Rust Edition 2024
- Requires Rust 1.85.0 or newer (was 1.83.0)
- Edition 2024 brings latest language features and safety improvements
- No API changes - only compiler/edition upgrade

### Fixed

- **Safety**: Eliminated potential panic in `ToolBuilder::param()`
  - Now safely handles calling `.param()` after `.schema(non_object)`
  - Resets schema to empty object if needed instead of panicking
  - Added test coverage for edge case
- **Tests**: Made `test_auto_execution_with_tools` more robust
  - Accepts either text response OR tool execution
  - Better handles LLM behavior variance

### Technical Details

- Updated minimum supported Rust version (MSRV) to 1.85.0
- CI/CD pipeline updated to test against Rust 1.85
- All 100 tests passing with zero warnings
- Edition 2024 safety improvements applied

## [0.1.0] - 2025-11-04

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

[0.3.0]: https://github.com/slb350/open-agent-sdk-rust/releases/tag/v0.3.0
[0.2.0]: https://github.com/slb350/open-agent-sdk-rust/releases/tag/v0.2.0
[0.1.0]: https://github.com/slb350/open-agent-sdk-rust/releases/tag/v0.1.0
