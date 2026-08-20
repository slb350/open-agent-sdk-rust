//! # Open Agent SDK - Rust Implementation
//!
//! A production-ready, streaming-first Rust SDK for building AI agents over two wire
//! protocols: OpenAI chat completions and Anthropic messages.
//!
//! ## Overview
//!
//! The protocol is a property of the endpoint. Select it with
//! `AgentOptions::builder().protocol(..)`; it defaults to [`ApiProtocol::OpenAiChat`].
//!
//! [`ApiProtocol::OpenAiChat`] posts to `{base_url}/chat/completions` with bearer auth:
//! - LM Studio, Ollama, llama.cpp, vLLM, and other local servers
//! - OpenAI, OpenRouter, z.ai, and other hosted OpenAI-compatible endpoints
//!
//! [`ApiProtocol::Anthropic`] posts to `{base_url}/messages` with `x-api-key` and
//! `anthropic-version`:
//! - Anthropic
//! - Moonshot Kimi for Coding, MiniMax, and other Anthropic-shaped endpoints
//!
//! ## Key Features
//!
//! - **Two Wire Protocols**: OpenAI chat completions or Anthropic messages, per endpoint
//! - **Local or Hosted**: Zero-cost inference on your own hardware, or a vendor endpoint
//! - **High Performance**: Native async/await with Tokio runtime
//! - **Streaming Responses**: Real-time token-by-token streaming
//! - **Finish Reasons**: Every stream reports why generation stopped
//! - **Reasoning Channel**: Extended thinking and reasoning deltas kept out of content
//! - **Tool Calling**: Define and execute tools with automatic schema generation
//! - **Lifecycle Hooks**: Intercept and control execution at key points
//! - **Interrupts**: Gracefully cancel long-running operations
//! - **Context Management**: Manual token estimation and history truncation
//! - **Retry Logic**: Exponential backoff with jitter for reliability
//!
//! ## Two Interaction Modes
//!
//! ### 1. Simple Query Function (`query()`)
//! For single-turn interactions without conversation state:
//!
//! ```rust,no_run
//! use open_agent::{query, AgentOptions, ContentBlock, FinishReason, StreamEvent};
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Configure the agent with required settings
//!     let options = AgentOptions::builder()
//!         .system_prompt("You are a helpful assistant")
//!         .model("qwen2.5-32b-instruct")
//!         .base_url("http://localhost:1234/v1")
//!         .build()?;
//!
//!     // Send a single query and stream the response
//!     let mut stream = query("What's the capital of France?", &options).await?;
//!
//!     // Process each event as it arrives; the stream always ends with one Finish
//!     while let Some(event) = stream.next().await {
//!         match event? {
//!             StreamEvent::Block(ContentBlock::Text(text_block)) => {
//!                 print!("{}", text_block.text);
//!             }
//!             StreamEvent::Block(ContentBlock::ToolUse(tool_block)) => {
//!                 println!("Tool called: {}", tool_block.name());
//!             }
//!             StreamEvent::Finish(FinishReason::Length) => {
//!                 eprintln!("response was truncated at the token cap");
//!             }
//!             _ => {}
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### 2. Client Object (`Client`)
//! For multi-turn conversations with persistent state:
//!
//! ```rust,no_run
//! use open_agent::{Client, AgentOptions, ContentBlock};
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let options = AgentOptions::builder()
//!         .system_prompt("You are a helpful assistant")
//!         .model("qwen2.5-32b-instruct")
//!         .base_url("http://localhost:1234/v1")
//!         .build()?;
//!
//!     // Create a stateful client that maintains conversation history
//!     let mut client = Client::new(options)?;
//!
//!     // First turn
//!     client.send("What's 2+2?").await?;
//!     while let Some(block) = client.receive().await? {
//!         match block {
//!             ContentBlock::Text(text) => print!("{}", text.text),
//!             ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {}
//!         }
//!     }
//!
//!     // Second turn - client remembers previous context
//!     client.send("What about if we multiply that by 3?").await?;
//!     while let Some(block) = client.receive().await? {
//!         match block {
//!             ContentBlock::Text(text) => print!("{}", text.text),
//!             ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {}
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! The SDK is organized into several modules, each with a specific responsibility:
//!
//! - **client**: Core streaming query engine and multi-turn client, and the transport
//!   boundary where the protocol is applied
//! - **types**: Data structures for messages, content blocks, configuration, and the
//!   OpenAI and Anthropic wire formats
//! - **tools**: Tool definition system with automatic JSON schema generation
//! - **hooks**: Lifecycle event system for intercepting execution
//! - **config**: Provider-specific configuration helpers
//! - **error**: Comprehensive error types and conversions
//! - **context**: Token estimation and message truncation utilities
//! - **retry**: Exponential backoff retry logic with jitter
//! - **utils**: Internal utilities for SSE parsing and tool aggregation

// ============================================================================
// MODULE DECLARATIONS
// ============================================================================
// These modules are private (internal implementation details) unless explicitly
// re-exported through `pub use` statements below.

/// Core client implementation providing streaming queries and stateful conversations.
/// Contains the `query()` function for single-turn queries and `Client` struct
/// for multi-turn conversations with automatic state management.
mod client;

/// Provider configuration helpers for LM Studio, Ollama, llama.cpp, and vLLM.
/// Simplifies endpoint and model name resolution with environment variable support.
mod config;

/// Context window management utilities for token estimation and history truncation.
/// Provides manual control over conversation memory to prevent context overflow.
mod context;

/// Error types and conversions for comprehensive error handling throughout the SDK.
/// Defines the `Error` enum and `Result<T>` type alias used across all public APIs.
mod error;

/// Lifecycle hooks system for intercepting and controlling execution at key points.
/// Enables security gates, audit logging, input/output modification, and compliance checks.
mod hooks;

/// Tool definition and execution system with automatic JSON schema generation.
/// Allows LLMs to call Rust functions with type-safe parameter handling.
mod tools;

/// Core type definitions for messages, content blocks, and agent configuration.
/// Includes builder patterns for ergonomic configuration, the `ApiProtocol` selector, and
/// the OpenAI and Anthropic wire formats.
mod types;

/// Internal utilities for Server-Sent Events (SSE) parsing and tool call aggregation.
/// Handles the low-level details of streaming response parsing for both protocols.
mod utils;

// ============================================================================
// PUBLIC EXPORTS
// ============================================================================
// These items form the public API of the SDK. Everything else is internal.

/// Retry utilities with exponential backoff and jitter.
/// Made public as a module so users can access retry configuration and functions
/// for their own operations that need retry logic.
pub mod retry;

// --- Core Client API ---

pub use client::{Client, EventStream, query};

// --- Provider Configuration ---

pub use config::{Provider, get_base_url, get_model};

// --- Context Management ---

pub use context::{estimate_tokens, is_approaching_limit, truncate_messages};

// --- Error Handling ---

pub use error::{Error, Result};

// --- Lifecycle Hooks ---

pub use hooks::{
    HOOK_POST_TOOL_USE, HOOK_PRE_TOOL_USE, HOOK_USER_PROMPT_SUBMIT, HookDecision, Hooks,
    PostToolUseEvent, PreToolUseEvent, UserPromptSubmitEvent,
};

// --- Tool System ---

pub use tools::{Tool, ToolBuilder, tool};

// --- Core Types ---

pub use types::{
    AgentOptions, AgentOptionsBuilder, ApiProtocol, BaseUrl, ContentBlock, FinishReason,
    ImageBlock, ImageDetail, Message, MessageRole, ModelName, OpenAIContent, OpenAIContentPart,
    StreamEvent, Temperature, TextBlock, ToolResultBlock, ToolUseBlock,
};

// --- Anthropic Wire Format ---
//
// Exported for the same reason the OpenAI wire types are: a caller building a gateway, a
// recording proxy, or a test double needs to name what goes over the wire.

pub use types::{
    AnthropicBlockStart, AnthropicDelta, AnthropicErrorBody, AnthropicEvent, AnthropicMessage,
    AnthropicMessageDelta, AnthropicRequest, anthropic_finish_reason,
};

// `AnthropicRequest::from_openai` takes an `OpenAIRequest`, so the request half of the
// OpenAI wire format has to be nameable for that constructor to be callable at all.
pub use types::{OpenAIFunction, OpenAIMessage, OpenAIRequest, OpenAIToolCall};

// ============================================================================
// CONVENIENCE PRELUDE
// ============================================================================

/// Convenience module containing the most commonly used types and functions.
/// Import with `use open_agent::prelude::*;` to get everything you need for typical usage.
///
/// This includes:
/// - Configuration: AgentOptions, AgentOptionsBuilder, ApiProtocol
/// - Client: Client, query()
/// - Content: ContentBlock, TextBlock, ToolUseBlock
/// - Streaming: StreamEvent, FinishReason
/// - Tools: Tool, tool()
/// - Hooks: Hooks, HookDecision, hook event types
/// - Errors: Error, Result
pub mod prelude {
    pub use crate::{
        AgentOptions, AgentOptionsBuilder, ApiProtocol, BaseUrl, Client, ContentBlock, Error,
        FinishReason, HookDecision, Hooks, ModelName, PostToolUseEvent, PreToolUseEvent, Result,
        StreamEvent, Temperature, TextBlock, Tool, ToolUseBlock, UserPromptSubmitEvent, query,
        tool,
    };
}
