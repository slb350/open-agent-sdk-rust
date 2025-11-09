//! # Open Agent SDK - Rust Implementation
//!
//! A production-ready, streaming-first Rust SDK for building AI agents with local OpenAI-compatible servers.
//!
//! ## Overview
//!
//! This SDK provides a clean, ergonomic API for working with local LLM servers such as:
//! - LM Studio
//! - Ollama
//! - llama.cpp
//! - vLLM
//!
//! ## Key Features
//!
//! - **Zero API Costs**: Run models on your own hardware
//! - **Privacy-First**: All data stays local on your machine
//! - **High Performance**: Native async/await with Tokio runtime
//! - **Streaming Responses**: Real-time token-by-token streaming
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
//! use open_agent::{query, AgentOptions, ContentBlock};
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
//!     // Process each content block as it arrives
//!     while let Some(block) = stream.next().await {
//!         match block? {
//!             ContentBlock::Text(text_block) => {
//!                 print!("{}", text_block.text);
//!             }
//!             ContentBlock::ToolUse(tool_block) => {
//!                 println!("Tool called: {}", tool_block.name);
//!             }
//!             ContentBlock::ToolResult(_) => {
//!                 // Tool results can be ignored in simple queries
//!             }
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
//!         if let ContentBlock::Text(text) = block {
//!             print!("{}", text.text);
//!         }
//!     }
//!
//!     // Second turn - client remembers previous context
//!     client.send("What about if we multiply that by 3?").await?;
//!     while let Some(block) = client.receive().await? {
//!         if let ContentBlock::Text(text) = block {
//!             print!("{}", text.text);
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
//! - **client**: Core streaming query engine and multi-turn client
//! - **types**: Data structures for messages, content blocks, and configuration
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
/// Includes builder patterns for ergonomic configuration and OpenAI API serialization.
mod types;

/// Internal utilities for Server-Sent Events (SSE) parsing and tool call aggregation.
/// Handles the low-level details of streaming response parsing.
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

pub use client::{Client, query};

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
    AgentOptions, AgentOptionsBuilder, ContentBlock, Message, MessageRole, TextBlock,
    ToolResultBlock, ToolUseBlock,
};

// ============================================================================
// CONVENIENCE PRELUDE
// ============================================================================

/// Convenience module containing the most commonly used types and functions.
/// Import with `use open_agent::prelude::*;` to get everything you need for typical usage.
///
/// This includes:
/// - Configuration: AgentOptions, AgentOptionsBuilder
/// - Client: Client, query()
/// - Content: ContentBlock, TextBlock, ToolUseBlock
/// - Tools: Tool, tool()
/// - Hooks: Hooks, HookDecision, hook event types
/// - Errors: Error, Result
pub mod prelude {
    pub use crate::{
        AgentOptions, AgentOptionsBuilder, Client, ContentBlock, Error, HookDecision, Hooks,
        PostToolUseEvent, PreToolUseEvent, Result, TextBlock, Tool, ToolUseBlock,
        UserPromptSubmitEvent, query, tool,
    };
}
