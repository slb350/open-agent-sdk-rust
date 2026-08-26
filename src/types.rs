//! Core type definitions for the Open Agent SDK.
//!
//! This module contains the fundamental data structures used throughout the SDK for
//! configuring and interacting with AI agents. The type system is organized into three
//! main categories:
//!
//! # Agent Configuration
//!
//! - [`AgentOptions`]: Main configuration struct for agent behavior, model settings,
//!   and tool management
//! - [`AgentOptionsBuilder`]: Builder pattern implementation for constructing
//!   [`AgentOptions`] with validation
//!
//! # Message System
//!
//! The SDK uses a flexible message system that supports multi-modal content:
//!
//! - [`Message`]: Container for conversation messages with role and content
//! - [`MessageRole`]: Enum defining who sent the message (System, User, Assistant, Tool)
//! - [`ContentBlock`]: Enum for different content types (text, tool use, tool results)
//! - [`TextBlock`]: Simple text content
//! - [`ToolUseBlock`]: Represents an AI request to execute a tool
//! - [`ToolResultBlock`]: Contains the result of a tool execution
//!
//! # Wire Formats
//!
//! [`ApiProtocol`] selects which wire format an endpoint speaks: `OpenAiChat` (the default)
//! or `Anthropic`. `OpenAIRequest` stays the single internal request representation, and the
//! translation to the Anthropic shape happens at the transport boundary.
//!
//! OpenAI chat completions:
//!
//! - [`OpenAIRequest`]: Request payload sent to the API
//! - [`OpenAIMessage`]: Message format for OpenAI API
//! - [`OpenAIChunk`]: Streaming response chunk from the API
//! - [`OpenAIToolCall`], [`OpenAIFunction`]: Tool calling format
//! - [`OpenAIDelta`], [`OpenAIToolCallDelta`]: Incremental updates in streaming
//!
//! Anthropic messages:
//!
//! - [`AnthropicRequest`], [`AnthropicMessage`]: Request payload, built with
//!   [`AnthropicRequest::from_openai`]
//! - [`AnthropicEvent`]: One decoded event from the streaming vocabulary
//! - [`AnthropicBlockStart`], [`AnthropicDelta`], [`AnthropicMessageDelta`]: Content block
//!   openings and incremental updates
//! - [`AnthropicErrorBody`]: A mid-stream `error` event
//! - [`anthropic_finish_reason`]: Maps Anthropic stop reasons onto [`FinishReason`]
//!
//! # Architecture Overview
//!
//! The type system is designed to:
//!
//! 1. **Separate concerns**: Internal SDK types (Message, ContentBlock) are distinct
//!    from the API wire formats, allowing flexibility in provider support
//! 2. **Enable streaming**: OpenAI types support incremental delta parsing for
//!    real-time responses
//! 3. **Support tool use**: First-class support for function calling with proper
//!    request/response tracking
//! 4. **Provide ergonomics**: Builder pattern and convenience constructors make
//!    common operations simple
//!
//! # Example
//!
//! ```no_run
//! use open_agent::{AgentOptions, Message};
//!
//! // Build agent configuration
//! let options = AgentOptions::builder()
//!     .model("qwen2.5-32b-instruct")
//!     .base_url("http://localhost:1234/v1")
//!     .system_prompt("You are a helpful assistant")
//!     .max_turns(10)
//!     .auto_execute_tools(true)
//!     .build()
//!     .expect("Valid configuration");
//!
//! // Create a user message
//! let msg = Message::user("Hello, how are you?");
//! ```

use crate::Error;
use crate::hooks::Hooks;
use crate::tools::Tool;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

include!("types/validated.rs");
include!("types/agent_options.rs");
include!("types/agent_options_builder.rs");
include!("types/message_blocks.rs");
include!("types/image.rs");
include!("types/message.rs");
include!("types/openai.rs");
// Real submodules rather than `include!` fragments, so `cargo-mutants` — which walks `mod`
// declarations but does not expand `include!` — can mutate the logic they hold. The
// re-exports keep every public path unchanged. The remaining fragments stay fragments
// because they reach into module-private fields of types their siblings define, which real
// modules could only reach by weakening that encapsulation.
mod anthropic;
mod anthropic_stream;
pub(crate) mod http_headers;
mod openai_stream;
mod protocol;
mod stream_event;

// Glob re-export: the wire types are consumed across the crate (and by unit tests) exactly
// as the `include!` fragment used to expose them, so this keeps every path identical.
pub use openai_stream::*;
pub use stream_event::{FinishReason, StreamEvent};

pub use anthropic::{AnthropicMessage, AnthropicRequest};
pub use anthropic_stream::{
    AnthropicBlockStart, AnthropicDelta, AnthropicErrorBody, AnthropicEvent, AnthropicMessageDelta,
    anthropic_finish_reason,
};
pub use protocol::ApiProtocol;

#[cfg(test)]
mod tests {
    use super::*;

    include!("types/tests/core.rs");
    include!("types/tests/image.rs");
}
