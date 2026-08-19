//! Client for streaming queries and multi-turn conversations
//!
//! This module provides the core streaming client implementation for the Open Agent SDK.
//! It handles communication with OpenAI-compatible APIs, manages conversation history,
//! and provides two modes of operation: manual and automatic tool execution.
//!
//! # Architecture Overview
//!
//! The SDK implements a **streaming-first architecture** where all responses from the model
//! are received as a stream of content blocks. This design enables:
//!
//! - **Progressive rendering**: Display text as it arrives without waiting for completion
//! - **Real-time tool execution**: Execute tools as they're requested by the model
//! - **Interruption support**: Cancel operations mid-stream without corrupting state
//! - **Memory efficiency**: Process large responses without buffering everything in memory
//!
//! ## Two Operating Modes
//!
//! ### 1. Manual Tool Execution Mode (default)
//!
//! In manual mode, the client streams content blocks directly to the caller. When the model
//! requests a tool, the caller receives a `ToolUseBlock`, executes the tool, adds the result
//! using `add_tool_result()`, and continues the conversation with another `send()` call.
//!
//! **Use cases**: Custom tool execution logic, interactive debugging, fine-grained control
//!
//! ### 2. Automatic Tool Execution Mode
//!
//! When `auto_execute_tools` is enabled, the client automatically executes tools and continues
//! the conversation until receiving a text-only response. The caller only receives the final
//! text blocks after all tool iterations complete.
//!
//! **Use cases**: Simple agentic workflows, automated task completion, batch processing
//!
//! ## Request Flow
//!
//! ```text
//! User sends prompt
//!     │
//!     ├─> UserPromptSubmit hook executes (can modify/block prompt)
//!     │
//!     ├─> Prompt added to history
//!     │
//!     ├─> HTTP request to OpenAI-compatible API
//!     │
//!     ├─> Response streamed as Server-Sent Events (SSE)
//!     │
//!     ├─> SSE chunks aggregated into ContentBlocks
//!     │
//!     └─> Blocks emitted to caller (or buffered for auto-execution)
//! ```
//!
//! ## Tool Execution Flow
//!
//! ### Manual Mode:
//! ```text
//! receive() → ToolUseBlock
//!     │
//!     ├─> Caller executes tool
//!     │
//!     ├─> Caller calls add_tool_result()
//!     │
//!     ├─> Caller calls send("") to continue
//!     │
//!     └─> receive() → TextBlock (model's response)
//! ```
//!
//! ### Auto Mode:
//! ```text
//! receive() triggers auto-execution loop
//!     │
//!     ├─> Collect all blocks from stream
//!     │
//!     ├─> For each ToolUseBlock:
//!     │   ├─> PreToolUse hook executes (can modify/block)
//!     │   ├─> Tool executed via Tool.execute()
//!     │   ├─> PostToolUse hook executes (can modify result)
//!     │   └─> Result added to history
//!     │
//!     ├─> Continue conversation with send("")
//!     │
//!     ├─> Repeat until text-only response or max iterations
//!     │
//!     └─> Return text blocks one-by-one via receive()
//! ```
//!
//! ## State Management
//!
//! The client maintains several pieces of state:
//!
//! - **history**: Full conversation history (`Vec<Message>`)
//! - **current_stream**: Active SSE stream being consumed (`Option<EventStream>`)
//! - **interrupted**: Atomic flag for cancellation (`Arc<AtomicBool>`)
//! - **auto_exec_buffer**: Buffered blocks for auto-execution mode (`Vec<ContentBlock>`)
//! - **auto_exec_index**: Current position in buffer (usize)
//!
//! ## Interruption Mechanism
//!
//! The interrupt system uses `Arc<AtomicBool>` to enable safe, thread-safe cancellation:
//!
//! ```rust,no_run
//! # use open_agent::{Client, AgentOptions};
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut client = Client::new(AgentOptions::default())?;
//! let handle = client.interrupt_handle(); // Clone Arc for use in other threads
//!
//! // In another thread or async task:
//! tokio::spawn(async move {
//!     tokio::time::sleep(std::time::Duration::from_secs(5)).await;
//!     handle.store(true, std::sync::atomic::Ordering::SeqCst);
//! });
//!
//! client.send("Long request").await?;
//! while let Some(block) = client.receive().await? {
//!     // Will stop when interrupted
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Hook Integration
//!
//! Hooks provide extension points throughout the request lifecycle:
//!
//! - **UserPromptSubmit**: Called before sending user prompt (can modify or block)
//! - **PreToolUse**: Called before executing each tool (can modify input or block execution)
//! - **PostToolUse**: Called after tool execution (can modify result)
//!
//! Hooks are only invoked in specific scenarios and have access to conversation history.
//!
//! ## Error Handling
//!
//! Errors are propagated immediately and leave the client in a valid state:
//!
//! - **HTTP errors**: Network failures, timeouts, connection issues
//! - **API errors**: Invalid model, authentication failures, rate limits
//! - **Parse errors**: Malformed SSE responses, invalid JSON
//! - **Tool errors**: Tool execution failures (converted to JSON error responses)
//! - **Hook errors**: Hook execution failures or blocked operations
//!
//! After an error, the client remains usable for new requests.
//!
//! # Examples
//!
//! ## Simple Single-Turn Query
//!
//! ```rust,no_run
//! use open_agent::{query, AgentOptions, ContentBlock};
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let options = AgentOptions::builder()
//!         .model("gpt-4")
//!         .api_key("sk-...")
//!         .build()?;
//!
//!     let mut stream = query("What is Rust?", &options).await?;
//!
//!     while let Some(event) = stream.next().await {
//!         if let Some(ContentBlock::Text(text)) = event?.into_block() {
//!             print!("{}", text.text);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Multi-Turn Conversation
//!
//! ```rust,no_run
//! use open_agent::{Client, AgentOptions, ContentBlock};
//! use futures::StreamExt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut client = Client::new(AgentOptions::builder()
//!     .model("gpt-4")
//!     .api_key("sk-...")
//!     .build()?)?;
//!
//! // First question
//! client.send("What's the capital of France?").await?;
//! while let Some(block) = client.receive().await? {
//!     if let ContentBlock::Text(text) = block {
//!         println!("{}", text.text);
//!     }
//! }
//!
//! // Follow-up question (history is maintained)
//! client.send("What's its population?").await?;
//! while let Some(block) = client.receive().await? {
//!     if let ContentBlock::Text(text) = block {
//!         println!("{}", text.text);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Manual Tool Execution
//!
//! ```rust,no_run
//! use open_agent::{Client, AgentOptions, ContentBlock, Tool};
//! use serde_json::json;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let calculator = Tool::new(
//!     "calculator",
//!     "Performs arithmetic operations",
//!     json!({"type": "object", "properties": {"operation": {"type": "string"}}}),
//!     |input| Box::pin(async move {
//!         // Custom execution logic
//!         Ok(json!({"result": 42}))
//!     })
//! );
//!
//! let mut client = Client::new(AgentOptions::builder()
//!     .model("gpt-4")
//!     .api_key("sk-...")
//!     .tools(vec![calculator])
//!     .build()?)?;
//!
//! client.send("Calculate 2+2").await?;
//!
//! while let Some(block) = client.receive().await? {
//!     match block {
//!         ContentBlock::ToolUse(tool_use) => {
//!             println!("Model wants to use: {}", tool_use.name());
//!
//!             // Execute tool manually
//!             let tool = client.get_tool(tool_use.name()).unwrap();
//!             let result = tool.execute(tool_use.input().clone()).await?;
//!
//!             // Add result and continue
//!             client.add_tool_result(tool_use.id(), result)?;
//!             client.send("").await?;
//!         }
//!         ContentBlock::Text(text) => {
//!             println!("Response: {}", text.text);
//!         }
//!         ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {}
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Automatic Tool Execution
//!
//! ```rust,no_run
//! use open_agent::{Client, AgentOptions, ContentBlock, Tool};
//! use serde_json::json;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let calculator = Tool::new(
//!     "calculator",
//!     "Performs arithmetic operations",
//!     json!({"type": "object"}),
//!     |input| Box::pin(async move { Ok(json!({"result": 42})) })
//! );
//!
//! let mut client = Client::new(AgentOptions::builder()
//!     .model("gpt-4")
//!     .api_key("sk-...")
//!     .tools(vec![calculator])
//!     .auto_execute_tools(true)  // Enable auto-execution
//!     .max_tool_iterations(5)    // Max 5 tool rounds
//!     .build()?)?;
//!
//! client.send("Calculate 2+2 and then multiply by 3").await?;
//!
//! // Tools are executed automatically - you only get final text response
//! while let Some(block) = client.receive().await? {
//!     if let ContentBlock::Text(text) = block {
//!         println!("{}", text.text);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## With Hooks
//!
//! ```ignore
//! use open_agent::{Client, AgentOptions, Hooks, HookDecision};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let hooks = Hooks::new()
//!     .add_user_prompt_submit(|event| async move {
//!         // Block prompts containing certain words
//!         if event.prompt.contains("forbidden") {
//!             return Some(HookDecision::block("Forbidden word detected"));
//!         }
//!         Some(HookDecision::continue_())
//!     })
//!     .add_pre_tool_use(|event| async move {
//!         // Log all tool uses
//!         println!("Executing tool: {}", event.tool_name);
//!         Some(HookDecision::continue_())
//!     });
//!
//! let mut client = Client::new(AgentOptions::builder()
//!     .model("gpt-4")
//!     .base_url("http://localhost:1234/v1")
//!     .hooks(hooks)
//!     .build()?)?;
//!
//! // Hooks will be executed automatically
//! client.send("Hello!").await?;
//! # Ok(())
//! # }
//! ```

use crate::types::{
    AgentOptions, ContentBlock, FinishReason, Message, MessageRole, OpenAIContent,
    OpenAIContentPart, OpenAIFunction, OpenAIMessage, OpenAIRequest, OpenAIToolCall, StreamEvent,
    TextBlock,
};
use crate::utils::{StreamAccumulator, parse_sse_stream};
use crate::{Error, Result};
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

fn serialize_history_message(message: &Message) -> Result<serde_json::Value> {
    serde_json::to_value(message).map_err(|error| {
        Error::other(format!(
            "Failed to serialize conversation history for hook: {error}"
        ))
    })
}

fn serialize_history_snapshot(history: &[Message]) -> Result<Vec<serde_json::Value>> {
    history.iter().map(serialize_history_message).collect()
}

async fn stream_request(
    http_client: &reqwest::Client,
    options: &AgentOptions,
    request: &OpenAIRequest,
) -> Result<EventStream> {
    let url = format!("{}/chat/completions", options.base_url());
    let response = http_client
        .post(&url)
        .header("Authorization", format!("Bearer {}", options.api_key()))
        .header("Content-Type", "application/json")
        .json(request)
        .send()
        .await
        .map_err(Error::Http)?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_else(|error| {
            eprintln!("WARNING: Failed to read error response body: {error}");
            "Unknown error (failed to read response body)".to_string()
        });
        return Err(Error::api_status(status, body));
    }

    // Append a `None` sentinel after the chunk stream so the accumulator gets an explicit
    // end-of-transport signal. Servers that stop sending without ever setting `finish_reason`
    // (llama.cpp, vLLM, several local gateways) would otherwise leave their content stranded
    // in the accumulator's buffers and yield a silently empty response. The sentinel is also
    // what emits the terminating `StreamEvent::Finish`, so every stream reports how it ended.
    let terminated = parse_sse_stream(response)
        .map(Some)
        .chain(futures::stream::iter([None]));

    let accumulator = StreamAccumulator::new().capture_reasoning(options.include_reasoning());

    // `scan` yields one batch per input item; batches are then flattened into individual
    // events. An empty batch simply flattens to nothing, so chunks that only accumulate
    // state need no special-casing here.
    let flattened = terminated
        .scan(accumulator, |accumulator, item| {
            let batch = match item {
                Some(Ok(chunk)) => accumulator.process_chunk(chunk),
                Some(Err(error)) => Err(error),
                None => accumulator.finalize(),
            };
            futures::future::ready(Some(batch))
        })
        .flat_map(|result| {
            futures::stream::iter(match result {
                Ok(events) => events.into_iter().map(Ok).collect(),
                Err(error) => vec![Err(error)],
            })
        });

    Ok(Box::pin(flattened))
}

include!("client/query.rs");
include!("client/state.rs");
include!("client/setup.rs");
include!("client/send.rs");
include!("client/streaming.rs");
include!("client/send_message.rs");
include!("client/receive.rs");
include!("client/history.rs");

#[cfg(test)]
mod tests {
    use super::*;

    include!("client/tests.rs");
}
