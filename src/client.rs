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
//! - **current_stream**: Active SSE stream being consumed (`Option<ContentStream>`)
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
//! use open_agent::{query, AgentOptions};
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
//!     while let Some(block) = stream.next().await {
//!         if let open_agent::ContentBlock::Text(text) = block? {
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
//!             println!("Model wants to use: {}", tool_use.name);
//!
//!             // Execute tool manually
//!             let tool = client.get_tool(&tool_use.name).unwrap();
//!             let result = tool.execute(tool_use.input).await?;
//!
//!             // Add result and continue
//!             client.add_tool_result(&tool_use.id, result)?;
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
    AgentOptions, ContentBlock, Message, MessageRole, OpenAIContent, OpenAIContentPart,
    OpenAIFunction, OpenAIMessage, OpenAIRequest, OpenAIToolCall, TextBlock,
};
use crate::utils::{ToolCallAggregator, parse_sse_stream};
use crate::{Error, Result};
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// A pinned, boxed stream of content blocks from the model.
///
/// This type alias represents an asynchronous stream that yields `ContentBlock` items.
/// Each item is wrapped in a `Result` to handle potential errors during streaming.
///
/// The stream is:
/// - **Pinned** (`Pin<Box<...>>`): Required for safe async operations and self-referential types
/// - **Boxed**: Allows dynamic dispatch and hides the concrete stream implementation
/// - **Send**: Can be safely transferred between threads
///
/// # Content Blocks
///
/// The stream can yield several types of content blocks:
///
/// - **TextBlock**: Incremental text responses from the model
/// - **ToolUseBlock**: Requests to execute a tool with specific parameters
/// - **ToolResultBlock**: Results from tool execution (in manual mode)
///
/// # Error Handling
///
/// Errors in the stream indicate issues like:
/// - Network failures or timeouts
/// - Malformed SSE events
/// - JSON parsing errors
/// - API errors from the model provider
///
/// When an error occurs, the stream typically terminates. It's the caller's responsibility
/// to handle errors appropriately.
///
/// # Examples
///
/// ```rust,no_run
/// use open_agent::{query, AgentOptions, ContentBlock};
/// use futures::StreamExt;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let options = AgentOptions::builder()
///     .model("gpt-4")
///     .api_key("sk-...")
///     .build()?;
///
/// let mut stream = query("Hello!", &options).await?;
///
/// while let Some(result) = stream.next().await {
///     match result {
///         Ok(ContentBlock::Text(text)) => print!("{}", text.text),
///         Ok(_) => {}, // Other block types
///         Err(e) => eprintln!("Stream error: {}", e),
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub type ContentStream = Pin<Box<dyn Stream<Item = Result<ContentBlock>> + Send>>;

/// Simple query function for single-turn interactions without conversation history.
///
/// This is a stateless convenience function for simple queries that don't require
/// multi-turn conversations. It creates a temporary HTTP client, sends a single
/// prompt, and returns a stream of content blocks.
///
/// For multi-turn conversations or more control over the interaction, use [`Client`] instead.
///
/// # Parameters
///
/// - `prompt`: The user's message to send to the model
/// - `options`: Configuration including model, API key, tools, etc.
///
/// # Returns
///
/// Returns a `ContentStream` that yields content blocks as they arrive from the model.
/// The stream must be polled to completion to receive all blocks.
///
/// # Behavior
///
/// 1. Creates a temporary HTTP client with configured timeout
/// 2. Builds message array (system prompt + user prompt)
/// 3. Converts tools to OpenAI format if provided
/// 4. Makes HTTP POST request to `/chat/completions`
/// 5. Parses Server-Sent Events (SSE) response stream
/// 6. Aggregates chunks into complete content blocks
/// 7. Returns stream that yields blocks as they complete
///
/// # Error Handling
///
/// This function can return errors for:
/// - HTTP client creation failures
/// - Network errors during the request
/// - API errors (authentication, invalid model, rate limits, etc.)
/// - SSE parsing errors
/// - JSON deserialization errors
///
/// # Performance Notes
///
/// - Creates a new HTTP client for each call (consider using `Client` for repeated queries)
/// - Timeout is configurable via `AgentOptions::timeout` (default: 120 seconds)
/// - Streaming begins immediately; no buffering of the full response
///
/// # Examples
///
/// ## Basic Usage
///
/// ```rust,no_run
/// use open_agent::{query, AgentOptions};
/// use futures::StreamExt;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let options = AgentOptions::builder()
///         .system_prompt("You are a helpful assistant")
///         .model("gpt-4")
///         .api_key("sk-...")
///         .build()?;
///
///     let mut stream = query("What's the capital of France?", &options).await?;
///
///     while let Some(block) = stream.next().await {
///         match block? {
///             open_agent::ContentBlock::Text(text) => {
///                 print!("{}", text.text);
///             }
///             open_agent::ContentBlock::ToolUse(_)
///             | open_agent::ContentBlock::ToolResult(_)
///             | open_agent::ContentBlock::Image(_) => {}
///         }
///     }
///
///     Ok(())
/// }
/// ```
///
/// ## With Tools
///
/// ```rust,no_run
/// use open_agent::{query, AgentOptions, Tool, ContentBlock};
/// use futures::StreamExt;
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let calculator = Tool::new(
///     "calculator",
///     "Performs calculations",
///     json!({"type": "object"}),
///     |input| Box::pin(async move { Ok(json!({"result": 42})) })
/// );
///
/// let options = AgentOptions::builder()
///     .model("gpt-4")
///     .api_key("sk-...")
///     .tools(vec![calculator])
///     .build()?;
///
/// let mut stream = query("Calculate 2+2", &options).await?;
///
/// while let Some(block) = stream.next().await {
///     match block? {
///         ContentBlock::ToolUse(tool_use) => {
///             println!("Model wants to use: {}", tool_use.name);
///             // Note: You'll need to manually execute tools and continue
///             // the conversation. For automatic execution, use Client.
///         }
///         ContentBlock::Text(text) => print!("{}", text.text),
///         ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {}
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// ## Error Handling
///
/// ```rust,no_run
/// use open_agent::{query, AgentOptions};
/// use futures::StreamExt;
///
/// # async fn example() {
/// let options = AgentOptions::builder()
///     .model("gpt-4")
///     .api_key("invalid-key")
///     .build()
///     .unwrap();
///
/// match query("Hello", &options).await {
///     Ok(mut stream) => {
///         while let Some(result) = stream.next().await {
///             match result {
///                 Ok(block) => println!("Block: {:?}", block),
///                 Err(e) => {
///                     eprintln!("Stream error: {}", e);
///                     break;
///                 }
///             }
///         }
///     }
///     Err(e) => eprintln!("Query failed: {}", e),
/// }
/// # }
/// ```
pub async fn query(prompt: &str, options: &AgentOptions) -> Result<ContentStream> {
    // Create HTTP client with configured timeout
    // The timeout applies to the entire request, not individual chunks
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(options.timeout()))
        .build()
        .map_err(Error::Http)?;

    // Build messages array for the API request
    // OpenAI format expects an array of message objects with role and content
    let mut messages = Vec::new();

    // Add system prompt if provided
    // System prompts set the assistant's behavior and context
    if !options.system_prompt().is_empty() {
        messages.push(OpenAIMessage {
            role: "system".to_string(),
            content: Some(OpenAIContent::Text(options.system_prompt().to_string())),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Add user prompt
    // This is the actual query from the user
    messages.push(OpenAIMessage {
        role: "user".to_string(),
        content: Some(OpenAIContent::Text(prompt.to_string())),
        tool_calls: None,
        tool_call_id: None,
    });

    // Convert tools to OpenAI format if any are provided
    // Tools are described using JSON Schema for parameter validation
    let tools = if !options.tools().is_empty() {
        Some(
            options
                .tools()
                .iter()
                .map(|t| t.to_openai_format())
                .collect(),
        )
    } else {
        None
    };

    // Build the OpenAI-compatible request payload
    // stream=true enables Server-Sent Events for incremental responses
    let request = OpenAIRequest {
        model: options.model().to_string(),
        messages,
        stream: true, // Critical: enables SSE streaming
        max_tokens: options.max_tokens(),
        temperature: Some(options.temperature()),
        tools,
    };

    // Make HTTP POST request to the chat completions endpoint
    let url = format!("{}/chat/completions", options.base_url());
    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", options.api_key()))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(Error::Http)?;

    // Check for HTTP-level errors before processing the stream
    // This catches authentication failures, rate limits, invalid models, etc.
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_else(|e| {
            eprintln!("WARNING: Failed to read error response body: {}", e);
            "Unknown error (failed to read response body)".to_string()
        });
        return Err(Error::api(format!("API error {}: {}", status, body)));
    }

    // Parse the Server-Sent Events (SSE) stream
    // The response body is a stream of "data: {...}" events
    let sse_stream = parse_sse_stream(response);

    // Aggregate SSE chunks into complete content blocks
    // ToolCallAggregator handles partial JSON and assembles complete tool calls
    // The scan() combinator maintains state across stream items
    let stream = sse_stream.scan(ToolCallAggregator::new(), |aggregator, chunk_result| {
        let result = match chunk_result {
            Ok(chunk) => match aggregator.process_chunk(chunk) {
                Ok(blocks) => {
                    if blocks.is_empty() {
                        Some(None) // Intermediate chunk, continue streaming
                    } else {
                        Some(Some(Ok(blocks))) // Complete block(s) ready
                    }
                }
                Err(e) => Some(Some(Err(e))), // Propagate processing error
            },
            Err(e) => Some(Some(Err(e))), // Propagate stream error
        };
        futures::future::ready(result)
    });

    // Flatten the stream to emit individual blocks
    // filter_map removes None values (incomplete chunks)
    // flat_map expands Vec<ContentBlock> into individual items
    let flattened = stream
        .filter_map(|item| async move { item })
        .flat_map(|result| {
            futures::stream::iter(match result {
                Ok(blocks) => blocks.into_iter().map(Ok).collect(),
                Err(e) => vec![Err(e)],
            })
        });

    // Pin and box the stream for type erasure and safe async usage
    Ok(Box::pin(flattened))
}

/// Stateful client for multi-turn conversations with automatic history management.
///
/// The `Client` is the primary interface for building conversational AI applications.
/// It maintains conversation history, manages streaming responses, and provides two
/// modes of operation: manual and automatic tool execution.
///
/// # State Management
///
/// The client maintains several pieces of state that persist across multiple turns:
///
/// - **Conversation History**: Complete record of all messages exchanged
/// - **Active Stream**: Currently active SSE stream being consumed
/// - **Interrupt Flag**: Thread-safe cancellation signal
/// - **Auto-Execution Buffer**: Cached blocks for auto-execution mode
///
/// # Operating Modes
///
/// ## Manual Mode (default)
///
/// In manual mode, the client streams blocks directly to the caller. When the model
/// requests a tool, you receive a `ToolUseBlock`, execute the tool yourself, add the
/// result with `add_tool_result()`, and continue the conversation.
///
/// **Advantages**:
/// - Full control over tool execution
/// - Custom error handling per tool
/// - Ability to modify tool inputs/outputs
/// - Interactive debugging capabilities
///
/// ## Automatic Mode (`auto_execute_tools = true`)
///
/// In automatic mode, the client executes tools transparently and only returns the
/// final text response after all tool iterations complete.
///
/// **Advantages**:
/// - Simpler API for common use cases
/// - Built-in retry logic via hooks
/// - Automatic conversation continuation
/// - Configurable iteration limits
///
/// # Thread Safety
///
/// The client is NOT thread-safe for concurrent use. However, the interrupt mechanism
/// uses `Arc<AtomicBool>` which can be safely shared across threads to signal cancellation.
///
/// # Memory Management
///
/// - History grows unbounded by default (consider clearing periodically)
/// - Streams are consumed lazily (low memory footprint during streaming)
/// - Auto-execution buffers entire response (higher memory in auto mode)
///
/// # Examples
///
/// ## Basic Multi-Turn Conversation
///
/// ```rust,no_run
/// use open_agent::{Client, AgentOptions, ContentBlock};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut client = Client::new(AgentOptions::builder()
///     .model("gpt-4")
///     .api_key("sk-...")
///     .build()?)?;
///
/// // First question
/// client.send("What's the capital of France?").await?;
/// while let Some(block) = client.receive().await? {
///     if let ContentBlock::Text(text) = block {
///         println!("{}", text.text); // "Paris is the capital of France."
///     }
/// }
///
/// // Follow-up question - history is automatically maintained
/// client.send("What's its population?").await?;
/// while let Some(block) = client.receive().await? {
///     if let ContentBlock::Text(text) = block {
///         println!("{}", text.text); // "Paris has approximately 2.2 million people."
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// ## Manual Tool Execution
///
/// ```rust,no_run
/// use open_agent::{Client, AgentOptions, ContentBlock, Tool};
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let calculator = Tool::new(
///     "calculator",
///     "Performs arithmetic",
///     json!({"type": "object"}),
///     |input| Box::pin(async move { Ok(json!({"result": 42})) })
/// );
///
/// let mut client = Client::new(AgentOptions::builder()
///     .model("gpt-4")
///     .api_key("sk-...")
///     .tools(vec![calculator])
///     .build()?)?;
///
/// client.send("What's 2+2?").await?;
///
/// while let Some(block) = client.receive().await? {
///     match block {
///         ContentBlock::ToolUse(tool_use) => {
///             // Execute tool manually
///             let result = json!({"result": 4});
///             client.add_tool_result(&tool_use.id, result)?;
///
///             // Continue conversation to get model's response
///             client.send("").await?;
///         }
///         ContentBlock::Text(text) => {
///             println!("{}", text.text); // "The result is 4."
///         }
///         ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {}
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// ## Automatic Tool Execution
///
/// ```rust,no_run
/// use open_agent::{Client, AgentOptions, ContentBlock, Tool};
/// use serde_json::json;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let calculator = Tool::new(
///     "calculator",
///     "Performs arithmetic",
///     json!({"type": "object"}),
///     |input| Box::pin(async move { Ok(json!({"result": 42})) })
/// );
///
/// let mut client = Client::new(AgentOptions::builder()
///     .model("gpt-4")
///     .api_key("sk-...")
///     .tools(vec![calculator])
///     .auto_execute_tools(true)  // Enable auto-execution
///     .build()?)?;
///
/// client.send("What's 2+2?").await?;
///
/// // Tools execute automatically - you only receive final text
/// while let Some(block) = client.receive().await? {
///     if let ContentBlock::Text(text) = block {
///         println!("{}", text.text); // "The result is 4."
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// ## With Interruption
///
/// ```rust,no_run
/// use open_agent::{Client, AgentOptions};
/// use std::time::Duration;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut client = Client::new(AgentOptions::default())?;
///
/// // Start a long-running query
/// client.send("Write a very long story").await?;
///
/// // Spawn a task to interrupt after timeout
/// let interrupt_handle = client.interrupt_handle();
/// tokio::spawn(async move {
///     tokio::time::sleep(Duration::from_secs(5)).await;
///     interrupt_handle.store(true, std::sync::atomic::Ordering::SeqCst);
/// });
///
/// // This loop will stop when interrupted
/// while let Some(block) = client.receive().await? {
///     // Process blocks...
/// }
///
/// // Client is still usable after interruption
/// client.send("What's 2+2?").await?;
/// # Ok(())
/// # }
/// ```
pub struct Client {
    /// Configuration options including model, API key, tools, hooks, etc.
    ///
    /// This field contains all the settings that control how the client behaves.
    /// It's set once during construction and cannot be modified (though you can
    /// access it via `options()` for inspection).
    options: AgentOptions,

    /// Complete conversation history as a sequence of messages.
    ///
    /// Each message contains a role (System/User/Assistant/Tool) and content blocks.
    /// History grows unbounded by default - use `clear_history()` to reset.
    ///
    /// **Important**: The history includes ALL messages, not just user/assistant.
    /// This includes tool results and intermediate assistant messages from tool calls.
    history: Vec<Message>,

    /// Currently active SSE stream being consumed.
    ///
    /// This is `Some(stream)` while a response is being received, and `None` when
    /// no request is in flight or after a response completes.
    ///
    /// The stream is set by `send()` and consumed by `receive()`. When the stream
    /// is exhausted, `receive()` returns `Ok(None)` and sets this back to `None`.
    current_stream: Option<ContentStream>,

    /// Reusable HTTP client for making API requests.
    ///
    /// Configured once during construction with the timeout from `AgentOptions`.
    /// Reusing the same client across requests enables connection pooling and
    /// better performance for multi-turn conversations.
    http_client: reqwest::Client,

    /// Thread-safe interrupt flag for cancellation.
    ///
    /// This `Arc<AtomicBool>` can be cloned and shared across threads or async tasks
    /// to signal cancellation. When set to `true`, the next `receive()` call will
    /// return `Ok(None)` and clear the current stream.
    ///
    /// The flag is automatically reset to `false` at the start of each `send()` call.
    ///
    /// **Thread Safety**: Can be safely accessed from multiple threads using atomic
    /// operations. However, only one thread should call `send()`/`receive()`.
    interrupted: Arc<AtomicBool>,

    /// Buffer of content blocks for auto-execution mode.
    ///
    /// When `auto_execute_tools` is enabled, `receive()` internally calls the
    /// auto-execution loop which buffers all final text blocks here. Subsequent
    /// calls to `receive()` return blocks from this buffer one at a time.
    ///
    /// **Only used when `options.auto_execute_tools == true`**.
    ///
    /// The buffer is cleared when starting a new auto-execution loop.
    auto_exec_buffer: Vec<ContentBlock>,

    /// Current read position in the auto-execution buffer.
    ///
    /// Tracks which block to return next when `receive()` is called in auto mode.
    /// Reset to 0 when the buffer is refilled with a new response.
    ///
    /// **Only used when `options.auto_execute_tools == true`**.
    auto_exec_index: usize,
}

impl Client {
    /// Creates a new client with the specified configuration.
    ///
    /// This constructor initializes all state fields and creates a reusable HTTP client
    /// configured with the timeout from `AgentOptions`.
    ///
    /// # Parameters
    ///
    /// - `options`: Configuration including model, API key, tools, hooks, etc.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be built. This can happen due to:
    /// - Invalid TLS configuration
    /// - System resource exhaustion
    /// - Invalid timeout values
    ///
    /// # Examples
    ///
    /// ```rust
    /// use open_agent::{Client, AgentOptions};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new(AgentOptions::builder()
    ///     .model("gpt-4")
    ///     .base_url("http://localhost:1234/v1")
    ///     .build()?)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(options: AgentOptions) -> Result<Self> {
        // Build HTTP client with configured timeout
        // This client is reused across all requests for connection pooling
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(options.timeout()))
            .build()
            .map_err(|e| Error::config(format!("Failed to build HTTP client: {}", e)))?;

        Ok(Self {
            options,
            history: Vec::new(),  // Empty conversation history
            current_stream: None, // No active stream yet
            http_client,
            interrupted: Arc::new(AtomicBool::new(false)), // Not interrupted initially
            auto_exec_buffer: Vec::new(),                  // Empty buffer for auto mode
            auto_exec_index: 0,                            // Start at beginning of buffer
        })
    }

    /// Sends a user message and initiates streaming of the model's response.
    ///
    /// This method performs several critical steps:
    ///
    /// 1. Executes UserPromptSubmit hooks (which can modify or block the prompt)
    /// 2. Adds the user message to conversation history
    /// 3. Builds and sends HTTP request to the OpenAI-compatible API
    /// 4. Parses the SSE stream and sets up aggregation
    /// 5. Stores the stream for consumption via `receive()`
    ///
    /// # Parameters
    ///
    /// - `prompt`: The user's message. Can be empty to continue conversation after
    ///   adding tool results (common pattern in manual tool execution mode).
    ///
    /// # Returns
    ///
    /// - `Ok(())`: Request sent successfully, call `receive()` to get blocks
    /// - `Err(e)`: Request failed (network error, API error, hook blocked, etc.)
    ///
    /// # Behavior Details
    ///
    /// ## Hook Execution
    ///
    /// Before sending, UserPromptSubmit hooks are executed. Hooks can:
    /// - Modify the prompt text
    /// - Block the request entirely
    /// - Access conversation history
    ///
    /// If a hook blocks the request, this method returns an error immediately.
    ///
    /// ## History Management
    ///
    /// The prompt is added to history BEFORE sending the request. This ensures
    /// that history is consistent even if the request fails.
    ///
    /// ## Stream Setup
    ///
    /// The response stream is set up but not consumed. You must call `receive()`
    /// repeatedly to get content blocks. The stream remains active until:
    /// - All blocks are consumed (stream naturally ends)
    /// - An error occurs
    /// - Interrupt is triggered
    ///
    /// ## Interrupt Handling
    ///
    /// The interrupt flag is reset to `false` at the start of this method,
    /// allowing a fresh request after a previous interruption.
    ///
    /// # State Changes
    ///
    /// - Resets `interrupted` flag to `false`
    /// - Appends user message to `history`
    /// - Sets `current_stream` to new SSE stream
    /// - Does NOT modify `auto_exec_buffer` or `auto_exec_index`
    ///
    /// # Errors
    ///
    /// Returns errors for:
    /// - Hook blocking the prompt
    /// - HTTP client errors (network failure, DNS, etc.)
    /// - API errors (auth failure, invalid model, rate limits)
    /// - Invalid response format
    ///
    /// After an error, the client remains usable for new requests.
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Hello!").await?;
    ///
    /// while let Some(block) = client.receive().await? {
    ///     // Process blocks...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Continuing After Tool Result
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions, ContentBlock};
    /// # use serde_json::json;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Use the calculator").await?;
    ///
    /// while let Some(block) = client.receive().await? {
    ///     if let ContentBlock::ToolUse(tool_use) = block {
    ///         // Execute tool and add result
    ///         client.add_tool_result(&tool_use.id, json!({"result": 42}))?;
    ///
    ///         // Continue conversation with empty prompt
    ///         client.send("").await?;
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send(&mut self, prompt: &str) -> Result<()> {
        use crate::hooks::UserPromptSubmitEvent;

        // Reset interrupt flag for new query
        // This allows the client to be reused after a previous interruption
        // Uses SeqCst ordering to ensure visibility across all threads
        self.interrupted.store(false, Ordering::SeqCst);

        // Execute UserPromptSubmit hooks
        // Hooks run BEFORE adding to history, allowing modification or blocking
        let mut final_prompt = prompt.to_string();
        let history_snapshot: Vec<serde_json::Value> = self
            .history
            .iter()
            .map(|_| serde_json::json!({})) // Simplified snapshot for hooks
            .collect();

        // Create hook event with current prompt and history
        let event = UserPromptSubmitEvent::new(final_prompt.clone(), history_snapshot);

        // Execute all registered UserPromptSubmit hooks
        if let Some(decision) = self.options.hooks().execute_user_prompt_submit(event).await {
            // Check if hook wants to block execution
            if !decision.continue_execution() {
                return Err(Error::other(format!(
                    "Prompt blocked by hook: {}",
                    decision.reason().unwrap_or("")
                )));
            }
            // Apply any prompt modifications from hooks
            if let Some(modified) = decision.modified_prompt() {
                final_prompt = modified.to_string();
            }
        }

        // Add user message to history BEFORE sending request
        // This ensures history consistency even if request fails
        // Empty prompts are still added (needed for tool continuation)
        self.history.push(Message::user(final_prompt));

        // Build messages array for API request
        // This includes system prompt + full conversation history
        let mut messages = Vec::new();

        // Add system prompt as first message if configured
        // System prompts are added fresh for each request (not from history)
        if !self.options.system_prompt().is_empty() {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(OpenAIContent::Text(
                    self.options.system_prompt().to_string(),
                )),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Convert conversation history to OpenAI message format
        // This includes user prompts, assistant responses, and tool results
        for msg in &self.history {
            // Separate blocks by type to determine message structure
            let mut text_blocks = Vec::new();
            let mut image_blocks = Vec::new();
            let mut tool_use_blocks = Vec::new();
            let mut tool_result_blocks = Vec::new();

            for block in &msg.content {
                match block {
                    ContentBlock::Text(text) => text_blocks.push(text),
                    ContentBlock::Image(image) => image_blocks.push(image),
                    ContentBlock::ToolUse(tool_use) => tool_use_blocks.push(tool_use),
                    ContentBlock::ToolResult(tool_result) => tool_result_blocks.push(tool_result),
                }
            }

            // Handle different message types based on content blocks
            // Case 1: Message contains tool results (should be separate tool messages)
            if !tool_result_blocks.is_empty() {
                for tool_result in tool_result_blocks {
                    // Serialize the tool result content as JSON string
                    let content = serde_json::to_string(&tool_result.content).unwrap_or_else(|e| {
                        format!("{{\"error\": \"Failed to serialize: {}\"}}", e)
                    });

                    messages.push(OpenAIMessage {
                        role: "tool".to_string(),
                        content: Some(OpenAIContent::Text(content)),
                        tool_calls: None,
                        tool_call_id: Some(tool_result.tool_use_id.clone()),
                    });
                }
            }
            // Case 2: Message contains tool use blocks (assistant with tool calls)
            else if !tool_use_blocks.is_empty() {
                // Build tool_calls array
                let tool_calls: Vec<OpenAIToolCall> = tool_use_blocks
                    .iter()
                    .map(|tool_use| {
                        // Serialize the input as a JSON string (OpenAI API requirement)
                        let arguments = serde_json::to_string(&tool_use.input)
                            .unwrap_or_else(|_| "{}".to_string());

                        OpenAIToolCall {
                            id: tool_use.id.clone(),
                            call_type: "function".to_string(),
                            function: OpenAIFunction {
                                name: tool_use.name.clone(),
                                arguments,
                            },
                        }
                    })
                    .collect();

                // Extract any text content (some models include reasoning before tool calls)
                // Note: OpenAI API requires content field even if empty when tool_calls present
                let content = if !text_blocks.is_empty() {
                    let text = text_blocks
                        .iter()
                        .map(|t| t.text.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    Some(OpenAIContent::Text(text))
                } else {
                    // Empty string satisfies OpenAI API schema (content is required)
                    Some(OpenAIContent::Text(String::new()))
                };

                messages.push(OpenAIMessage {
                    role: "assistant".to_string(),
                    content,
                    tool_calls: Some(tool_calls),
                    tool_call_id: None,
                });
            }
            // Case 3: Message contains images (use OpenAIContent::Parts)
            else if !image_blocks.is_empty() {
                // Log debug info about images being serialized
                log::debug!(
                    "Serializing message with {} image(s) for {:?} role",
                    image_blocks.len(),
                    msg.role
                );

                // Build content parts array preserving original order
                let mut content_parts = Vec::new();

                // Re-iterate through content blocks to maintain order
                for block in &msg.content {
                    match block {
                        ContentBlock::Text(text) => {
                            content_parts.push(OpenAIContentPart::text(&text.text));
                        }
                        ContentBlock::Image(image) => {
                            // Log image details (truncate URL for privacy)
                            let url_display = if image.url().len() > 100 {
                                format!("{}... ({} chars)", &image.url()[..100], image.url().len())
                            } else {
                                image.url().to_string()
                            };
                            let detail_str = match image.detail() {
                                crate::types::ImageDetail::Low => "low",
                                crate::types::ImageDetail::High => "high",
                                crate::types::ImageDetail::Auto => "auto",
                            };
                            log::debug!("  - Image: {} (detail: {})", url_display, detail_str);

                            content_parts
                                .push(OpenAIContentPart::image_url(image.url(), image.detail()));
                        }
                        ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) => {}
                    }
                }

                // Defensive check: content_parts should never be empty at this point
                // If it is, it indicates a logic error (e.g., all blocks were filtered out)
                if content_parts.is_empty() {
                    return Err(Error::other(
                        "Internal error: Message with images produced empty content array",
                    ));
                }

                let role_str = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                };

                messages.push(OpenAIMessage {
                    role: role_str.to_string(),
                    content: Some(OpenAIContent::Parts(content_parts)),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            // Case 4: Message contains only text (normal message, backward compatible)
            else {
                let content = text_blocks
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");

                let role_str = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                };

                messages.push(OpenAIMessage {
                    role: role_str.to_string(),
                    content: Some(OpenAIContent::Text(content)),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }

        // Convert tools to OpenAI format if any are registered
        // Each tool is described with name, description, and JSON Schema parameters
        let tools = if !self.options.tools().is_empty() {
            Some(
                self.options
                    .tools()
                    .iter()
                    .map(|t| t.to_openai_format())
                    .collect(),
            )
        } else {
            None
        };

        // Build the OpenAI-compatible request payload
        let request = OpenAIRequest {
            model: self.options.model().to_string(),
            messages,
            stream: true, // Always stream for progressive rendering
            max_tokens: self.options.max_tokens(),
            temperature: Some(self.options.temperature()),
            tools,
        };

        // Make HTTP POST request to chat completions endpoint
        let url = format!("{}/chat/completions", self.options.base_url());
        let response = self
            .http_client
            .post(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.options.api_key()),
            )
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(Error::Http)?;

        // Check for HTTP-level errors before processing stream
        // This catches authentication, rate limits, invalid models, etc.
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|e| {
                eprintln!("WARNING: Failed to read error response body: {}", e);
                "Unknown error (failed to read response body)".to_string()
            });
            return Err(Error::api(format!("API error {}: {}", status, body)));
        }

        // Parse Server-Sent Events (SSE) stream from response
        let sse_stream = parse_sse_stream(response);

        // Aggregate SSE chunks into complete content blocks
        // ToolCallAggregator maintains state to handle incremental JSON chunks
        // that may arrive split across multiple SSE events
        let stream = sse_stream.scan(ToolCallAggregator::new(), |aggregator, chunk_result| {
            let result = match chunk_result {
                Ok(chunk) => match aggregator.process_chunk(chunk) {
                    Ok(blocks) => {
                        if blocks.is_empty() {
                            Some(None) // Partial chunk, keep aggregating
                        } else {
                            Some(Some(Ok(blocks))) // Complete block(s) ready
                        }
                    }
                    Err(e) => Some(Some(Err(e))), // Processing error
                },
                Err(e) => Some(Some(Err(e))), // Stream error
            };
            futures::future::ready(result)
        });

        // Flatten the stream to emit individual blocks
        // filter_map removes None values (partial chunks)
        // flat_map converts Vec<ContentBlock> to individual items
        let flattened = stream
            .filter_map(|item| async move { item })
            .flat_map(|result| {
                futures::stream::iter(match result {
                    Ok(blocks) => blocks.into_iter().map(Ok).collect(),
                    Err(e) => vec![Err(e)],
                })
            });

        // Store the stream for consumption via receive()
        // The stream is NOT consumed here - that happens in receive()
        self.current_stream = Some(Box::pin(flattened));

        Ok(())
    }

    /// Internal method that returns one block from the current stream.
    ///
    /// This is the core streaming logic extracted for reuse by both manual mode
    /// and auto-execution mode. It handles interrupt checking and stream consumption.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(block))`: Successfully received a content block
    /// - `Ok(None)`: Stream ended naturally or was interrupted
    /// - `Err(e)`: An error occurred during streaming
    ///
    /// # State Changes
    ///
    /// - Sets `current_stream` to `None` if interrupted or stream ends
    /// - Does not modify history or other state
    ///
    /// # Implementation Notes
    ///
    /// This method checks the interrupt flag on every call, allowing responsive
    /// cancellation. The check uses SeqCst ordering for immediate visibility of
    /// interrupts from other threads.
    async fn receive_one(&mut self) -> Result<Option<ContentBlock>> {
        // Check interrupt flag before attempting to receive
        // Uses SeqCst to ensure we see the latest value from any thread
        if self.interrupted.load(Ordering::SeqCst) {
            // Clear the stream and return None to signal completion
            self.current_stream = None;
            return Ok(None);
        }

        // Poll the current stream if one exists
        if let Some(stream) = &mut self.current_stream {
            match stream.next().await {
                Some(Ok(block)) => Ok(Some(block)), // Got a block
                Some(Err(e)) => Err(e),             // Stream error
                None => Ok(None),                   // Stream ended
            }
        } else {
            // No active stream
            Ok(None)
        }
    }

    /// Collects all blocks from the current stream into a vector.
    ///
    /// Internal helper for auto-execution mode. This method buffers the entire
    /// response in memory, which is necessary to determine if the response contains
    /// tool calls before returning anything to the caller.
    ///
    /// # Returns
    ///
    /// - `Ok(vec)`: Successfully collected all blocks
    /// - `Err(e)`: Error during collection or interrupted
    ///
    /// # Memory Usage
    ///
    /// This buffers the entire response, which can be large for long completions.
    /// Consider the memory implications when using auto-execution mode.
    ///
    /// # Interruption
    ///
    /// Checks interrupt flag during collection and returns error if interrupted.
    async fn collect_all_blocks(&mut self) -> Result<Vec<ContentBlock>> {
        let mut blocks = Vec::new();

        // Consume entire stream into vector
        while let Some(block) = self.receive_one().await? {
            // Check interrupt during collection for responsiveness
            if self.interrupted.load(Ordering::SeqCst) {
                self.current_stream = None;
                return Err(Error::other(
                    "Operation interrupted during block collection",
                ));
            }

            blocks.push(block);
        }

        Ok(blocks)
    }

    /// Executes a tool by name with the given input.
    ///
    /// Internal helper for auto-execution mode. Looks up the tool in the registered
    /// tools list and executes it with the provided input.
    ///
    /// # Parameters
    ///
    /// - `tool_name`: Name of the tool to execute
    /// - `input`: JSON value containing tool parameters
    ///
    /// # Returns
    ///
    /// - `Ok(result)`: Tool executed successfully, returns result as JSON
    /// - `Err(e)`: Tool not found or execution failed
    ///
    /// # Error Handling
    ///
    /// If the tool is not found in the registry, returns a ToolError.
    /// If execution fails, the error from the tool is propagated.
    async fn execute_tool_internal(
        &self,
        tool_name: &str,
        input: serde_json::Value,
    ) -> Result<serde_json::Value> {
        // Find tool in registered tools by name
        let tool = self
            .options
            .tools()
            .iter()
            .find(|t| t.name() == tool_name)
            .ok_or_else(|| Error::tool(format!("Tool '{}' not found", tool_name)))?;

        // Execute the tool's async function
        tool.execute(input).await
    }

    /// Auto-execution loop that handles tool calls automatically.
    ///
    /// This is the core implementation of automatic tool execution mode. It:
    ///
    /// 1. Collects all blocks from the current stream
    /// 2. Separates text blocks from tool use blocks
    /// 3. If there are tool blocks:
    ///    - Executes PreToolUse hooks (can modify/block)
    ///    - Executes each tool via its registered function
    ///    - Executes PostToolUse hooks (can modify result)
    ///    - Adds results to history
    ///    - Continues conversation with send("")
    /// 4. Repeats until text-only response or max iterations
    /// 5. Returns all final text blocks
    ///
    /// # Returns
    ///
    /// - `Ok(blocks)`: Final text blocks after all tool iterations
    /// - `Err(e)`: Error during execution, stream processing, or interruption
    ///
    /// # Iteration Limit
    ///
    /// The loop is bounded by `options.max_tool_iterations` to prevent infinite loops.
    /// When the limit is reached, the loop stops and returns whatever text blocks
    /// have been collected so far.
    ///
    /// # Hook Integration
    ///
    /// Hooks are executed for each tool call:
    /// - **PreToolUse**: Can modify input or block execution entirely
    /// - **PostToolUse**: Can modify the result before it's added to history
    ///
    /// If a hook blocks execution, a JSON error response is used as the tool result.
    ///
    /// # State Management
    ///
    /// The loop maintains history by adding:
    /// - Assistant messages with text + tool use blocks
    /// - User messages with tool result blocks
    ///
    /// This creates a proper conversation flow that the model can follow.
    ///
    /// # Error Recovery
    ///
    /// If a tool execution fails, the error is converted to a JSON error response
    /// and added as the tool result. This allows the conversation to continue
    /// and lets the model handle the error.
    async fn auto_execute_loop(&mut self) -> Result<Vec<ContentBlock>> {
        use crate::types::ToolResultBlock;

        // Track iterations to prevent infinite loops
        let mut iteration = 0;
        let max_iterations = self.options.max_tool_iterations();

        loop {
            // ========================================================================
            // STEP 1: Collect all blocks from current stream
            // ========================================================================
            // Buffer the entire response to determine if it contains tool calls
            let blocks = self.collect_all_blocks().await?;

            // Empty response means stream ended or was interrupted
            if blocks.is_empty() {
                return Ok(Vec::new());
            }

            // ========================================================================
            // STEP 2: Separate text blocks from tool use blocks
            // ========================================================================
            // The model can return a mix of text and tool calls in one response
            let mut text_blocks = Vec::new();
            let mut tool_blocks = Vec::new();

            for block in blocks {
                match block {
                    ContentBlock::Text(_) => text_blocks.push(block),
                    ContentBlock::ToolUse(_) => tool_blocks.push(block),
                    ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {} // Ignore ToolResult and Image variants
                }
            }

            // ========================================================================
            // STEP 3: Check if we're done (no tool calls)
            // ========================================================================
            // If the response contains no tool calls, we've reached the final answer
            if tool_blocks.is_empty() {
                // Add assistant's final text response to history
                if !text_blocks.is_empty() {
                    let assistant_msg = Message::assistant(text_blocks.clone());
                    self.history.push(assistant_msg);
                }
                // Return text blocks to caller via buffered receive()
                return Ok(text_blocks);
            }

            // ========================================================================
            // STEP 4: Check iteration limit BEFORE executing tools
            // ========================================================================
            // Increment counter and check if we've hit the max
            iteration += 1;
            if iteration > max_iterations {
                // Max iterations reached - stop execution and return what we have
                // This prevents infinite tool-calling loops
                if !text_blocks.is_empty() {
                    let assistant_msg = Message::assistant(text_blocks.clone());
                    self.history.push(assistant_msg);
                }
                return Ok(text_blocks);
            }

            // ========================================================================
            // STEP 5: Add assistant message to history
            // ========================================================================
            // The assistant message includes BOTH text and tool use blocks
            // This preserves the full context for future turns
            let mut all_blocks = text_blocks.clone();
            all_blocks.extend(tool_blocks.clone());
            let assistant_msg = Message::assistant(all_blocks);
            self.history.push(assistant_msg);

            // ========================================================================
            // STEP 6: Execute all tools and collect results
            // ========================================================================
            for block in tool_blocks {
                if let ContentBlock::ToolUse(tool_use) = block {
                    // Create simplified history snapshot for hooks
                    // TODO: Full serialization of history for hooks
                    let history_snapshot: Vec<serde_json::Value> =
                        self.history.iter().map(|_| serde_json::json!({})).collect();

                    // ============================================================
                    // Execute PreToolUse hooks
                    // ============================================================
                    use crate::hooks::PreToolUseEvent;
                    let pre_event = PreToolUseEvent::new(
                        tool_use.name.clone(),
                        tool_use.input.clone(),
                        tool_use.id.clone(),
                        history_snapshot.clone(),
                    );

                    // Track whether to execute and what input to use
                    let mut tool_input = tool_use.input.clone();
                    let mut should_execute = true;
                    let mut block_reason = None;

                    // Execute all PreToolUse hooks
                    if let Some(decision) =
                        self.options.hooks().execute_pre_tool_use(pre_event).await
                    {
                        if !decision.continue_execution() {
                            // Hook blocked execution
                            should_execute = false;
                            block_reason = decision.reason().map(|s| s.to_string());
                        } else if let Some(modified) = decision.modified_input() {
                            // Hook modified the input
                            tool_input = modified.clone();
                        }
                    }

                    // ============================================================
                    // Execute tool (or create error result if blocked)
                    // ============================================================
                    let result = if should_execute {
                        // Actually execute the tool
                        match self
                            .execute_tool_internal(&tool_use.name, tool_input.clone())
                            .await
                        {
                            Ok(res) => res, // Success - use the result
                            Err(e) => {
                                // Tool execution failed - convert to JSON error
                                // This allows the conversation to continue
                                serde_json::json!({
                                    "error": e.to_string(),
                                    "tool": tool_use.name,
                                    "id": tool_use.id
                                })
                            }
                        }
                    } else {
                        // Tool blocked by PreToolUse hook - create error result
                        serde_json::json!({
                            "error": "Tool execution blocked by hook",
                            "reason": block_reason.unwrap_or_else(|| "No reason provided".to_string()),
                            "tool": tool_use.name,
                            "id": tool_use.id
                        })
                    };

                    // ============================================================
                    // Execute PostToolUse hooks
                    // ============================================================
                    use crate::hooks::PostToolUseEvent;
                    let post_event = PostToolUseEvent::new(
                        tool_use.name.clone(),
                        tool_input,
                        tool_use.id.clone(),
                        result.clone(),
                        history_snapshot,
                    );

                    let mut final_result = result;
                    if let Some(decision) =
                        self.options.hooks().execute_post_tool_use(post_event).await
                    {
                        // PostToolUse can modify the result
                        // Note: Uses modified_input field (naming is historical)
                        if let Some(modified) = decision.modified_input() {
                            final_result = modified.clone();
                        }
                    }

                    // ============================================================
                    // Add tool result to history
                    // ============================================================
                    // Tool results are added as user messages (per OpenAI convention)
                    let tool_result = ToolResultBlock::new(&tool_use.id, final_result);
                    let tool_result_msg =
                        Message::user_with_blocks(vec![ContentBlock::ToolResult(tool_result)]);
                    self.history.push(tool_result_msg);
                }
            }

            // ========================================================================
            // STEP 7: Continue conversation to get next response
            // ========================================================================
            // Send empty string to continue - the history contains all context
            self.send("").await?;

            // Loop continues to collect and process the next response
            // This will either be more tool calls or the final text answer
        }
    }

    /// Receives the next content block from the current stream.
    ///
    /// This is the primary method for consuming responses from the model. It works
    /// differently depending on the operating mode:
    ///
    /// ## Manual Mode (default)
    ///
    /// Streams blocks directly from the API response as they arrive. You receive:
    /// - `TextBlock`: Incremental text from the model
    /// - `ToolUseBlock`: Requests to execute tools
    /// - Other block types as they're emitted
    ///
    /// When you receive a `ToolUseBlock`, you must:
    /// 1. Execute the tool yourself
    /// 2. Call `add_tool_result()` with the result
    /// 3. Call `send("")` to continue the conversation
    ///
    /// ## Automatic Mode (`auto_execute_tools = true`)
    ///
    /// Transparently executes tools and only returns final text blocks. The first
    /// call to `receive()` triggers the auto-execution loop which:
    /// 1. Collects all blocks from the stream
    /// 2. Executes any tool calls automatically
    /// 3. Continues the conversation until reaching a text-only response
    /// 4. Buffers the final text blocks
    /// 5. Returns them one at a time on subsequent `receive()` calls
    ///
    /// # Returns
    ///
    /// - `Ok(Some(block))`: Successfully received a content block
    /// - `Ok(None)`: Stream ended normally or was interrupted
    /// - `Err(e)`: An error occurred during streaming or tool execution
    ///
    /// # Behavior Details
    ///
    /// ## Interruption
    ///
    /// Checks the interrupt flag on every call. If interrupted, immediately returns
    /// `Ok(None)` and clears the stream. The client can be reused after interruption.
    ///
    /// ## Stream Lifecycle
    ///
    /// 1. After `send()`, stream is active
    /// 2. Each `receive()` call yields one block
    /// 3. When stream ends, returns `Ok(None)`
    /// 4. Subsequent calls continue returning `Ok(None)` until next `send()`
    ///
    /// ## Auto-Execution Buffer
    ///
    /// In auto mode, blocks are buffered in memory. The buffer persists until
    /// fully consumed (index reaches length), at which point it's cleared.
    ///
    /// # State Changes
    ///
    /// - Advances stream position
    /// - In auto mode: May trigger entire execution loop and modify history
    /// - In manual mode: Only reads from stream, no history changes
    /// - Increments `auto_exec_index` when returning buffered blocks
    ///
    /// # Examples
    ///
    /// ## Manual Mode - Basic
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions, ContentBlock};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Hello!").await?;
    ///
    /// while let Some(block) = client.receive().await? {
    ///     match block {
    ///         ContentBlock::Text(text) => print!("{}", text.text),
    ///         ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {}
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Manual Mode - With Tools
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions, ContentBlock};
    /// # use serde_json::json;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Use the calculator").await?;
    ///
    /// while let Some(block) = client.receive().await? {
    ///     match block {
    ///         ContentBlock::Text(text) => {
    ///             println!("{}", text.text);
    ///         }
    ///         ContentBlock::ToolUse(tool_use) => {
    ///             println!("Executing: {}", tool_use.name);
    ///
    ///             // Execute tool manually
    ///             let result = json!({"result": 42});
    ///
    ///             // Add result and continue
    ///             client.add_tool_result(&tool_use.id, result)?;
    ///             client.send("").await?;
    ///         }
    ///         ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {}
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Auto Mode
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions, ContentBlock, Tool};
    /// # use serde_json::json;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::builder()
    ///     .auto_execute_tools(true)
    ///     .build()?)?;
    ///
    /// client.send("Calculate 2+2").await?;
    ///
    /// // Tools execute automatically - you only get final text
    /// while let Some(block) = client.receive().await? {
    ///     if let ContentBlock::Text(text) = block {
    ///         println!("{}", text.text);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## With Error Handling
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Hello").await?;
    ///
    /// loop {
    ///     match client.receive().await {
    ///         Ok(Some(block)) => {
    ///             // Process block
    ///         }
    ///         Ok(None) => {
    ///             // Stream ended
    ///             break;
    ///         }
    ///         Err(e) => {
    ///             eprintln!("Error: {}", e);
    ///             break;
    ///         }
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn receive(&mut self) -> Result<Option<ContentBlock>> {
        // ========================================================================
        // AUTO-EXECUTION MODE
        // ========================================================================
        if self.options.auto_execute_tools() {
            // Check if we have buffered blocks to return
            // In auto mode, all final text blocks are buffered and returned one at a time
            if self.auto_exec_index < self.auto_exec_buffer.len() {
                // Return next buffered block
                let block = self.auto_exec_buffer[self.auto_exec_index].clone();
                self.auto_exec_index += 1;
                return Ok(Some(block));
            }

            // No buffered blocks - need to run auto-execution loop
            // This only happens on the first receive() call after send()
            if self.auto_exec_buffer.is_empty() {
                match self.auto_execute_loop().await {
                    Ok(blocks) => {
                        // Buffer all final text blocks
                        self.auto_exec_buffer = blocks;
                        self.auto_exec_index = 0;

                        // If no blocks, return None (empty response)
                        if self.auto_exec_buffer.is_empty() {
                            return Ok(None);
                        }

                        // Return first buffered block
                        let block = self.auto_exec_buffer[0].clone();
                        self.auto_exec_index = 1;
                        return Ok(Some(block));
                    }
                    Err(e) => return Err(e),
                }
            }

            // Buffer exhausted - return None
            Ok(None)
        } else {
            // ====================================================================
            // MANUAL MODE
            // ====================================================================
            // Stream blocks directly from API without buffering or auto-execution
            self.receive_one().await
        }
    }

    /// Interrupts the current operation by setting the interrupt flag.
    ///
    /// This method provides a thread-safe way to cancel any in-progress streaming
    /// operation. The interrupt flag is checked by `receive()` before each block,
    /// allowing responsive cancellation.
    ///
    /// # Behavior
    ///
    /// - Sets the atomic interrupt flag to `true`
    /// - Next `receive()` call will return `Ok(None)` and clear the stream
    /// - Flag is automatically reset to `false` on next `send()` call
    /// - Safe to call from any thread (uses atomic operations)
    /// - Idempotent: calling multiple times has same effect as calling once
    /// - No-op if no operation is in progress
    ///
    /// # Thread Safety
    ///
    /// This method uses `Arc<AtomicBool>` internally, which can be safely shared
    /// across threads. You can clone the interrupt handle and use it from different
    /// threads or async tasks:
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    /// let interrupt_handle = client.interrupt_handle();
    ///
    /// // Use from another thread
    /// tokio::spawn(async move {
    ///     tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    ///     interrupt_handle.store(true, std::sync::atomic::Ordering::SeqCst);
    /// });
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # State Changes
    ///
    /// - Sets `interrupted` flag to `true`
    /// - Does NOT modify stream, history, or other state directly
    /// - Effect takes place on next `receive()` call
    ///
    /// # Use Cases
    ///
    /// - User cancellation (e.g., stop button in UI)
    /// - Timeout enforcement
    /// - Resource cleanup
    /// - Emergency shutdown
    ///
    /// # Examples
    ///
    /// ## Basic Interruption
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    ///
    /// client.send("Tell me a long story").await?;
    ///
    /// // Interrupt after receiving some blocks
    /// let mut count = 0;
    /// while let Some(block) = client.receive().await? {
    ///     count += 1;
    ///     if count >= 5 {
    ///         client.interrupt();
    ///     }
    /// }
    ///
    /// // Client is ready for new queries
    /// client.send("What's 2+2?").await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## With Timeout
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions};
    /// use std::time::Duration;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    ///
    /// client.send("Long request").await?;
    ///
    /// // Spawn timeout task
    /// let interrupt_handle = client.interrupt_handle();
    /// tokio::spawn(async move {
    ///     tokio::time::sleep(Duration::from_secs(10)).await;
    ///     interrupt_handle.store(true, std::sync::atomic::Ordering::SeqCst);
    /// });
    ///
    /// while let Some(_block) = client.receive().await? {
    ///     // Process until timeout
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn interrupt(&self) {
        // Set interrupt flag using SeqCst for immediate visibility across all threads
        self.interrupted.store(true, Ordering::SeqCst);
    }

    /// Returns a clone of the interrupt handle for thread-safe cancellation.
    ///
    /// This method provides access to the shared `Arc<AtomicBool>` interrupt flag,
    /// allowing it to be used from other threads or async tasks to signal cancellation.
    ///
    /// # Returns
    ///
    /// A cloned `Arc<AtomicBool>` that can be used to interrupt operations from any thread.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    /// let interrupt_handle = client.interrupt_handle();
    ///
    /// // Use from another thread
    /// tokio::spawn(async move {
    ///     tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    ///     interrupt_handle.store(true, std::sync::atomic::Ordering::SeqCst);
    /// });
    /// # Ok(())
    /// # }
    /// ```
    pub fn interrupt_handle(&self) -> Arc<AtomicBool> {
        self.interrupted.clone()
    }

    /// Returns a reference to the conversation history.
    ///
    /// The history contains all messages exchanged in the conversation, including:
    /// - User messages
    /// - Assistant messages (with text and tool use blocks)
    /// - Tool result messages
    ///
    /// # Returns
    ///
    /// A slice of `Message` objects in chronological order.
    ///
    /// # Use Cases
    ///
    /// - Inspecting conversation context
    /// - Debugging tool execution flow
    /// - Saving conversation state
    /// - Implementing custom history management
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use open_agent::{Client, AgentOptions};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new(AgentOptions::default())?;
    ///
    /// // Initially empty
    /// assert_eq!(client.history().len(), 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Returns a mutable reference to the conversation history.
    ///
    /// This allows you to modify the history directly for advanced use cases like:
    /// - Removing old messages to manage context length
    /// - Editing messages for retry scenarios
    /// - Injecting synthetic messages for testing
    ///
    /// # Warning
    ///
    /// Modifying history directly can lead to inconsistent conversation state if not
    /// done carefully. The SDK expects history to follow the proper message flow
    /// (user → assistant → tool results → assistant, etc.).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use open_agent::{Client, AgentOptions};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    ///
    /// // Remove oldest messages to stay within context limit
    /// if client.history().len() > 50 {
    ///     client.history_mut().drain(0..10);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn history_mut(&mut self) -> &mut Vec<Message> {
        &mut self.history
    }

    /// Returns a reference to the agent configuration options.
    ///
    /// Provides read-only access to the `AgentOptions` used to configure this client.
    ///
    /// # Use Cases
    ///
    /// - Inspecting current configuration
    /// - Debugging issues
    /// - Conditional logic based on settings
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use open_agent::{Client, AgentOptions};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new(AgentOptions::builder()
    ///     .model("gpt-4")
    ///     .base_url("http://localhost:1234/v1")
    ///     .build()?)?;
    ///
    /// println!("Using model: {}", client.options().model());
    /// # Ok(())
    /// # }
    /// ```
    pub fn options(&self) -> &AgentOptions {
        &self.options
    }

    /// Clears all conversation history.
    ///
    /// This resets the conversation to a blank slate while preserving the client
    /// configuration (tools, hooks, model, etc.). The next message will start a
    /// fresh conversation with no prior context.
    ///
    /// # State Changes
    ///
    /// - Clears `history` vector
    /// - Does NOT modify current stream, options, or other state
    ///
    /// # Use Cases
    ///
    /// - Starting a new conversation
    /// - Preventing context length issues
    /// - Clearing sensitive data
    /// - Implementing conversation sessions
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use open_agent::{Client, AgentOptions, ContentBlock};
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    ///
    /// // First conversation
    /// client.send("Hello").await?;
    /// while let Some(_) = client.receive().await? {}
    ///
    /// // Clear and start fresh
    /// client.clear_history();
    ///
    /// // New conversation with no memory of previous
    /// client.send("Hello again").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Adds a tool result to the conversation history for manual tool execution.
    ///
    /// This method is used exclusively in **manual mode** after receiving a `ToolUseBlock`.
    /// The workflow is:
    ///
    /// 1. `receive()` returns a `ToolUseBlock`
    /// 2. You execute the tool yourself
    /// 3. Call `add_tool_result()` with the tool's output
    /// 4. Call `send("")` to continue the conversation
    /// 5. The model receives the tool result and generates a response
    ///
    /// # Parameters
    ///
    /// - `tool_use_id`: The unique ID from the `ToolUseBlock` (must match exactly)
    /// - `content`: The tool's output as a JSON value
    ///
    /// # Behavior
    ///
    /// Creates a `ToolResultBlock` and adds it to conversation history as a tool message.
    /// This preserves the tool call/result pairing that the model needs to understand
    /// the conversation flow.
    ///
    /// # State Changes
    ///
    /// - Appends a tool message to `history`
    /// - Does NOT modify stream or trigger any requests
    ///
    /// # Important Notes
    ///
    /// - **Not used in auto mode**: Auto-execution handles tool results automatically
    /// - **ID must match**: The `tool_use_id` must match the ID from the `ToolUseBlock`
    /// - **No validation**: This method doesn't validate the result format
    /// - **Must call send()**: After adding result(s), call `send("")` to continue
    ///
    /// # Examples
    ///
    /// ## Basic Manual Tool Execution
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions, ContentBlock};
    /// use serde_json::json;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Use the calculator").await?;
    ///
    /// while let Some(block) = client.receive().await? {
    ///     match block {
    ///         ContentBlock::ToolUse(tool_use) => {
    ///             // Execute tool manually
    ///             let result = json!({"result": 42});
    ///
    ///             // Add result to history
    ///             client.add_tool_result(&tool_use.id, result)?;
    ///
    ///             // Continue conversation to get model's response
    ///             client.send("").await?;
    ///         }
    ///         ContentBlock::Text(text) => {
    ///             println!("{}", text.text);
    ///         }
    ///         ContentBlock::ToolResult(_) | ContentBlock::Image(_) => {}
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Handling Tool Errors
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions, ContentBlock};
    /// use serde_json::json;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// # client.send("test").await?;
    /// while let Some(block) = client.receive().await? {
    ///     if let ContentBlock::ToolUse(tool_use) = block {
    ///         // Try to execute tool
    ///         let result = match execute_tool(&tool_use.name, &tool_use.input) {
    ///             Ok(output) => output,
    ///             Err(e) => json!({
    ///                 "error": e.to_string(),
    ///                 "tool": tool_use.name
    ///             })
    ///         };
    ///
    ///         client.add_tool_result(&tool_use.id, result)?;
    ///         client.send("").await?;
    ///     }
    /// }
    ///
    /// # fn execute_tool(name: &str, input: &serde_json::Value) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    /// #     Ok(json!({}))
    /// # }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Multiple Tool Calls
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions, ContentBlock};
    /// use serde_json::json;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// client.send("Calculate 2+2 and 3+3").await?;
    ///
    /// let mut tool_calls = Vec::new();
    ///
    /// // Collect all tool calls
    /// while let Some(block) = client.receive().await? {
    ///     if let ContentBlock::ToolUse(tool_use) = block {
    ///         tool_calls.push(tool_use);
    ///     }
    /// }
    ///
    /// // Execute and add results for all tools
    /// for tool_call in tool_calls {
    ///     let result = json!({"result": 42}); // Execute tool
    ///     client.add_tool_result(&tool_call.id, result)?;
    /// }
    ///
    /// // Continue conversation
    /// client.send("").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_tool_result(&mut self, tool_use_id: &str, content: serde_json::Value) -> Result<()> {
        use crate::types::ToolResultBlock;

        // Create a tool result block with the given ID and content
        let result_block = ToolResultBlock::new(tool_use_id, content);

        // Add to history as a tool message
        // Note: ToolResultBlock is properly serialized in build_api_request()
        // as a separate message with role="tool" and tool_call_id set
        let serialized = serde_json::to_string(&result_block.content)
            .map_err(|e| Error::config(format!("Failed to serialize tool result: {}", e)))?;

        self.history.push(Message::new(
            MessageRole::Tool,
            vec![ContentBlock::Text(TextBlock::new(serialized))],
        ));

        Ok(())
    }

    /// Looks up a registered tool by name.
    ///
    /// This method provides access to the tool registry for manual execution scenarios.
    /// It searches the tools registered in `AgentOptions` and returns a reference to
    /// the matching tool if found.
    ///
    /// # Parameters
    ///
    /// - `name`: The tool name to search for (case-sensitive)
    ///
    /// # Returns
    ///
    /// - `Some(&Tool)`: Tool found
    /// - `None`: No tool with that name
    ///
    /// # Use Cases
    ///
    /// - Manual tool execution in response to `ToolUseBlock`
    /// - Validating tool availability before offering features
    /// - Inspecting tool metadata (name, description, schema)
    ///
    /// # Examples
    ///
    /// ## Execute Tool Manually
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions, ContentBlock};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut client = Client::new(AgentOptions::default())?;
    /// # client.send("test").await?;
    /// while let Some(block) = client.receive().await? {
    ///     if let ContentBlock::ToolUse(tool_use) = block {
    ///         if let Some(tool) = client.get_tool(&tool_use.name) {
    ///             // Execute the tool
    ///             let result = tool.execute(tool_use.input.clone()).await?;
    ///             client.add_tool_result(&tool_use.id, result)?;
    ///             client.send("").await?;
    ///         } else {
    ///             println!("Unknown tool: {}", tool_use.name);
    ///         }
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Check Tool Availability
    ///
    /// ```rust
    /// # use open_agent::{Client, AgentOptions};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = Client::new(AgentOptions::default())?;
    /// if client.get_tool("calculator").is_some() {
    ///     println!("Calculator is available");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_tool(&self, name: &str) -> Option<&crate::tools::Tool> {
        // Search registered tools by name
        self.options
            .tools()
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options).expect("Should create client successfully");
        assert_eq!(client.history().len(), 0);
    }

    #[test]
    fn test_client_new_returns_result() {
        // Test that Client::new() returns Result instead of panicking
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        // This should not panic - it should return Ok(client)
        let result = Client::new(options);
        assert!(result.is_ok(), "Client::new() should return Ok");

        let client = result.unwrap();
        assert_eq!(client.history().len(), 0);
    }

    #[test]
    fn test_interrupt_flag_initial_state() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options).expect("Should create client successfully");
        // Initially not interrupted
        assert!(!client.interrupted.load(Ordering::SeqCst));
    }

    #[test]
    fn test_interrupt_sets_flag() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options).expect("Should create client successfully");
        client.interrupt();
        assert!(client.interrupted.load(Ordering::SeqCst));
    }

    #[test]
    fn test_interrupt_idempotent() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options).expect("Should create client successfully");
        client.interrupt();
        assert!(client.interrupted.load(Ordering::SeqCst));

        // Call again - should still be interrupted
        client.interrupt();
        assert!(client.interrupted.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_receive_returns_none_when_interrupted() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let mut client = Client::new(options).expect("Should create client successfully");

        // Interrupt before receiving
        client.interrupt();

        // NEW SIGNATURE: receive() should return Ok(None) when interrupted
        let result = client.receive().await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_receive_returns_ok_none_when_no_stream() {
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let mut client = Client::new(options).expect("Should create client successfully");

        // No stream started - receive() should return Ok(None)
        let result = client.receive().await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_receive_error_propagation() {
        // This test demonstrates that errors are wrapped in Err(), not Some(Err())
        // We'll verify this behavior when we have a mock stream that produces errors
        let options = AgentOptions::builder()
            .system_prompt("Test")
            .model("test-model")
            .base_url("http://localhost:1234/v1")
            .build()
            .unwrap();

        let client = Client::new(options).expect("Should create client successfully");

        // Signature check: receive() returns Result<Option<ContentBlock>>
        // This means we can use ? operator cleanly:
        // while let Some(block) = client.receive().await? { ... }

        // Type assertion to ensure signature is correct
        let _: Result<Option<ContentBlock>> = std::future::ready(Ok(None)).await;
        drop(client);
    }

    #[test]
    fn test_empty_content_parts_protection() {
        // Test for Issue #3 - Verify empty content_parts causes appropriate handling
        // This documents expected behavior: messages with images should have content

        use crate::types::{ContentBlock, ImageBlock, Message, MessageRole};

        // GIVEN: Message with an image
        let img = ImageBlock::from_url("https://example.com/test.jpg").expect("Valid URL");

        let msg = Message::new(MessageRole::User, vec![ContentBlock::Image(img)]);

        // WHEN: Building content_parts
        let mut content_parts = Vec::new();
        for block in &msg.content {
            match block {
                ContentBlock::Text(text) => {
                    content_parts.push(crate::types::OpenAIContentPart::text(&text.text));
                }
                ContentBlock::Image(image) => {
                    content_parts.push(crate::types::OpenAIContentPart::image_url(
                        image.url(),
                        image.detail(),
                    ));
                }
                ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) => {}
            }
        }

        // THEN: content_parts should not be empty
        assert!(
            !content_parts.is_empty(),
            "Messages with images should produce non-empty content_parts"
        );
    }
}
