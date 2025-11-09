//! Streaming utilities for SSE parsing and tool call aggregation.
//!
//! This module provides the core infrastructure for handling streaming responses from the
//! OpenAI-compatible API. It addresses two key challenges:
//!
//! # Challenge 1: SSE (Server-Sent Events) Parsing
//!
//! The API returns data in SSE format, which looks like:
//!
//! ```text
//! data: {"id":"msg_123","object":"chat.completion.chunk","choices":[...]}
//!
//! data: {"id":"msg_123","object":"chat.completion.chunk","choices":[...]}
//!
//! data: [DONE]
//! ```
//!
//! Each line starting with `data: ` contains a JSON chunk. The stream ends with `data: [DONE]`.
//! The [`parse_sse_stream`] function transforms the raw HTTP response into a stream of parsed
//! [`OpenAIChunk`] objects.
//!
//! # Challenge 2: Tool Call Delta Aggregation
//!
//! Tool calls arrive as **incremental deltas** across multiple chunks, not as complete objects.
//! For example, a single tool call might arrive as:
//!
//! ```text
//! Chunk 1: { tool_calls: [{ index: 0, id: "call_abc123", function: { name: "get_weather" } }] }
//! Chunk 2: { tool_calls: [{ index: 0, function: { arguments: "{\"loc" } }] }
//! Chunk 3: { tool_calls: [{ index: 0, function: { arguments: "ation" } }] }
//! Chunk 4: { tool_calls: [{ index: 0, function: { arguments: "\":\"Paris\"}" } }] }
//! Chunk 5: { finish_reason: "tool_calls" }
//! ```
//!
//! The [`ToolCallAggregator`] accumulates these deltas and only emits complete [`ContentBlock`]s
//! when a `finish_reason` is encountered (indicating the end of generation).
//!
//! # Complete Flow Example
//!
//! Here's how raw SSE data flows through this module to produce final content blocks:
//!
//! ```text
//! Raw HTTP Stream (SSE format)
//!     │
//!     │ parse_sse_stream()
//!     ▼
//! Stream<OpenAIChunk>
//!     │
//!     │ ToolCallAggregator::process_chunk()
//!     ▼
//! Vec<ContentBlock>  (only when finish_reason is present)
//! ```
//!
//! ## Example: Text Response
//!
//! ```text
//! Input chunks:
//!   1. content: "Hello"
//!   2. content: " world", finish_reason: "stop"
//!
//! Output: [ContentBlock::Text(TextBlock { text: "Hello world" })]
//! ```
//!
//! ## Example: Tool Call Response
//!
//! ```text
//! Input chunks:
//!   1. tool_calls: [{ index: 0, id: "call_123", function: { name: "search" } }]
//!   2. tool_calls: [{ index: 0, function: { arguments: "{\"q" } }]
//!   3. tool_calls: [{ index: 0, function: { arguments: "\":\"rust\"}" } }]
//!   4. finish_reason: "tool_calls"
//!
//! Output: [ContentBlock::ToolUse(ToolUseBlock {
//!     id: "call_123",
//!     name: "search",
//!     input: {"q": "rust"}
//! })]
//! ```
//!
//! # Why This Matters
//!
//! Without proper aggregation, consumers would receive incomplete tool calls with partial JSON
//! strings, making the API unusable. This module ensures that all tool calls are fully assembled
//! and validated before being exposed to the application.

use crate::types::{ContentBlock, OpenAIChunk, TextBlock, ToolUseBlock};
use crate::{Error, Result};
use futures::stream::{Stream, StreamExt};
use std::collections::HashMap;
use std::pin::Pin;

/// Aggregates streaming deltas into complete content blocks.
///
/// This is a **stateful accumulator** that processes [`OpenAIChunk`] objects one at a time,
/// building up complete text and tool call content over multiple chunks. It only returns
/// complete [`ContentBlock`]s when a `finish_reason` is encountered.
///
/// # State Management
///
/// The aggregator maintains two pieces of state:
///
/// 1. **Text Buffer** (`text_buffer`): Accumulates text content across chunks. Text deltas
///    are concatenated as they arrive. When generation finishes, the complete text is
///    emitted as a [`ContentBlock::Text`].
///
/// 2. **Tool Call Map** (`tool_calls`): A HashMap indexed by tool call index (provided by
///    the API) that tracks partially-received tool calls. Each entry accumulates the tool's
///    ID, name, and JSON argument string. When generation finishes, all tool calls are
///    emitted as [`ContentBlock::ToolUse`] blocks.
///
/// # Why Index-Based Storage?
///
/// The API can return multiple tool calls in a single response, and they arrive interleaved:
///
/// ```text
/// Chunk 1: tool_calls[0] = { id: "call_1", name: "search" }
/// Chunk 2: tool_calls[1] = { id: "call_2", name: "calculate" }
/// Chunk 3: tool_calls[0] = { arguments: "{\"q\"" }
/// Chunk 4: tool_calls[1] = { arguments: "{\"expr\"" }
/// Chunk 5: tool_calls[0] = { arguments: ":\"rust\"}" }
/// Chunk 6: tool_calls[1] = { arguments: ":\"2+2\"}" }
/// ```
///
/// The HashMap keyed by index allows us to correctly accumulate each tool call independently.
///
/// # Usage Pattern
///
/// ```rust,ignore
/// let mut aggregator = ToolCallAggregator::new();
///
/// for chunk in stream {
///     let blocks = aggregator.process_chunk(chunk)?;
///     // blocks is empty until finish_reason is encountered
///     if !blocks.is_empty() {
///         // Generation complete, blocks contains all content
///         handle_completed_blocks(blocks);
///     }
/// }
/// ```
///
/// # Important Invariants
///
/// - **Buffers are cleared after finish**: Once a `finish_reason` is seen, both the text
///   buffer and tool call map are cleared, readying the aggregator for the next turn.
///
/// - **Partial JSON accumulation**: Tool call arguments are accumulated as raw strings and
///   only parsed as JSON when the tool call is complete. This allows JSON to be split at
///   arbitrary boundaries across chunks.
///
/// - **Empty responses**: If generation finishes with no content (empty text buffer and no
///   tool calls), an empty `Vec<ContentBlock>` is returned.
pub struct ToolCallAggregator {
    /// Buffer for accumulating text content deltas across chunks.
    /// Cleared when a finish_reason is encountered.
    text_buffer: String,

    /// Map of partially-received tool calls, indexed by their API-provided index.
    /// Each entry accumulates ID, name, and argument deltas.
    /// Cleared when a finish_reason is encountered.
    tool_calls: HashMap<u32, PartialToolCall>,
}

/// Represents an in-progress tool call that is being assembled from deltas.
///
/// Tool calls arrive fragmented across multiple chunks. This struct accumulates the pieces
/// until we have a complete tool call ready to be converted into a [`ToolUseBlock`].
///
/// # Field Evolution
///
/// As chunks arrive, fields are populated incrementally:
///
/// ```text
/// Initial state:     { id: None, name: None, arguments: "" }
/// After chunk 1:     { id: Some("call_123"), name: Some("search"), arguments: "" }
/// After chunk 2:     { id: Some("call_123"), name: Some("search"), arguments: "{\"q" }
/// After chunk 3:     { id: Some("call_123"), name: Some("search"), arguments: "{\"q\":\"rust\"}" }
/// ```
///
/// # Fields
///
/// - **`id`**: The unique identifier for this tool call (e.g., "call_abc123"). Usually arrives
///   in the first chunk containing this tool call. Required to create a valid [`ToolUseBlock`].
///
/// - **`name`**: The function/tool name (e.g., "get_weather", "search"). Usually arrives in
///   the first chunk. Required to create a valid [`ToolUseBlock`].
///
/// - **`arguments`**: The JSON string of tool arguments, assembled piece by piece. Starts empty
///   and grows as argument deltas arrive. May be split at arbitrary positions (even mid-string
///   or mid-number). Only parsed as JSON when the tool call is complete.
///
/// # Completion Criteria
///
/// A `PartialToolCall` is considered **complete** when:
/// 1. A `finish_reason` is encountered in the stream
/// 2. Both `id` and `name` are `Some(_)`
/// 3. The `arguments` string is valid JSON (validated during parsing)
///
/// Incomplete tool calls (missing ID or name) are silently dropped during aggregation.
#[derive(Debug, Default)]
struct PartialToolCall {
    /// Unique identifier for the tool call. Usually arrives in the first chunk.
    id: Option<String>,

    /// Name of the tool/function to call. Usually arrives in the first chunk.
    name: Option<String>,

    /// Accumulated JSON argument string. Built up incrementally across chunks.
    /// May be split at arbitrary byte positions across chunks.
    arguments: String,
}

impl ToolCallAggregator {
    /// Creates a new aggregator with empty buffers.
    ///
    /// The aggregator starts with no accumulated state and is ready to process chunks.
    pub fn new() -> Self {
        Self {
            text_buffer: String::new(),
            tool_calls: HashMap::new(),
        }
    }

    /// Processes a single chunk and returns completed content blocks.
    ///
    /// This is the core method of the aggregator. It accumulates deltas from the chunk into
    /// internal buffers and returns completed [`ContentBlock`]s **only when** a `finish_reason`
    /// is present in the chunk.
    ///
    /// # Arguments
    ///
    /// * `chunk` - A single [`OpenAIChunk`] from the streaming response
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<ContentBlock>)` - Empty vector if generation is ongoing, or a vector of
    ///   completed blocks when `finish_reason` is encountered
    /// * `Err(Error)` - If tool call argument JSON is invalid
    ///
    /// # Behavior
    ///
    /// 1. **Accumulation Phase**: Adds any text or tool call deltas to internal buffers
    /// 2. **Completion Phase**: When `finish_reason` is present, flushes buffers and returns
    ///    all completed content blocks
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut aggregator = ToolCallAggregator::new();
    ///
    /// // Process first chunk (partial text)
    /// let chunk1 = /* chunk with content: "Hello" */;
    /// let blocks = aggregator.process_chunk(chunk1)?;
    /// assert!(blocks.is_empty()); // No finish_reason yet
    ///
    /// // Process final chunk
    /// let chunk2 = /* chunk with content: " world", finish_reason: "stop" */;
    /// let blocks = aggregator.process_chunk(chunk2)?;
    /// assert_eq!(blocks.len(), 1); // Now we have the complete text
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if tool call arguments cannot be parsed as valid JSON. This indicates
    /// either a streaming error or malformed data from the API.
    pub fn process_chunk(&mut self, chunk: OpenAIChunk) -> Result<Vec<ContentBlock>> {
        // Vector to collect completed blocks. Will be empty unless finish_reason is present.
        let mut blocks = Vec::new();

        // A chunk can contain multiple choices, though typically there's only one.
        // Each choice represents a separate generation path (used in n>1 scenarios).
        for choice in chunk.choices {
            // === PHASE 1: ACCUMULATE TEXT DELTAS ===
            // If this chunk contains text content, append it to our buffer.
            // Text arrives as incremental strings: "Hello", " ", "world", etc.
            if let Some(content) = choice.delta.content {
                self.text_buffer.push_str(&content);
            }

            // === PHASE 2: ACCUMULATE TOOL CALL DELTAS ===
            // Tool calls are more complex - they can arrive as multiple interleaved deltas.
            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    // Get or create the partial tool call for this index.
                    // The index is provided by the API and identifies which tool call
                    // this delta belongs to (important when multiple tools are called).
                    let entry = self.tool_calls.entry(tool_call.index).or_default();

                    // Update the ID if present. Usually only in the first delta for this tool.
                    if let Some(id) = tool_call.id {
                        entry.id = Some(id);
                    }

                    // Handle function/tool details
                    if let Some(function) = tool_call.function {
                        // Update the name if present. Usually only in the first delta.
                        if let Some(name) = function.name {
                            entry.name = Some(name);
                        }

                        // Append argument delta. This is where JSON gets assembled character by
                        // character. The API may split JSON at any position, even mid-string:
                        // Chunk 1: "{\"loc"
                        // Chunk 2: "ation\":"
                        // Chunk 3: "\"Paris\"}"
                        if let Some(args) = function.arguments {
                            entry.arguments.push_str(&args);
                        }
                    }
                }
            }

            // === PHASE 3: CHECK FOR COMPLETION ===
            // finish_reason indicates that generation is complete. Common values:
            // - "stop": Natural completion
            // - "tool_calls": Model wants to call tools
            // - "length": Hit max_tokens limit
            // - "content_filter": Content filtered
            if choice.finish_reason.is_some() {
                // === PHASE 3A: FLUSH TEXT BUFFER ===
                // If we accumulated any text, emit it as a TextBlock
                if !self.text_buffer.is_empty() {
                    blocks.push(ContentBlock::Text(TextBlock::new(self.text_buffer.clone())));
                    self.text_buffer.clear();
                }

                // === PHASE 3B: FLUSH AND VALIDATE TOOL CALLS ===
                // drain() consumes the HashMap, giving us ownership of all partial tool calls
                for (_, partial) in self.tool_calls.drain() {
                    // Only emit tool calls that have both ID and name.
                    // Incomplete tool calls are silently dropped (shouldn't happen with valid API).
                    if let (Some(id), Some(name)) = (partial.id, partial.name) {
                        // Parse the accumulated JSON argument string.
                        // If arguments is empty, default to an empty object {}.
                        let input: serde_json::Value = if partial.arguments.is_empty() {
                            serde_json::json!({})
                        } else {
                            // This is where we validate that all the assembled JSON is valid.
                            // If the streaming was corrupted or incomplete, this will error.
                            serde_json::from_str(&partial.arguments).map_err(|e| {
                                Error::stream(format!("Failed to parse tool arguments: {}", e))
                            })?
                        };

                        blocks.push(ContentBlock::ToolUse(ToolUseBlock::new(id, name, input)));
                    }
                }
            }
        }

        Ok(blocks)
    }
}

/// Parses a raw HTTP response body as a Server-Sent Events (SSE) stream.
///
/// Transforms an HTTP streaming response into a stream of parsed [`OpenAIChunk`] objects.
/// This function handles the SSE protocol details, extracting JSON data from the SSE format.
///
/// # SSE Format
///
/// Server-Sent Events is a standard protocol for server-to-client streaming. The format is:
///
/// ```text
/// data: {"id":"msg_123","object":"chat.completion.chunk",...}
///
/// data: {"id":"msg_123","object":"chat.completion.chunk",...}
///
/// data: [DONE]
///
/// ```
///
/// Key characteristics:
/// - Each message starts with `data: `
/// - Messages are separated by double newlines (`\n\n`)
/// - The stream ends with `data: [DONE]`
/// - Everything after `data: ` (until newline) is the payload
///
/// # Arguments
///
/// * `body` - The raw HTTP response from the API request
///
/// # Returns
///
/// A pinned, boxed stream that yields `Result<OpenAIChunk>` for each successfully parsed event.
/// The stream is `Send` to allow use across thread boundaries.
///
/// # Error Handling
///
/// Each stream item can be an error:
/// - **HTTP errors**: Network issues, connection drops (wrapped as [`Error::Http`])
/// - **Parse errors**: Invalid JSON in the SSE data field (wrapped as [`Error::Stream`])
/// - **Protocol errors**: SSE chunks that don't contain a `data:` line (wrapped as [`Error::Stream`])
///
/// Errors are per-chunk, not fatal to the stream. Consumers should handle errors gracefully.
///
/// # Example Flow
///
/// ```text
/// Raw HTTP bytes: b"data: {\"id\":\"123\"}\n\ndata: [DONE]\n\n"
///        ↓
/// bytes_stream() splits into chunks
///        ↓
/// Parse each chunk:
///   - Find lines starting with "data: "
///   - Skip "[DONE]" sentinel
///   - Parse JSON into OpenAIChunk
///        ↓
/// Stream<Result<OpenAIChunk>>
/// ```
///
/// # Protocol Notes
///
/// - **`[DONE]` sentinel**: OpenAI's SSE streams end with `data: [DONE]`. This is not valid
///   JSON, so we skip it rather than attempting to parse.
///
/// - **Chunk boundaries**: HTTP streaming can split data at arbitrary byte positions. Each
///   `bytes_stream()` chunk may contain partial events, complete events, or multiple events.
///   The line-by-line parsing handles this naturally.
///
/// - **UTF-8 handling**: We use `from_utf8_lossy()` to be resilient to split UTF-8 sequences
///   at chunk boundaries, though the API should always send well-formed UTF-8.
///
/// # Usage
///
/// ```rust,ignore
/// let response = client.post(url).send().await?;
/// let mut stream = parse_sse_stream(response);
///
/// while let Some(result) = stream.next().await {
///     match result {
///         Ok(chunk) => process_chunk(chunk),
///         Err(e) => eprintln!("Stream error: {}", e),
///     }
/// }
/// ```
pub fn parse_sse_stream(
    body: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<OpenAIChunk>> + Send>> {
    let stream = body.bytes_stream().filter_map(move |result| async move {
        // Convert HTTP errors to our Error type
        let bytes = match result.map_err(Error::Http) {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        // Convert bytes to string. Use lossy conversion to handle potential
        // UTF-8 boundary splits (though the API should send well-formed UTF-8).
        let text = String::from_utf8_lossy(&bytes);

        // Parse SSE format by examining each line.
        // Format: "data: <payload>\n\n"
        // Lines not starting with "data: " are ignored (e.g., comments, event types).
        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                // Skip the end-of-stream sentinel.
                // OpenAI sends "data: [DONE]" to signal stream completion.
                if data == "[DONE]" {
                    continue;
                }

                // Parse the JSON payload into an OpenAIChunk.
                // This is where we deserialize the actual chunk data.
                let chunk: OpenAIChunk = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(e) => {
                        return Some(Err(Error::stream(format!("Failed to parse chunk: {}", e))));
                    }
                };

                return Some(Ok(chunk));
            }
        }

        // If we processed all lines and found no "data: " line, skip this chunk.
        // This handles heartbeats, comments, and other SSE events gracefully.
        None
    });

    // Pin the stream to the heap and box it for dynamic dispatch.
    // This allows the function to return a uniform type regardless of the
    // concrete stream implementation.
    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OpenAIChoice, OpenAIDelta, OpenAIFunctionDelta, OpenAIToolCallDelta};

    #[test]
    fn test_tool_call_aggregator_text() {
        let mut aggregator = ToolCallAggregator::new();

        let chunk = OpenAIChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: Some("Hello ".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        };

        let blocks = aggregator.process_chunk(chunk).unwrap();
        assert_eq!(blocks.len(), 0); // Not finished yet

        let chunk2 = OpenAIChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: Some("world".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };

        let blocks = aggregator.process_chunk(chunk2).unwrap();
        assert_eq!(blocks.len(), 1);

        match &blocks[0] {
            ContentBlock::Text(text) => assert_eq!(text.text, "Hello world"),
            _ => panic!("Expected text block"),
        }
    }

    #[test]
    fn test_tool_call_aggregator_tool() {
        let mut aggregator = ToolCallAggregator::new();

        let chunk = OpenAIChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(vec![OpenAIToolCallDelta {
                        index: 0,
                        id: Some("call_123".to_string()),
                        call_type: Some("function".to_string()),
                        function: Some(OpenAIFunctionDelta {
                            name: Some("get_weather".to_string()),
                            arguments: Some(r#"{"location":"#.to_string()),
                        }),
                    }]),
                },
                finish_reason: None,
            }],
        };

        let blocks = aggregator.process_chunk(chunk).unwrap();
        assert_eq!(blocks.len(), 0);

        let chunk2 = OpenAIChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(vec![OpenAIToolCallDelta {
                        index: 0,
                        id: None,
                        call_type: None,
                        function: Some(OpenAIFunctionDelta {
                            name: None,
                            arguments: Some(r#""Paris"}"#.to_string()),
                        }),
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
        };

        let blocks = aggregator.process_chunk(chunk2).unwrap();
        assert_eq!(blocks.len(), 1);

        match &blocks[0] {
            ContentBlock::ToolUse(tool) => {
                assert_eq!(tool.id, "call_123");
                assert_eq!(tool.name, "get_weather");
                assert_eq!(tool.input["location"], "Paris");
            }
            _ => panic!("Expected tool use block"),
        }
    }
}
