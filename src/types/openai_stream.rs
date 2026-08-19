//! Wire types for the OpenAI-compatible streaming response.
//!
//! Split out of `openai.rs` so the request/message half and the chunk/delta half each stay
//! within the repository's source-file limits. A real module rather than an `include!`
//! fragment because `OpenAIDelta::reasoning_delta` is executable logic that the mutation
//! gate must see; the re-export in `types.rs` keeps the public paths unchanged. The types
//! here have public fields throughout, so nothing depends on module-private access.

use serde::Deserialize;

/// A single chunk from OpenAI's streaming response.
///
/// When the SDK requests streaming responses (`stream: true`), the API
/// returns the response incrementally as a series of chunks. Each chunk
/// represents a small piece of the complete response, allowing the SDK
/// to process and display content as it's generated.
///
/// # Streaming Architecture
///
/// Instead of waiting for the entire response, streaming sends many small
/// chunks in rapid succession. Each chunk contains:
/// - Metadata (id, model, timestamp)
/// - One or more choices (usually just one for single completions)
/// - Incremental deltas with new content
///
/// # Server-Sent Events Format
///
/// Chunks are transmitted as Server-Sent Events (SSE) over HTTP:
/// ```text
/// data: {"id":"chunk_1","object":"chat.completion.chunk",...}
/// data: {"id":"chunk_2","object":"chat.completion.chunk",...}
/// data: [DONE]
/// ```
///
/// # Example Chunk JSON
///
/// ```json
/// {
///   "id": "chatcmpl-123",
///   "object": "chat.completion.chunk",
///   "created": 1677652288,
///   "model": "gpt-4",
///   "choices": [{
///     "index": 0,
///     "delta": {"content": "Hello"},
///     "finish_reason": null
///   }]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChunk {
    /// Unique identifier for this completion.
    ///
    /// All chunks in a single streaming response share the same ID.
    /// Not actively used by the SDK but preserved for debugging.
    #[allow(dead_code)]
    pub id: String,

    /// Object type (always "chat.completion.chunk" for streaming).
    ///
    /// Not actively used by the SDK but preserved for debugging.
    #[allow(dead_code)]
    pub object: String,

    /// Unix timestamp of when this chunk was created.
    ///
    /// Not actively used by the SDK but preserved for debugging.
    #[allow(dead_code)]
    pub created: i64,

    /// Model that generated this chunk.
    ///
    /// Not actively used by the SDK but preserved for debugging.
    #[allow(dead_code)]
    pub model: String,

    /// Array of completion choices (usually contains one element).
    ///
    /// Each choice represents a possible completion. In normal usage,
    /// there's only one choice per chunk. This is the critical field
    /// that the SDK processes to extract content and tool calls.
    pub choices: Vec<OpenAIChoice>,
}

/// A single choice/completion option in a streaming chunk.
///
/// In streaming responses, each chunk can theoretically contain multiple
/// choices (parallel completions), but in practice there's usually just one.
/// Each choice contains a delta with incremental updates and optionally a
/// finish reason when the generation is complete.
///
/// # Delta vs Complete Content
///
/// Unlike non-streaming responses that send complete messages, streaming
/// sends deltas - just the new content added in this chunk. The SDK
/// accumulates these deltas to build the complete response.
///
/// # Finish Reason
///
/// - `None`: More content is coming
/// - `Some("stop")`: Normal completion
/// - `Some("length")`: Hit max token limit
/// - `Some("tool_calls")`: Model wants to call tools
/// - `Some("content_filter")`: Blocked by content policy
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChoice {
    /// Index of this choice in the choices array.
    ///
    /// Usually 0 since most requests generate a single completion.
    /// Not actively used by the SDK but preserved for debugging.
    #[allow(dead_code)]
    pub index: u32,

    /// Incremental update/delta for this chunk.
    ///
    /// Contains the new content, tool calls, or other updates added in
    /// this specific chunk. The SDK processes this to update its internal
    /// state and accumulate the full response.
    pub delta: OpenAIDelta,

    /// Reason why generation finished (None if still generating).
    ///
    /// Only present in the final chunk of a stream:
    /// - `None`: Generation is still in progress
    /// - `Some("stop")`: Completed normally
    /// - `Some("length")`: Hit token limit
    /// - `Some("tool_calls")`: Model requested tools
    /// - `Some("content_filter")`: Content was filtered
    ///
    /// The SDK uses this to detect completion and determine next actions.
    pub finish_reason: Option<String>,
}

/// Incremental update in a streaming chunk.
///
/// Represents the new content/changes added in this specific chunk.
/// Unlike complete messages, deltas only contain what's new, not the
/// entire accumulated content. The SDK accumulates these deltas to
/// build the complete response.
///
/// # Incremental Nature
///
/// If the complete response is "Hello, world!", the deltas might be:
/// 1. `content: Some("Hello")`
/// 2. `content: Some(", ")`
/// 3. `content: Some("world")`
/// 4. `content: Some("!")`
///
/// The SDK concatenates these to build the full text.
///
/// # Tool Call Deltas
///
/// Tool calls are also streamed incrementally. The first delta might
/// include the tool ID and name, while subsequent deltas stream the
/// arguments JSON string piece by piece.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIDelta {
    /// Role of the message (only in first chunk).
    ///
    /// Typically "assistant". Only appears in the first delta of a response
    /// to establish who's speaking. Subsequent deltas omit this field.
    /// Not actively used by the SDK but preserved for completeness.
    #[allow(dead_code)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Incremental text content added in this chunk.
    ///
    /// Contains the new text tokens generated. `None` if this chunk doesn't
    /// add text (e.g., it might only have tool call updates). The SDK
    /// concatenates these across chunks to build the complete response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Incremental tool call updates added in this chunk.
    ///
    /// When the model wants to call tools, tool call information is streamed
    /// incrementally. Each delta might add to different parts of the tool
    /// call (ID, name, arguments). The SDK accumulates these to reconstruct
    /// complete tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCallDelta>>,

    /// Chain-of-thought delta as streamed by DeepSeek and compatible servers.
    ///
    /// Declared so that the SDK *decides* what happens to reasoning rather than letting
    /// serde drop it as an unknown field. Reasoning is routed to its own channel and is
    /// never appended to [`OpenAIDelta::content`] — concatenating deliberation prose into
    /// the content a caller parses as JSON is exactly the corruption this separation
    /// prevents. See [`OpenAIDelta::reasoning_delta`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,

    /// Chain-of-thought delta as streamed by OpenRouter and compatible gateways.
    ///
    /// The same content as [`OpenAIDelta::reasoning_content`] under the field name
    /// OpenRouter uses. Servers send one or the other; a few mirror both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

impl OpenAIDelta {
    /// Returns this delta's reasoning text from whichever channel the server used.
    ///
    /// `reasoning_content` (DeepSeek) wins over `reasoning` (OpenRouter) when a gateway
    /// mirrors the trace on both fields, so a duplicated trace is counted once rather than
    /// twice.
    pub fn reasoning_delta(&self) -> Option<&str> {
        self.reasoning_content
            .as_deref()
            .or(self.reasoning.as_deref())
    }
}

/// Incremental update for a tool call in streaming.
///
/// Tool calls are streamed piece-by-piece, with different chunks potentially
/// updating different parts. The SDK must accumulate these deltas to
/// reconstruct complete tool calls.
///
/// # Streaming Pattern
///
/// A complete tool call is typically streamed as:
/// 1. First chunk: `index: 0, id: Some("call_123"), type: Some("function")`
/// 2. Second chunk: `index: 0, function: Some(FunctionDelta { name: Some("search"), ... })`
/// 3. Multiple chunks: `index: 0, function: Some(FunctionDelta { arguments: Some("part") })`
///
/// The SDK uses the `index` to know which tool call to update, as multiple
/// tool calls can be streamed simultaneously.
///
/// # Index-Based Accumulation
///
/// The `index` field is crucial for tracking which tool call is being updated.
/// When the model calls multiple tools, each has a different index, and deltas
/// specify which one they're updating.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIToolCallDelta {
    /// Index identifying which tool call this delta updates.
    ///
    /// When multiple tools are called, each has an index (0, 1, 2, ...).
    /// The SDK uses this to route delta updates to the correct tool call
    /// in its accumulation buffer.
    pub index: u32,

    /// Tool call ID (only in first delta for this tool call).
    ///
    /// Generated by the model. Present in the first chunk for each tool
    /// call, then omitted in subsequent chunks. The SDK stores this to
    /// correlate results later.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Type of call (always "function" when present).
    ///
    /// Only appears in the first delta for each tool call. Subsequent
    /// deltas omit this field. Not actively used by the SDK but preserved
    /// for completeness.
    #[allow(dead_code)]
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub call_type: Option<String>,

    /// Incremental function details (name and/or arguments).
    ///
    /// Contains partial updates to the function name and arguments.
    /// The SDK accumulates these across chunks to build the complete
    /// function call specification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<OpenAIFunctionDelta>,
}

/// Incremental update for function details in streaming tool calls.
///
/// As the model streams a tool call, the function name and arguments are
/// sent incrementally. The name usually comes first in one chunk, then
/// arguments are streamed piece-by-piece as a JSON string.
///
/// # Arguments Streaming
///
/// The arguments field is particularly important to understand. It contains
/// **fragments of a JSON string** that must be accumulated and then parsed:
///
/// 1. Chunk 1: `arguments: Some("{")`
/// 2. Chunk 2: `arguments: Some("\"query\":")`
/// 3. Chunk 3: `arguments: Some("\"hello\"")`
/// 4. Chunk 4: `arguments: Some("}")`
///
/// The SDK concatenates these into `"{\"query\":\"hello\"}"` and then
/// parses it as JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIFunctionDelta {
    /// Function/tool name (only in first delta for this function).
    ///
    /// Present when the model first starts calling this function, then
    /// omitted in subsequent chunks. The SDK stores this to know which
    /// tool to execute.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Incremental fragment of the arguments JSON string.
    ///
    /// Contains a piece of the complete JSON arguments string. The SDK
    /// must concatenate all argument fragments across chunks, then parse
    /// the complete string as JSON to get the actual parameters.
    ///
    /// For example, if the complete arguments should be:
    /// `{"x": 1, "y": 2}`
    ///
    /// This might be streamed as:
    /// - `Some("{\"x\": ")`
    /// - `Some("1, \"y\": ")`
    /// - `Some("2}")`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}
