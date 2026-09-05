//! OpenAI streaming wire types.
//!
//! This is a real module so mutation testing can inspect `reasoning_delta`.

use serde::Deserialize;

/// One JSON event from an OpenAI chat-completions SSE response.
///
/// Contains response metadata and incremental choices. `[DONE]` is an SSE sentinel,
/// not an instance of this type.
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

/// One completion choice with a delta and optional finish reason.
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

    /// Optional server finish reason, such as `stop`, `length`, or `tool_calls`.
    ///
    /// Some servers never report it; `None` does not imply more content will arrive.
    pub finish_reason: Option<String>,
}

/// New text, reasoning, or tool-call fragments from one choice.
///
/// Text and reasoning are forwarded as they arrive. Tool arguments are assembled
/// across deltas before they can be parsed as JSON.
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

    /// Tool-call fragments, routed by each call's index.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCallDelta>>,

    /// Reasoning from DeepSeek-compatible servers.
    ///
    /// Kept separate from `content`; see [`OpenAIDelta::reasoning_delta`] for precedence
    /// when a server sends both reasoning fields.
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

/// A tool-call update routed by index.
///
/// Parallel calls may interleave fragments. The ID, function name, and arguments
/// can arrive in separate deltas.
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

/// Function name and JSON-argument fragments for a streamed tool call.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIFunctionDelta {
    /// Function/tool name (only in first delta for this function).
    ///
    /// Present when the model first starts calling this function, then
    /// omitted in subsequent chunks. The SDK stores this to know which
    /// tool to execute.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// A fragment of JSON text. Concatenate the argument fragments before parsing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}
