/// OpenAI API message format for serialization.
///
/// This struct represents the wire format for messages when communicating
/// with OpenAI-compatible APIs. It differs from the internal [`Message`]
/// type to accommodate the specific serialization requirements of the
/// OpenAI API.
///
/// # Key Differences from Internal Message Type
///
/// - Content is a flat string rather than structured blocks
/// - Tool calls are represented in OpenAI's specific format
/// - Supports both sending tool calls (via `tool_calls`) and tool results
///   (via `tool_call_id`)
///
/// # Serialization
///
/// Optional fields are skipped when `None` to keep payloads minimal.
///
/// # Usage
///
/// This type is typically created by the SDK internally when converting
/// from [`Message`] to API format. Users rarely need to construct these
/// directly.
///
/// # OpenAI Content Format
///
/// OpenAI content format supporting both string and array.
///
/// For backward compatibility, text-only messages use string format.
/// Messages with images use array format with multiple content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIContent {
    /// Simple text string (backward compatible)
    Text(String),
    /// Array of content parts (text and/or images)
    Parts(Vec<OpenAIContentPart>),
}

/// A single content part in an OpenAI message.
///
/// Can be either text or an image URL. This is a tagged enum that prevents
/// invalid states (e.g., having both text and image_url, or neither).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIContentPart {
    /// Text content part
    Text {
        /// The text content
        text: String,
    },
    /// Image URL content part
    #[serde(rename = "image_url")]
    ImageUrl {
        /// The image URL details
        image_url: OpenAIImageUrl,
    },
}

impl OpenAIContentPart {
    /// Creates a text content part.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::OpenAIContentPart;
    ///
    /// let part = OpenAIContentPart::text("Hello world");
    /// ```
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Creates an image content part from a validated ImageBlock.
    ///
    /// This is the preferred way to create image content parts as it ensures
    /// the image URL has been validated against security issues (XSS, file disclosure, etc.)
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::{OpenAIContentPart, ImageBlock, ImageDetail};
    ///
    /// let image = ImageBlock::from_url("https://example.com/img.jpg")
    ///     .expect("Valid URL");
    /// let part = OpenAIContentPart::from_image(&image);
    /// ```
    pub fn from_image(image: &ImageBlock) -> Self {
        Self::ImageUrl {
            image_url: OpenAIImageUrl {
                url: image.url().to_string(),
                detail: Some(image.detail().to_string()),
            },
        }
    }

    /// Creates an image URL content part directly (DEPRECATED).
    ///
    /// # Security Warning
    ///
    /// This method bypasses validation checks performed by `ImageBlock::from_url()`
    /// and `ImageBlock::from_base64()`. Prefer using `from_image()` instead.
    ///
    /// # Deprecation
    ///
    /// This method is deprecated and will be removed in v1.0. Use `from_image()` instead.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::{OpenAIContentPart, ImageDetail};
    ///
    /// // Deprecated approach:
    /// let part = OpenAIContentPart::image_url("https://example.com/img.jpg", ImageDetail::High);
    ///
    /// // Preferred approach:
    /// use open_agent::ImageBlock;
    /// let image = ImageBlock::from_url("https://example.com/img.jpg").expect("Valid URL");
    /// let part = OpenAIContentPart::from_image(&image);
    /// ```
    #[deprecated(
        since = "0.6.0",
        note = "Use `from_image()` instead to ensure proper validation"
    )]
    pub fn image_url(url: impl Into<String>, detail: ImageDetail) -> Self {
        Self::ImageUrl {
            image_url: OpenAIImageUrl {
                url: url.into(),
                detail: Some(detail.to_string()),
            },
        }
    }
}

/// OpenAI image URL structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIImageUrl {
    /// Image URL or data URI
    pub url: String,
    /// Detail level: "low", "high", or "auto"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessage {
    /// Message role as a string ("system", "user", "assistant", "tool").
    pub role: String,

    /// Message content (string for text-only, array for text+images).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAIContent>,

    /// Tool calls requested by the assistant (assistant messages only).
    ///
    /// When the model wants to call tools, this field contains the list
    /// of tool invocations with their parameters. Only present in assistant
    /// messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,

    /// ID of the tool call this message is responding to (tool messages only).
    ///
    /// When sending tool results back to the model, this field links the
    /// result to the original tool call request. Only present in tool
    /// messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// OpenAI tool call representation in API messages.
///
/// Represents a request from the model to execute a specific function/tool.
/// This is the wire format used in the OpenAI API, distinct from the internal
/// [`ToolUseBlock`] representation.
///
/// # Structure
///
/// Each tool call has:
/// - A unique ID for correlation with results
/// - A type (always "function" in current OpenAI API)
/// - Function details (name and arguments)
///
/// # Example JSON
///
/// ```json
/// {
///   "id": "call_abc123",
///   "type": "function",
///   "function": {
///     "name": "get_weather",
///     "arguments": "{\"location\":\"San Francisco\"}"
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    /// Unique identifier for this tool call.
    ///
    /// Generated by the model. Used to correlate tool results back to
    /// this specific call.
    pub id: String,

    /// Type of the call (always "function" in current API).
    ///
    /// The `rename` attribute ensures this serializes as `"type"` in JSON
    /// since `type` is a Rust keyword.
    #[serde(rename = "type")]
    pub call_type: String,

    /// Function/tool details (name and arguments).
    pub function: OpenAIFunction,
}

/// OpenAI function call details.
///
/// Contains the function name and its arguments in the OpenAI API format.
/// Note that arguments are serialized as a JSON string, not a JSON object,
/// which is an OpenAI API quirk.
///
/// # Arguments Format
///
/// The `arguments` field is a **JSON string**, not a parsed JSON object.
/// For example: `"{\"x\": 1, \"y\": 2}"` not `{"x": 1, "y": 2}`.
/// This must be parsed before use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunction {
    /// Name of the function/tool to call.
    pub name: String,

    /// Function arguments as a **JSON string** (OpenAI API quirk).
    ///
    /// Must be parsed as JSON before use. For example, this might contain
    /// the string `"{\"location\":\"NYC\",\"units\":\"fahrenheit\"}"` which
    /// needs to be parsed into an actual JSON value.
    pub arguments: String,
}

/// Complete request payload for OpenAI chat completions API.
///
/// This struct is serialized and sent as the request body when making
/// API calls to OpenAI-compatible endpoints. It includes the model,
/// conversation history, and configuration parameters.
///
/// # Streaming
///
/// The SDK always uses streaming mode (`stream: true`) to enable real-time
/// response processing and better user experience.
///
/// # Optional Fields
///
/// Fields marked with `skip_serializing_if` are omitted from the JSON payload
/// when `None`, allowing the API provider to use its defaults.
///
/// # Example
///
/// ```ignore
/// use open_agent_sdk::types::{OpenAIRequest, OpenAIMessage};
///
/// let request = OpenAIRequest {
///     model: "gpt-4".to_string(),
///     messages: vec![
///         OpenAIMessage {
///             role: "user".to_string(),
///             content: "Hello!".to_string(),
///             tool_calls: None,
///             tool_call_id: None,
///         }
///     ],
///     stream: true,
///     max_tokens: Some(1000),
///     temperature: Some(0.7),
///     tools: None,
/// };
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct OpenAIRequest {
    /// Model identifier (e.g., "gpt-4", "qwen2.5-32b-instruct").
    pub model: String,

    /// Conversation history as a sequence of messages.
    ///
    /// Includes system prompt, user messages, assistant responses, and
    /// tool results. Order matters - messages are processed sequentially.
    pub messages: Vec<OpenAIMessage>,

    /// Whether to stream the response.
    ///
    /// The SDK always sets this to `true` for better user experience.
    /// Streaming allows incremental processing of responses rather than
    /// waiting for the entire completion.
    pub stream: bool,

    /// Maximum tokens to generate (optional).
    ///
    /// `None` uses the provider's default. Some providers require this
    /// to be set explicitly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature (optional).
    ///
    /// `None` uses the provider's default. Controls randomness in
    /// generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Tools/functions available to the model (optional).
    ///
    /// When present, enables function calling. Each tool is described
    /// with a JSON schema defining its parameters. `None` means no
    /// tools are available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
}

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
