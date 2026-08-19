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
