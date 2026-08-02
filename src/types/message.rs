/// A complete message in a conversation.
///
/// Messages are the primary unit of communication in the agent system. Each
/// message has a role (who sent it) and content (what it contains). Content
/// is structured as a vector of blocks to support multi-modal communication.
///
/// # Structure
///
/// - `role`: Who sent the message ([`MessageRole`])
/// - `content`: What the message contains (one or more [`ContentBlock`]s)
///
/// # Message Patterns
///
/// ## Simple Text Message
/// ```
/// use open_agent::Message;
///
/// let msg = Message::user("What's the weather?");
/// ```
///
/// ## Assistant Response with Tool Call
/// ```
/// use open_agent::{Message, ContentBlock, TextBlock, ToolUseBlock};
/// use serde_json::json;
///
/// let msg = Message::assistant(vec![
///     ContentBlock::Text(TextBlock::new("Let me check that for you.")),
///     ContentBlock::ToolUse(ToolUseBlock::new(
///         "call_123",
///         "get_weather",
///         json!({"location": "San Francisco"})
///     ))
/// ]);
/// ```
///
/// ## Tool Result
/// ```
/// use open_agent::{Message, ContentBlock, ToolResultBlock};
/// use serde_json::json;
///
/// let msg = Message::user_with_blocks(vec![
///     ContentBlock::ToolResult(ToolResultBlock::new(
///         "call_123",
///         json!({"temp": 72, "conditions": "sunny"})
///     ))
/// ]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// The role/sender of this message.
    pub role: MessageRole,

    /// The content blocks that make up this message.
    ///
    /// A message can contain multiple blocks of different types. For example,
    /// an assistant message might have both text and tool use blocks.
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Creates a new message with the specified role and content.
    ///
    /// This is the most general constructor. For convenience, use the
    /// role-specific constructors like [`user()`](Message::user),
    /// [`assistant()`](Message::assistant), etc.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::{Message, MessageRole, ContentBlock, TextBlock};
    ///
    /// let msg = Message::new(
    ///     MessageRole::User,
    ///     vec![ContentBlock::Text(TextBlock::new("Hello"))]
    /// );
    /// ```
    pub fn new(role: MessageRole, content: Vec<ContentBlock>) -> Self {
        Self { role, content }
    }

    /// Creates a user message with simple text content.
    ///
    /// This is the most common way to create user messages. For more complex
    /// content with multiple blocks, use [`user_with_blocks()`](Message::user_with_blocks).
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::Message;
    ///
    /// let msg = Message::user("What is 2+2?");
    /// ```
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::Text(TextBlock::new(text))],
        }
    }

    /// Creates an assistant message with the specified content blocks.
    ///
    /// Assistant messages often contain multiple content blocks (text + tool use).
    /// This method takes a vector of blocks for maximum flexibility.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::{Message, ContentBlock, TextBlock};
    ///
    /// let msg = Message::assistant(vec![
    ///     ContentBlock::Text(TextBlock::new("The answer is 4"))
    /// ]);
    /// ```
    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content,
        }
    }

    /// Creates a system message with simple text content.
    ///
    /// System messages establish the agent's behavior and context. They're
    /// typically sent at the start of a conversation.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::Message;
    ///
    /// let msg = Message::system("You are a helpful assistant. Be concise.");
    /// ```
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: vec![ContentBlock::Text(TextBlock::new(text))],
        }
    }

    /// Creates a user message with custom content blocks.
    ///
    /// Use this when you need to send structured content beyond simple text,
    /// such as tool results. For simple text messages, prefer
    /// [`user()`](Message::user).
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::{Message, ContentBlock, ToolResultBlock};
    /// use serde_json::json;
    ///
    /// let msg = Message::user_with_blocks(vec![
    ///     ContentBlock::ToolResult(ToolResultBlock::new(
    ///         "call_123",
    ///         json!({"result": "success"})
    ///     ))
    /// ]);
    /// ```
    pub fn user_with_blocks(content: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::User,
            content,
        }
    }

    /// Creates a user message with text and an image from a URL.
    ///
    /// This is a convenience method for the common pattern of sending text with
    /// an image. The image uses `ImageDetail::Auto` by default. For more control
    /// over detail level, use [`user_with_image_detail()`](Message::user_with_image_detail).
    ///
    /// # Arguments
    ///
    /// * `text` - The text prompt
    /// * `image_url` - URL of the image (http/https or data URI)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if the image URL is invalid (empty, wrong scheme, etc.)
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::Message;
    ///
    /// let msg = Message::user_with_image(
    ///     "What's in this image?",
    ///     "https://example.com/photo.jpg"
    /// )?;
    /// # Ok::<(), open_agent::Error>(())
    /// ```
    pub fn user_with_image(
        text: impl Into<String>,
        image_url: impl Into<String>,
    ) -> crate::Result<Self> {
        Ok(Self {
            role: MessageRole::User,
            content: vec![
                ContentBlock::Text(TextBlock::new(text)),
                ContentBlock::Image(ImageBlock::from_url(image_url)?),
            ],
        })
    }

    /// Creates a user message with text and an image with specified detail level.
    ///
    /// Use this when you need control over the image detail level for token cost
    /// management. On OpenAI's Vision API: `ImageDetail::Low` uses ~85 tokens,
    /// `ImageDetail::High` uses more tokens based on image dimensions, and
    /// `ImageDetail::Auto` lets the model decide. Local models may have very different token costs.
    ///
    /// # Arguments
    ///
    /// * `text` - The text prompt
    /// * `image_url` - URL of the image (http/https or data URI)
    /// * `detail` - Detail level (Low, High, or Auto)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if the image URL is invalid (empty, wrong scheme, etc.)
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::{Message, ImageDetail};
    ///
    /// let msg = Message::user_with_image_detail(
    ///     "Analyze this diagram in detail",
    ///     "https://example.com/diagram.png",
    ///     ImageDetail::High
    /// )?;
    /// # Ok::<(), open_agent::Error>(())
    /// ```
    pub fn user_with_image_detail(
        text: impl Into<String>,
        image_url: impl Into<String>,
        detail: ImageDetail,
    ) -> crate::Result<Self> {
        Ok(Self {
            role: MessageRole::User,
            content: vec![
                ContentBlock::Text(TextBlock::new(text)),
                ContentBlock::Image(ImageBlock::from_url(image_url)?.with_detail(detail)),
            ],
        })
    }

    /// Creates a user message with text and a base64-encoded image.
    ///
    /// This is useful when you have image data in memory and want to send it
    /// without uploading to a URL first. The image will be encoded as a data URI.
    ///
    /// # Arguments
    ///
    /// * `text` - The text prompt
    /// * `base64_data` - Base64-encoded image data
    /// * `mime_type` - MIME type (e.g., "image/png", "image/jpeg")
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if the base64 data or MIME type is invalid
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::Message;
    ///
    /// // Use properly formatted base64 (length divisible by 4, valid chars)
    /// let base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    /// let msg = Message::user_with_base64_image(
    ///     "What's this image?",
    ///     base64_data,
    ///     "image/png"
    /// )?;
    /// # Ok::<(), open_agent::Error>(())
    /// ```
    pub fn user_with_base64_image(
        text: impl Into<String>,
        base64_data: impl AsRef<str>,
        mime_type: impl AsRef<str>,
    ) -> crate::Result<Self> {
        Ok(Self {
            role: MessageRole::User,
            content: vec![
                ContentBlock::Text(TextBlock::new(text)),
                ContentBlock::Image(ImageBlock::from_base64(base64_data, mime_type)?),
            ],
        })
    }
}
