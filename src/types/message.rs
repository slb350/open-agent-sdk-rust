/// A complete message in a conversation.
///
/// Each message has a role and a sequence of text, image, tool-call, or tool-result
/// blocks.
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
    /// Creates a message with the supplied role and content blocks.
    pub fn new(role: MessageRole, content: Vec<ContentBlock>) -> Self {
        Self { role, content }
    }

    /// Creates a user message containing one text block.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::Text(TextBlock::new(text))],
        }
    }

    /// Creates an assistant message with the supplied text and/or tool-call blocks.
    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content,
        }
    }

    /// Creates a system message containing one text block.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: vec![ContentBlock::Text(TextBlock::new(text))],
        }
    }

    /// Creates a user message with arbitrary blocks, including tool results.
    ///
    /// See [`Message`] for a tool-result example.
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
    /// Detail is a provider hint; its effect on quality and token costs depends on
    /// the endpoint and model.
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
