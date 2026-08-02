/// Identifies the sender/role of a message in the conversation.
///
/// This enum follows the standard chat completion role system used by most
/// LLM APIs. The role determines how the message is interpreted and processed.
///
/// # Serialization
///
/// Serializes to lowercase strings via serde (`"system"`, `"user"`, etc.)
/// to match OpenAI API format.
///
/// # Role Semantics
///
/// - [`System`](MessageRole::System): Establishes context, instructions, and behavior
/// - [`User`](MessageRole::User): Input from the human or calling application
/// - [`Assistant`](MessageRole::Assistant): Response from the AI model
/// - [`Tool`](MessageRole::Tool): Results from tool/function execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System message that establishes agent behavior and context.
    ///
    /// Typically the first message in a conversation. Used for instructions,
    /// personality definition, and constraints that apply throughout the
    /// conversation.
    System,

    /// User message representing human or application input.
    ///
    /// The prompt or query that the agent should respond to. In multi-turn
    /// conversations, user messages alternate with assistant messages.
    User,

    /// Assistant message containing the AI model's response.
    ///
    /// Can include text, tool use requests, or both. When the model wants to
    /// call a tool, it includes ToolUseBlock content.
    Assistant,

    /// Tool result message containing function execution results.
    ///
    /// Sent back to the model after executing a requested tool. Contains the
    /// tool's output that the model can use in its next response.
    Tool,
}

/// Multi-modal content blocks that can appear in messages.
///
/// Messages are composed of one or more content blocks, allowing rich,
/// structured communication between the user, assistant, and tools.
///
/// # Serialization
///
/// Uses serde's "externally tagged" enum format with a `"type"` field:
/// ```json
/// {"type": "text", "text": "Hello"}
/// {"type": "tool_use", "id": "call_123", "name": "search", "input": {...}}
/// {"type": "tool_result", "tool_use_id": "call_123", "content": {...}}
/// ```
///
/// # Block Types
///
/// - [`Text`](ContentBlock::Text): Simple text content
/// - [`Image`](ContentBlock::Image): Image content (URL or base64)
/// - [`ToolUse`](ContentBlock::ToolUse): Request from model to execute a tool
/// - [`ToolResult`](ContentBlock::ToolResult): Result of tool execution
///
/// # Usage
///
/// Messages can contain multiple blocks. For example, a user message might
/// include text and an image, or an assistant message might include text
/// followed by a tool use request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content block containing a string message.
    Text(TextBlock),

    /// Image content block for vision-capable models.
    Image(ImageBlock),

    /// Tool use request from the model to execute a function.
    ToolUse(ToolUseBlock),

    /// Tool execution result sent back to the model.
    ToolResult(ToolResultBlock),
}

/// Simple text content in a message.
///
/// The most common content type, representing plain text communication.
/// Both users and assistants primarily use text blocks for their messages.
///
/// # Example
///
/// ```
/// use open_agent::{TextBlock, ContentBlock};
///
/// let block = TextBlock::new("Hello, world!");
/// let content = ContentBlock::Text(block);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBlock {
    /// The text content.
    pub text: String,
}

impl TextBlock {
    /// Creates a new text block from any string-like type.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::TextBlock;
    ///
    /// let block = TextBlock::new("Hello");
    /// assert_eq!(block.text, "Hello");
    /// ```
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Tool use request from the AI model.
///
/// When the model determines it needs to call a tool/function, it returns
/// a ToolUseBlock specifying which tool to call and with what parameters.
/// The application must then execute the tool and return results via
/// [`ToolResultBlock`].
///
/// # Fields
///
/// - `id`: Unique identifier for this tool call, used to correlate results
/// - `name`: Name of the tool to execute (must match a registered tool)
/// - `input`: JSON parameters to pass to the tool
///
/// # Example
///
/// ```
/// use open_agent::{ToolUseBlock, ContentBlock};
/// use serde_json::json;
///
/// let block = ToolUseBlock::new(
///     "call_123",
///     "calculate",
///     json!({"expression": "2 + 2"})
/// );
/// assert_eq!(block.id(), "call_123");
/// assert_eq!(block.name(), "calculate");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUseBlock {
    /// Unique identifier for this tool call.
    ///
    /// Generated by the model. Used to correlate the tool result back to
    /// this specific request, especially when multiple tools are called.
    id: String,

    /// Name of the tool to execute.
    ///
    /// Must match the name of a tool that was provided in the agent's
    /// configuration, otherwise execution will fail.
    name: String,

    /// JSON parameters to pass to the tool.
    ///
    /// The structure should match the tool's input schema. The tool's
    /// execution function receives this value as input.
    input: serde_json::Value,
}

impl ToolUseBlock {
    /// Creates a new tool use block.
    ///
    /// # Parameters
    ///
    /// - `id`: Unique identifier for this tool call
    /// - `name`: Name of the tool to execute
    /// - `input`: JSON parameters for the tool
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::ToolUseBlock;
    /// use serde_json::json;
    ///
    /// let block = ToolUseBlock::new(
    ///     "call_abc",
    ///     "search",
    ///     json!({"query": "Rust async programming"})
    /// );
    /// ```
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    /// Returns the unique identifier for this tool call.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the name of the tool to execute.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the JSON parameters for the tool.
    pub fn input(&self) -> &serde_json::Value {
        &self.input
    }
}

/// Tool execution result sent back to the model.
///
/// After executing a tool requested via [`ToolUseBlock`], the application
/// creates a ToolResultBlock containing the tool's output and sends it back
/// to the model. The model then uses this information in its next response.
///
/// # Fields
///
/// - `tool_use_id`: Must match the `id` from the corresponding ToolUseBlock
/// - `content`: JSON result from the tool execution
///
/// # Example
///
/// ```
/// use open_agent::{ToolResultBlock, ContentBlock};
/// use serde_json::json;
///
/// let result = ToolResultBlock::new(
///     "call_123",
///     json!({"result": 4})
/// );
/// assert_eq!(result.tool_use_id(), "call_123");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultBlock {
    /// ID of the tool use request this result corresponds to.
    ///
    /// Must match the `id` field from the ToolUseBlock that requested
    /// this tool execution. This correlation is essential for the model
    /// to understand which tool call produced which result.
    tool_use_id: String,

    /// JSON result from executing the tool.
    ///
    /// Contains the tool's output data. Can be any valid JSON structure -
    /// the model will interpret it based on the tool's description and
    /// output schema.
    content: serde_json::Value,
}

impl ToolResultBlock {
    /// Creates a new tool result block.
    ///
    /// # Parameters
    ///
    /// - `tool_use_id`: ID from the corresponding ToolUseBlock
    /// - `content`: JSON result from tool execution
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::ToolResultBlock;
    /// use serde_json::json;
    ///
    /// let result = ToolResultBlock::new(
    ///     "call_xyz",
    ///     json!({
    ///         "status": "success",
    ///         "data": {"temperature": 72}
    ///     })
    /// );
    /// ```
    pub fn new(tool_use_id: impl Into<String>, content: serde_json::Value) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content,
        }
    }

    /// Returns the ID of the tool use request this result corresponds to.
    pub fn tool_use_id(&self) -> &str {
        &self.tool_use_id
    }

    /// Returns the JSON result from executing the tool.
    pub fn content(&self) -> &serde_json::Value {
        &self.content
    }
}
