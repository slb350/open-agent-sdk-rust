/// Snapshot passed to an automatic pre-tool hook.
///
/// `history` contains structured messages up to the pending call, before its result
/// is added. A decision can block the call or replace its input.
#[derive(Debug, Clone)]
pub struct PreToolUseEvent {
    /// Name of the tool about to be executed (e.g., "Bash", "Read", "Edit")
    pub tool_name: String,
    /// Input parameters for the tool as a JSON value
    pub tool_input: Value,
    /// Unique identifier for this tool use (for correlation with PostToolUseEvent)
    pub tool_use_id: String,
    /// Structured JSON snapshot of conversation history before this tool call executes.
    pub history: Vec<Value>,
}

impl PreToolUseEvent {
    /// Creates a new PreToolUseEvent.
    ///
    /// This constructor is typically called by the agent runtime, not by user code.
    /// Users receive instances of this struct in their hook handlers.
    pub fn new(
        tool_name: String,
        tool_input: Value,
        tool_use_id: String,
        history: Vec<Value>,
    ) -> Self {
        Self {
            tool_name,
            tool_input,
            tool_use_id,
            history,
        }
    }
}

/// Snapshot passed to an automatic post-tool hook.
///
/// `tool_input` reflects any pre-hook replacement. `history` includes the pending,
/// unmodified result; a post-hook replacement affects the result stored afterward.
#[derive(Debug, Clone)]
pub struct PostToolUseEvent {
    /// Name of the tool that was executed
    pub tool_name: String,
    /// Input parameters that were actually used (may differ from original if modified by PreToolUse)
    pub tool_input: Value,
    /// Unique identifier for this tool use (correlates with PreToolUseEvent)
    pub tool_use_id: String,
    /// Result returned by the tool - may contain "content" on success or "error" on failure
    pub tool_result: Value,
    /// Structured JSON snapshot including this tool call and its unmodified result.
    pub history: Vec<Value>,
}

impl PostToolUseEvent {
    /// Creates a new PostToolUseEvent.
    ///
    /// This constructor is typically called by the agent runtime after tool execution,
    /// not by user code. Users receive instances of this struct in their hook handlers.
    pub fn new(
        tool_name: String,
        tool_input: Value,
        tool_use_id: String,
        tool_result: Value,
        history: Vec<Value>,
    ) -> Self {
        Self {
            tool_name,
            tool_input,
            tool_use_id,
            tool_result,
            history,
        }
    }
}

/// Snapshot passed to a prompt hook before the user message is added to history.
///
/// Invoked by [`Client::send`](crate::Client::send), including an empty continuation
/// prompt. [`Client::send_message`](crate::Client::send_message) bypasses this hook.
#[derive(Debug, Clone)]
pub struct UserPromptSubmitEvent {
    /// The user's original input prompt text
    pub prompt: String,
    /// Structured JSON snapshot of history before this prompt is added.
    pub history: Vec<Value>,
}

impl UserPromptSubmitEvent {
    /// Creates a new UserPromptSubmitEvent.
    ///
    /// This constructor is typically called by the agent runtime when processing user input,
    /// not by user code. Users receive instances of this struct in their hook handlers.
    pub fn new(prompt: String, history: Vec<Value>) -> Self {
        Self { prompt, history }
    }
}
