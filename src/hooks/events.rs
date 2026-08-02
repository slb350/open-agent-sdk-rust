/// Event fired **before** a tool is executed, enabling validation, modification, or blocking.
///
/// This event provides complete visibility into the tool that's about to be executed,
/// allowing you to implement security policies, modify inputs, or collect telemetry
/// before any potentially dangerous or expensive operations occur.
///
/// # Use Cases
///
/// - **Security gates**: Block dangerous operations (file deletion, network access)
/// - **Input validation**: Ensure tool inputs meet schema or business rules
/// - **Parameter injection**: Add authentication tokens, user context, or default values
/// - **Rate limiting**: Track and limit tool usage per user/session
/// - **Audit logging**: Record who is calling what tools with what parameters
///
/// # Fields
///
/// - `tool_name`: The name of the tool about to execute (e.g., "Bash", "Read", "WebFetch")
/// - `tool_input`: The parameters that will be passed to the tool (as JSON)
/// - `tool_use_id`: Unique identifier for this specific tool invocation
/// - `history`: Read-only snapshot of the conversation history up to this point
///
/// # Example: Security Gate
///
/// ```rust
/// use open_agent::{PreToolUseEvent, HookDecision};
/// use serde_json::json;
///
/// async fn security_gate(event: PreToolUseEvent) -> Option<HookDecision> {
///     // Block all Bash commands containing 'rm -rf'
///     if event.tool_name == "Bash" {
///         if let Some(command) = event.tool_input.get("command") {
///             if command.as_str()?.contains("rm -rf") {
///                 return Some(HookDecision::block(
///                     "Dangerous command blocked for safety"
///                 ));
///             }
///         }
///     }
///     None // Allow other tools
/// }
/// ```
///
/// # Example: Parameter Injection
///
/// ```rust
/// use open_agent::{PreToolUseEvent, HookDecision};
/// use serde_json::json;
///
/// async fn inject_auth(event: PreToolUseEvent) -> Option<HookDecision> {
///     // Add authentication header to all API calls
///     if event.tool_name == "WebFetch" {
///         let mut modified = event.tool_input.clone();
///         modified["headers"] = json!({
///             "Authorization": "Bearer secret-token"
///         });
///         return Some(HookDecision::modify_input(
///             modified,
///             "Injected auth token"
///         ));
///     }
///     None
/// }
/// ```
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

/// Event fired **after** a tool completes execution, enabling audit, filtering, or validation.
///
/// This event provides complete visibility into what a tool did, including both the input
/// parameters and the output result. Use this for auditing, metrics collection, output
/// filtering, or post-execution validation.
///
/// # Use Cases
///
/// - **Audit logging**: Record all tool executions with inputs and outputs for compliance
/// - **Output filtering**: Redact sensitive information from tool results
/// - **Metrics collection**: Track tool performance, success rates, error patterns
/// - **Result validation**: Ensure tool outputs meet quality or safety standards
/// - **Error handling**: Implement custom error recovery or alerting
///
/// # Fields
///
/// - `tool_name`: The name of the tool that was executed
/// - `tool_input`: The parameters that were actually used (may have been modified by PreToolUse hooks)
/// - `tool_use_id`: Unique identifier for this invocation (matches PreToolUseEvent.tool_use_id)
/// - `tool_result`: The result returned by the tool (contains either success data or error info)
/// - `history`: Read-only snapshot of conversation history including this tool's execution
///
/// # Example: Audit Logging
///
/// ```rust
/// use open_agent::{PostToolUseEvent, HookDecision};
///
/// async fn audit_logger(event: PostToolUseEvent) -> Option<HookDecision> {
///     // Log all tool executions to your audit system
///     let is_error = event.tool_result.get("error").is_some();
///
///     println!(
///         "[AUDIT] Tool: {}, ID: {}, Status: {}",
///         event.tool_name,
///         event.tool_use_id,
///         if is_error { "ERROR" } else { "SUCCESS" }
///     );
///
///     // Send to external logging service
///     // log_to_service(&event).await;
///
///     None // Don't interfere with execution
/// }
/// ```
///
/// # Example: Sensitive Data Redaction
///
/// ```rust
/// use open_agent::{PostToolUseEvent, HookDecision};
/// use serde_json::json;
///
/// async fn redact_secrets(event: PostToolUseEvent) -> Option<HookDecision> {
///     // Redact API keys from Read tool output
///     if event.tool_name == "Read" {
///         if let Some(content) = event.tool_result.get("content") {
///             if let Some(text) = content.as_str() {
///                 if text.contains("API_KEY=") {
///                     let redacted = text.replace(
///                         |c: char| c.is_alphanumeric(),
///                         "*"
///                     );
///                     // Note: PostToolUse hooks typically don't modify results,
///                     // but you could log this for security review
///                     println!("Warning: Potential API key detected in output");
///                 }
///             }
///         }
///     }
///     None
/// }
/// ```
///
/// # Note on Modification
///
/// While `HookDecision` theoretically allows modification in PostToolUse hooks, this is
/// rarely used in practice. The tool has already executed, and most agents don't support
/// modifying historical results. PostToolUse hooks are primarily for observation and auditing.
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

/// Event fired **before** processing user input, enabling content moderation and prompt enhancement.
///
/// This event is triggered whenever a user submits a prompt to the agent, before the agent
/// begins processing it. Use this to implement content moderation, add context, inject
/// instructions, or track user interactions.
///
/// # Use Cases
///
/// - **Content moderation**: Filter inappropriate or harmful user inputs
/// - **Prompt enhancement**: Add system context, timestamps, or user information
/// - **Input validation**: Ensure prompts meet format or length requirements
/// - **Usage tracking**: Log user interactions for analytics or billing
/// - **Context injection**: Add relevant background information to every prompt
///
/// # Fields
///
/// - `prompt`: The user's original input text
/// - `history`: Read-only snapshot of the conversation history before this prompt
///
/// # Example: Content Moderation
///
/// ```rust
/// use open_agent::{UserPromptSubmitEvent, HookDecision};
///
/// async fn content_moderator(event: UserPromptSubmitEvent) -> Option<HookDecision> {
///     // Block prompts containing banned words
///     let banned_words = ["spam", "malware", "hack"];
///
///     for word in banned_words {
///         if event.prompt.to_lowercase().contains(word) {
///             return Some(HookDecision::block(
///                 format!("Content policy violation: contains '{}'", word)
///             ));
///         }
///     }
///     None // Allow clean prompts
/// }
/// ```
///
/// # Example: Automatic Context Enhancement
///
/// ```rust
/// use open_agent::{UserPromptSubmitEvent, HookDecision};
///
/// async fn add_context(event: UserPromptSubmitEvent) -> Option<HookDecision> {
///     // Add helpful context to every user prompt
///     let enhanced = format!(
///         "{}\n\n---\nContext: User timezone is UTC, current session started at 2025-11-07",
///         event.prompt
///     );
///
///     Some(HookDecision::modify_prompt(
///         enhanced,
///         "Added session context"
///     ))
/// }
/// ```
///
/// # Example: Usage Tracking
///
/// ```rust
/// use open_agent::{UserPromptSubmitEvent, HookDecision};
///
/// async fn track_usage(event: UserPromptSubmitEvent) -> Option<HookDecision> {
///     // Log every user interaction for analytics
///     println!(
///         "[ANALYTICS] User submitted prompt of {} characters at history depth {}",
///         event.prompt.len(),
///         event.history.len()
///     );
///
///     // Could also:
///     // - Update usage quotas
///     // - Send to analytics service
///     // - Check rate limits
///
///     None // Don't modify the prompt
/// }
/// ```
///
/// # Modification Behavior
///
/// If you return `HookDecision::modify_prompt()`, the modified prompt completely replaces
/// the original user input before the agent processes it. This is powerful but should be
/// used carefully to avoid confusing the user or the agent.
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
