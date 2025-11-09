//! Lifecycle Hooks System for Agent Execution Control
//!
//! This module provides a powerful hooks system for intercepting, monitoring, and controlling
//! agent behavior at critical lifecycle points. Hooks enable you to implement security gates,
//! audit logging, input validation, output filtering, and dynamic behavior modification without
//! modifying the core agent logic.
//!
//! # Overview
//!
//! The hooks system operates on an event-driven model with three key interception points:
//!
//! 1. **PreToolUse**: Fired before any tool is executed, allowing you to:
//!    - Block dangerous operations (security gates)
//!    - Modify tool inputs (parameter injection, sanitization)
//!    - Log tool usage for auditing
//!    - Implement rate limiting or quotas
//!
//! 2. **PostToolUse**: Fired after tool execution completes, allowing you to:
//!    - Audit tool results
//!    - Filter or redact sensitive information in outputs
//!    - Collect metrics and telemetry
//!    - Validate tool behavior
//!
//! 3. **UserPromptSubmit**: Fired before processing user input, allowing you to:
//!    - Filter inappropriate content
//!    - Modify prompts (add context, instructions)
//!    - Implement content moderation
//!    - Track user interactions
//!
//! # Execution Model
//!
//! Hooks follow a **sequential "first non-None wins"** execution model:
//! - Hooks are executed in the order they were registered
//! - Each hook can return `None` (pass-through) or `Some(HookDecision)` (take control)
//! - The **first hook** that returns `Some(HookDecision)` determines the outcome
//! - Subsequent hooks are **not executed** after a decision is made
//! - If all hooks return `None`, execution continues normally
//!
//! This model ensures predictable behavior and allows you to create hook chains where
//! earlier hooks can implement critical security checks that later hooks cannot override.
//!
//! # Common Use Cases
//!
//! ## Security Gate (Block Dangerous Operations)
//!
//! ```rust,no_run
//! use open_agent::{Hooks, PreToolUseEvent, HookDecision};
//!
//! let hooks = Hooks::new().add_pre_tool_use(|event| async move {
//!     // Block file deletion in production
//!     if event.tool_name == "delete_file" {
//!         return Some(HookDecision::block("File deletion not allowed"));
//!     }
//!     None // Allow other operations
//! });
//! ```
//!
//! ## Audit Logging
//!
//! ```rust,no_run
//! use open_agent::{Hooks, PostToolUseEvent, HookDecision};
//!
//! let hooks = Hooks::new().add_post_tool_use(|event| async move {
//!     // Log all tool executions for compliance
//!     println!("Tool '{}' executed with result: {:?}",
//!              event.tool_name, event.tool_result);
//!     None // Don't interfere with execution
//! });
//! ```
//!
//! ## Input Modification (Parameter Injection)
//!
//! ```rust,no_run
//! use open_agent::{Hooks, PreToolUseEvent, HookDecision};
//! use serde_json::json;
//!
//! let hooks = Hooks::new().add_pre_tool_use(|event| async move {
//!     if event.tool_name == "query_database" {
//!         // Inject security context into all database queries
//!         let mut input = event.tool_input.clone();
//!         input["user_id"] = json!("current_user_123");
//!         return Some(HookDecision::modify_input(input, "Injected user context"));
//!     }
//!     None
//! });
//! ```
//!
//! ## Content Moderation
//!
//! ```rust,no_run
//! use open_agent::{Hooks, UserPromptSubmitEvent, HookDecision};
//!
//! let hooks = Hooks::new().add_user_prompt_submit(|event| async move {
//!     if event.prompt.contains("inappropriate_content") {
//!         return Some(HookDecision::block("Content policy violation"));
//!     }
//!     None
//! });
//! ```
//!
//! ## Dynamic Prompt Enhancement
//!
//! ```ignore
//! use open_agent::{Hooks, UserPromptSubmitEvent, HookDecision};
//!
//! let hooks = Hooks::new().add_user_prompt_submit(|event| async move {
//!     // Add context to user prompts
//!     let enhanced = format!(
//!         "{}\n\nAdditional Context: Current time is {}",
//!         event.prompt,
//!         chrono::Utc::now()
//!     );
//!     Some(HookDecision::modify_prompt(enhanced, "Added timestamp context"))
//! });
//! ```
//!
//! # Thread Safety and Async
//!
//! All hooks are async functions wrapped in `Arc` to enable:
//! - **Thread-safe sharing** across multiple agent instances
//! - **Async operations** like database queries, API calls, or file I/O
//! - **Zero-cost cloning** when passing hooks between threads
//!
//! Hooks can safely perform I/O operations, make network requests, or access shared state
//! as long as that state is thread-safe (e.g., wrapped in `Arc<Mutex<T>>`).
//!
//! # Error Handling
//!
//! If a hook panics or returns an error, the entire agent operation will be aborted.
//! Design your hooks to be robust and handle errors gracefully within the hook itself:
//!
//! ```rust,no_run
//! use open_agent::{Hooks, PreToolUseEvent, HookDecision};
//!
//! let hooks = Hooks::new().add_pre_tool_use(|event| async move {
//!     match risky_validation(&event).await {
//!         Ok(is_valid) => {
//!             if !is_valid {
//!                 Some(HookDecision::block("Validation failed"))
//!             } else {
//!                 None
//!             }
//!         }
//!         Err(e) => {
//!             eprintln!("Hook validation error: {}", e);
//!             // Fail safe: block on errors
//!             Some(HookDecision::block(format!("Validation error: {}", e)))
//!         }
//!     }
//! });
//!
//! async fn risky_validation(_event: &PreToolUseEvent) -> Result<bool, String> {
//!     // Your validation logic here
//!     Ok(true)
//! }
//! ```

use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

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
    /// Snapshot of conversation history (read-only) - useful for context-aware decisions
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
    /// Snapshot of conversation history (read-only) including this tool execution
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
    /// Snapshot of conversation history (read-only) - does not include this prompt yet
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

/// Decision returned by a hook handler to control agent execution flow.
///
/// When a hook returns `Some(HookDecision)`, it takes control of the execution flow.
/// This struct determines whether execution should continue, whether inputs/prompts should
/// be modified, and provides a reason for logging and debugging.
///
/// # "First Non-None Wins" Model
///
/// The hooks system uses a **sequential "first non-None wins"** execution model:
///
/// 1. Hooks are executed in the order they were registered
/// 2. Each hook returns `Option<HookDecision>`:
///    - `None` = "I don't care, let the next hook decide"
///    - `Some(decision)` = "I'm taking control, stop checking other hooks"
/// 3. The **first** hook that returns `Some(decision)` determines the outcome
/// 4. Remaining hooks are **skipped** after a decision is made
/// 5. If **all** hooks return `None`, execution continues normally
///
/// This model ensures:
/// - Predictable behavior (order matters)
/// - Performance (no unnecessary hook executions)
/// - Priority (earlier hooks can't be overridden by later ones)
///
/// # Fields
///
/// - `continue_execution`: If `false`, abort the current operation (tool execution or prompt processing)
/// - `modified_input`: For PreToolUse hooks - replaces the tool input with this value
/// - `modified_prompt`: For UserPromptSubmit hooks - replaces the user prompt with this value
/// - `reason`: Optional explanation for why this decision was made (useful for debugging/logging)
///
/// # Example: Hook Priority Order
///
/// ```rust
/// use open_agent::{Hooks, PreToolUseEvent, HookDecision};
///
/// let hooks = Hooks::new()
///     // First hook - security gate (highest priority)
///     .add_pre_tool_use(|event| async move {
///         if event.tool_name == "dangerous_tool" {
///             // This blocks execution - later hooks won't run
///             return Some(HookDecision::block("Blocked by security"));
///         }
///         None // Pass to next hook
///     })
///     // Second hook - rate limiting
///     .add_pre_tool_use(|event| async move {
///         // This only runs if first hook returned None
///         if over_rate_limit(&event) {
///             return Some(HookDecision::block("Rate limit exceeded"));
///         }
///         None
///     })
///     // Third hook - logging
///     .add_pre_tool_use(|event| async move {
///         // This only runs if previous hooks returned None
///         println!("Tool {} called", event.tool_name);
///         None // Always pass through
///     });
///
/// fn over_rate_limit(_event: &PreToolUseEvent) -> bool { false }
/// ```
///
/// # Builder Methods
///
/// The struct provides convenient builder methods for common scenarios:
///
/// - `HookDecision::continue_()` - Allow execution to proceed normally
/// - `HookDecision::block(reason)` - Block execution with a reason
/// - `HookDecision::modify_input(input, reason)` - Continue with modified tool input
/// - `HookDecision::modify_prompt(prompt, reason)` - Continue with modified user prompt
#[derive(Debug, Clone, Default)]
pub struct HookDecision {
    /// Whether to continue execution. If `false`, the operation is aborted.
    /// Default: `false` (via Default trait), but builder methods set this appropriately.
    continue_execution: bool,

    /// For PreToolUse hooks: If set, replaces the original tool input with this value.
    /// The tool will execute with this modified input instead of the original.
    modified_input: Option<Value>,

    /// For UserPromptSubmit hooks: If set, replaces the user's prompt with this value.
    /// The agent will process this modified prompt instead of the original.
    modified_prompt: Option<String>,

    /// Optional human-readable explanation for why this decision was made.
    /// Useful for logging, debugging, and audit trails.
    reason: Option<String>,
}

impl HookDecision {
    /// Creates a decision to continue execution normally without modifications.
    ///
    /// This is typically used when a hook wants to explicitly signal "continue" rather
    /// than returning `None`. In most cases, returning `None` is simpler and preferred.
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::{PreToolUseEvent, HookDecision};
    ///
    /// async fn my_hook(event: PreToolUseEvent) -> Option<HookDecision> {
    ///     // Log the tool use
    ///     println!("Tool called: {}", event.tool_name);
    ///
    ///     // Explicitly continue (though returning None would be simpler)
    ///     Some(HookDecision::continue_())
    /// }
    /// ```
    ///
    /// Note: Named `continue_()` with trailing underscore because `continue` is a Rust keyword.
    pub fn continue_() -> Self {
        Self {
            continue_execution: true,
            modified_input: None,
            modified_prompt: None,
            reason: None,
        }
    }

    /// Creates a decision to block execution with a reason.
    ///
    /// When a hook returns this decision, the current operation (tool execution or
    /// prompt processing) is aborted, and the reason is logged.
    ///
    /// # Parameters
    ///
    /// - `reason`: Human-readable explanation for why execution was blocked
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::{PreToolUseEvent, HookDecision};
    ///
    /// async fn security_gate(event: PreToolUseEvent) -> Option<HookDecision> {
    ///     if event.tool_name == "Bash" {
    ///         if let Some(cmd) = event.tool_input.get("command") {
    ///             if cmd.as_str()?.contains("rm -rf /") {
    ///                 return Some(HookDecision::block(
    ///                     "Dangerous recursive delete blocked"
    ///                 ));
    ///             }
    ///         }
    ///     }
    ///     None
    /// }
    /// ```
    pub fn block(reason: impl Into<String>) -> Self {
        Self {
            continue_execution: false,
            modified_input: None,
            modified_prompt: None,
            reason: Some(reason.into()),
        }
    }

    /// Creates a decision to modify tool input before execution.
    ///
    /// Use this in PreToolUse hooks to change the parameters that will be passed to the tool.
    /// The tool will execute with the modified input instead of the original.
    ///
    /// # Parameters
    ///
    /// - `input`: The new tool input (as JSON Value) that replaces the original
    /// - `reason`: Explanation for why the input was modified
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::{PreToolUseEvent, HookDecision};
    /// use serde_json::json;
    ///
    /// async fn inject_security_token(event: PreToolUseEvent) -> Option<HookDecision> {
    ///     if event.tool_name == "WebFetch" {
    ///         // Add authentication to all web requests
    ///         let mut modified = event.tool_input.clone();
    ///         modified["headers"] = json!({
    ///             "Authorization": "Bearer secret-token",
    ///             "X-User-ID": "user-123"
    ///         });
    ///
    ///         return Some(HookDecision::modify_input(
    ///             modified,
    ///             "Injected authentication headers"
    ///         ));
    ///     }
    ///     None
    /// }
    /// ```
    pub fn modify_input(input: Value, reason: impl Into<String>) -> Self {
        Self {
            continue_execution: true,
            modified_input: Some(input),
            modified_prompt: None,
            reason: Some(reason.into()),
        }
    }

    /// Creates a decision to modify the user's prompt before processing.
    ///
    /// Use this in UserPromptSubmit hooks to enhance, sanitize, or transform user input.
    /// The agent will process the modified prompt instead of the original.
    ///
    /// # Parameters
    ///
    /// - `prompt`: The new prompt text that replaces the user's original input
    /// - `reason`: Explanation for why the prompt was modified
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::{UserPromptSubmitEvent, HookDecision};
    ///
    /// async fn add_context(event: UserPromptSubmitEvent) -> Option<HookDecision> {
    ///     // Add system context to every user prompt
    ///     let enhanced = format!(
    ///         "{}\n\n[System Context: You are in production mode. Be extra careful with destructive operations.]",
    ///         event.prompt
    ///     );
    ///
    ///     Some(HookDecision::modify_prompt(
    ///         enhanced,
    ///         "Added production safety context"
    ///     ))
    /// }
    /// ```
    ///
    /// # Warning
    ///
    /// Modifying prompts can be confusing for users if done excessively or without clear
    /// communication. Use this feature judiciously and consider logging modifications.
    pub fn modify_prompt(prompt: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            continue_execution: true,
            modified_input: None,
            modified_prompt: Some(prompt.into()),
            reason: Some(reason.into()),
        }
    }

    /// Returns whether execution should continue.
    pub fn continue_execution(&self) -> bool {
        self.continue_execution
    }

    /// Returns the modified input, if any.
    pub fn modified_input(&self) -> Option<&Value> {
        self.modified_input.as_ref()
    }

    /// Returns the modified prompt, if any.
    pub fn modified_prompt(&self) -> Option<&str> {
        self.modified_prompt.as_deref()
    }

    /// Returns the reason, if any.
    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }
}

/// Type alias for PreToolUse hook handler functions.
///
/// This complex type signature enables powerful async hook functionality while maintaining
/// thread safety and zero-cost abstraction. Let's break it down:
///
/// # Type Breakdown
///
/// ```text
/// Arc<                                  // Reference counting for thread-safe sharing
///     dyn Fn(PreToolUseEvent)           // Function taking the event
///         -> Pin<Box<                    // Heap-allocated, pinned future
///             dyn Future<Output = Option<HookDecision>>  // Async result
///                 + Send                 // Can be sent across threads
///         >>
///         + Send + Sync                  // The function itself is thread-safe
/// >
/// ```
///
/// # Why This Design?
///
/// - **`Arc`**: Enables zero-cost cloning when passing hooks between threads or agent instances.
///   Multiple agents can share the same hook without duplicating memory.
///
/// - **`dyn Fn`**: Allows any function or closure to be used as a hook, as long as it matches
///   the signature. This is trait object type erasure.
///
/// - **`Pin<Box<dyn Future>>`**: Async functions in Rust return opaque Future types. We need
///   to box them for dynamic dispatch and pin them because futures may contain self-references.
///
/// - **`Send + Sync`**: Ensures the hook can be safely called from multiple threads. Essential
///   for async runtimes like Tokio that may schedule tasks on different threads.
///
/// # Return Value
///
/// Hook handlers return `Option<HookDecision>`:
/// - `None`: "I don't care, continue normally or let next hook decide"
/// - `Some(HookDecision)`: "I'm taking control" - blocks remaining hooks from running
///
/// # Example Usage
///
/// You don't typically construct these types directly. Instead, use the builder methods:
///
/// ```rust
/// use open_agent::{Hooks, PreToolUseEvent, HookDecision};
///
/// let hooks = Hooks::new().add_pre_tool_use(|event| async move {
///     // Your async logic here
///     if event.tool_name == "dangerous" {
///         Some(HookDecision::block("Not allowed"))
///     } else {
///         None
///     }
/// });
/// ```
///
/// The builder automatically wraps your closure in `Arc<...>` and handles the `Pin<Box<...>>`.
pub type PreToolUseHandler = Arc<
    dyn Fn(PreToolUseEvent) -> Pin<Box<dyn Future<Output = Option<HookDecision>> + Send>>
        + Send
        + Sync,
>;

/// Type alias for PostToolUse hook handler functions.
///
/// Identical in structure to `PreToolUseHandler` but receives `PostToolUseEvent` instead.
/// See [`PreToolUseHandler`] for detailed explanation of the type signature.
///
/// # Common Usage Pattern
///
/// PostToolUse hooks typically don't modify execution (they return `None`) but are used
/// for observation, logging, and metrics:
///
/// ```rust
/// use open_agent::{Hooks, PostToolUseEvent, HookDecision};
///
/// let hooks = Hooks::new().add_post_tool_use(|event| async move {
///     // Log tool execution for audit trail
///     println!("Tool {} completed with result: {:?}",
///              event.tool_name, event.tool_result);
///
///     // Send metrics to monitoring system
///     // metrics::counter!("tool_executions", 1, "tool" => event.tool_name);
///
///     None // Don't interfere with execution
/// });
/// ```
pub type PostToolUseHandler = Arc<
    dyn Fn(PostToolUseEvent) -> Pin<Box<dyn Future<Output = Option<HookDecision>> + Send>>
        + Send
        + Sync,
>;

/// Type alias for UserPromptSubmit hook handler functions.
///
/// Identical in structure to `PreToolUseHandler` but receives `UserPromptSubmitEvent` instead.
/// See [`PreToolUseHandler`] for detailed explanation of the type signature.
///
/// # Common Usage Pattern
///
/// UserPromptSubmit hooks are often used for content moderation and prompt enhancement:
///
/// ```rust
/// use open_agent::{Hooks, UserPromptSubmitEvent, HookDecision};
///
/// let hooks = Hooks::new().add_user_prompt_submit(|event| async move {
///     // Block inappropriate content
///     if event.prompt.to_lowercase().contains("banned_word") {
///         return Some(HookDecision::block("Content policy violation"));
///     }
///
///     // Or enhance prompts with context
///     let enhanced = format!("{}\n\nContext: Session ID 12345", event.prompt);
///     Some(HookDecision::modify_prompt(enhanced, "Added session context"))
/// });
/// ```
pub type UserPromptSubmitHandler = Arc<
    dyn Fn(UserPromptSubmitEvent) -> Pin<Box<dyn Future<Output = Option<HookDecision>> + Send>>
        + Send
        + Sync,
>;

/// Container for registering and managing lifecycle hooks.
///
/// The `Hooks` struct stores collections of hook handlers for different lifecycle events.
/// It provides a builder pattern for registering hooks and executor methods for running them.
///
/// # Design Principles
///
/// - **Builder Pattern**: Hooks can be chained during construction using `.add_*()` methods
/// - **Multiple Hooks**: You can register multiple hooks for the same event type
/// - **Execution Order**: Hooks execute in the order they were registered (FIFO)
/// - **First Wins**: The first hook returning `Some(HookDecision)` determines the outcome
/// - **Thread Safe**: The struct is `Clone` and all handlers are `Arc`-wrapped for sharing
///
/// # Example: Building a Hooks Collection
///
/// ```rust
/// use open_agent::{Hooks, PreToolUseEvent, PostToolUseEvent, HookDecision};
///
/// let hooks = Hooks::new()
///     // First: Security gate (highest priority)
///     .add_pre_tool_use(|event| async move {
///         if event.tool_name == "dangerous" {
///             return Some(HookDecision::block("Security violation"));
///         }
///         None
///     })
///     // Second: Rate limiting
///     .add_pre_tool_use(|event| async move {
///         // Check rate limits...
///         None
///     })
///     // Audit logging (happens after execution)
///     .add_post_tool_use(|event| async move {
///         println!("Tool '{}' executed", event.tool_name);
///         None
///     });
/// ```
///
/// # Fields
///
/// - `pre_tool_use`: Handlers invoked before tool execution
/// - `post_tool_use`: Handlers invoked after tool execution
/// - `user_prompt_submit`: Handlers invoked before processing user prompts
///
/// All fields are public, allowing direct manipulation if needed, though the builder
/// methods are the recommended approach.
#[derive(Clone, Default)]
pub struct Hooks {
    /// Collection of PreToolUse hook handlers, executed in registration order
    pub pre_tool_use: Vec<PreToolUseHandler>,

    /// Collection of PostToolUse hook handlers, executed in registration order
    pub post_tool_use: Vec<PostToolUseHandler>,

    /// Collection of UserPromptSubmit hook handlers, executed in registration order
    pub user_prompt_submit: Vec<UserPromptSubmitHandler>,
}

impl Hooks {
    /// Creates a new, empty `Hooks` container.
    ///
    /// Use this as the starting point for building a hooks collection using the builder pattern.
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Hooks;
    ///
    /// let hooks = Hooks::new()
    ///     .add_pre_tool_use(|event| async move { None });
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a PreToolUse hook handler using the builder pattern.
    ///
    /// This method takes ownership of `self` and returns it back, allowing method chaining.
    /// The handler is wrapped in `Arc` and added to the collection of PreToolUse hooks.
    ///
    /// # Parameters
    ///
    /// - `handler`: An async function or closure that takes `PreToolUseEvent` and returns
    ///   `Option<HookDecision>`. Must be `Send + Sync + 'static` for thread safety.
    ///
    /// # Type Parameters
    ///
    /// - `F`: The function/closure type
    /// - `Fut`: The future type returned by the function
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::{Hooks, HookDecision};
    ///
    /// let hooks = Hooks::new()
    ///     .add_pre_tool_use(|event| async move {
    ///         println!("About to execute: {}", event.tool_name);
    ///         None
    ///     })
    ///     .add_pre_tool_use(|event| async move {
    ///         // This runs second (if first returns None)
    ///         if event.tool_name == "blocked" {
    ///             Some(HookDecision::block("Not allowed"))
    ///         } else {
    ///             None
    ///         }
    ///     });
    /// ```
    pub fn add_pre_tool_use<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(PreToolUseEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<HookDecision>> + Send + 'static,
    {
        // Wrap the user's function in Arc and Box::pin for type erasure and heap allocation
        self.pre_tool_use
            .push(Arc::new(move |event| Box::pin(handler(event))));
        self
    }

    /// Registers a PostToolUse hook handler using the builder pattern.
    ///
    /// Identical to `add_pre_tool_use` but for PostToolUse events. See [`Self::add_pre_tool_use`]
    /// for detailed documentation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::Hooks;
    ///
    /// let hooks = Hooks::new()
    ///     .add_post_tool_use(|event| async move {
    ///         // Audit log all tool executions
    ///         println!("Tool '{}' completed: {:?}",
    ///                  event.tool_name, event.tool_result);
    ///         None // Don't interfere with execution
    ///     });
    /// ```
    pub fn add_post_tool_use<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(PostToolUseEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<HookDecision>> + Send + 'static,
    {
        // Wrap the user's function in Arc and Box::pin for type erasure and heap allocation
        self.post_tool_use
            .push(Arc::new(move |event| Box::pin(handler(event))));
        self
    }

    /// Registers a UserPromptSubmit hook handler using the builder pattern.
    ///
    /// Identical to `add_pre_tool_use` but for UserPromptSubmit events. See [`Self::add_pre_tool_use`]
    /// for detailed documentation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::{Hooks, HookDecision};
    ///
    /// let hooks = Hooks::new()
    ///     .add_user_prompt_submit(|event| async move {
    ///         // Content moderation
    ///         if event.prompt.contains("forbidden") {
    ///             Some(HookDecision::block("Content violation"))
    ///         } else {
    ///             None
    ///         }
    ///     });
    /// ```
    pub fn add_user_prompt_submit<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(UserPromptSubmitEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<HookDecision>> + Send + 'static,
    {
        // Wrap the user's function in Arc and Box::pin for type erasure and heap allocation
        self.user_prompt_submit
            .push(Arc::new(move |event| Box::pin(handler(event))));
        self
    }

    /// Executes all registered PreToolUse hooks in order and returns the first decision.
    ///
    /// This method implements the **"first non-None wins"** execution model:
    ///
    /// 1. Iterates through hooks in registration order (FIFO)
    /// 2. Calls each hook with a clone of the event
    /// 3. If a hook returns `Some(decision)`, immediately returns that decision
    /// 4. Remaining hooks are **not executed**
    /// 5. If all hooks return `None`, returns `None`
    ///
    /// # Parameters
    ///
    /// - `event`: The PreToolUseEvent to pass to each hook
    ///
    /// # Returns
    ///
    /// - `Some(HookDecision)`: A hook made a decision (block, modify, or continue)
    /// - `None`: All hooks returned `None` (continue normally)
    ///
    /// # Example
    ///
    /// ```rust
    /// use open_agent::{Hooks, PreToolUseEvent, HookDecision};
    /// use serde_json::json;
    ///
    /// # async fn example() {
    /// let hooks = Hooks::new()
    ///     .add_pre_tool_use(|e| async move { None }) // Runs first
    ///     .add_pre_tool_use(|e| async move {
    ///         Some(HookDecision::block("Blocked")) // Runs second, blocks
    ///     })
    ///     .add_pre_tool_use(|e| async move {
    ///         None // NEVER runs because previous hook returned Some
    ///     });
    ///
    /// let event = PreToolUseEvent::new(
    ///     "test".to_string(),
    ///     json!({}),
    ///     "id".to_string(),
    ///     vec![]
    /// );
    ///
    /// let decision = hooks.execute_pre_tool_use(event).await;
    /// assert!(decision.is_some());
    /// assert!(!decision.unwrap().continue_execution());
    /// # }
    /// ```
    pub async fn execute_pre_tool_use(&self, event: PreToolUseEvent) -> Option<HookDecision> {
        // Sequential execution: iterate through handlers in order
        for handler in &self.pre_tool_use {
            // Clone the event for this handler (events are cheaply cloneable)
            let decision = handler(event.clone()).await;

            // First non-None wins: return immediately if this hook made a decision
            if decision.is_some() {
                return decision;
            }
            // Otherwise, continue to next hook
        }

        // All hooks returned None: no decision made, continue normally
        None
    }

    /// Executes all registered PostToolUse hooks in order and returns the first decision.
    ///
    /// Identical in behavior to [`Self::execute_pre_tool_use`] but for PostToolUse events.
    /// See that method for detailed documentation of the execution model.
    ///
    /// # Note
    ///
    /// PostToolUse hooks rarely return decisions in practice. They're primarily used for
    /// observation (logging, metrics) and typically always return `None`.
    pub async fn execute_post_tool_use(&self, event: PostToolUseEvent) -> Option<HookDecision> {
        // Sequential execution with "first non-None wins" model
        for handler in &self.post_tool_use {
            let decision = handler(event.clone()).await;
            if decision.is_some() {
                return decision;
            }
        }
        None
    }

    /// Executes all registered UserPromptSubmit hooks in order and returns the first decision.
    ///
    /// Identical in behavior to [`Self::execute_pre_tool_use`] but for UserPromptSubmit events.
    /// See that method for detailed documentation of the execution model.
    pub async fn execute_user_prompt_submit(
        &self,
        event: UserPromptSubmitEvent,
    ) -> Option<HookDecision> {
        // Sequential execution with "first non-None wins" model
        for handler in &self.user_prompt_submit {
            let decision = handler(event.clone()).await;
            if decision.is_some() {
                return decision;
            }
        }
        None
    }
}

/// Custom Debug implementation for Hooks.
///
/// Since hook handlers are closures (which don't implement Debug), we provide a custom
/// implementation that shows the number of registered handlers instead of trying to
/// debug-print the closures themselves.
///
/// # Example Output
///
/// ```text
/// Hooks {
///     pre_tool_use: 3 handlers,
///     post_tool_use: 1 handlers,
///     user_prompt_submit: 2 handlers
/// }
/// ```
impl std::fmt::Debug for Hooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hooks")
            .field(
                "pre_tool_use",
                &format!("{} handlers", self.pre_tool_use.len()),
            )
            .field(
                "post_tool_use",
                &format!("{} handlers", self.post_tool_use.len()),
            )
            .field(
                "user_prompt_submit",
                &format!("{} handlers", self.user_prompt_submit.len()),
            )
            .finish()
    }
}

/// String constant for the PreToolUse hook event name.
///
/// This constant can be used for logging, metrics, or when you need a string
/// representation of the hook type. It's primarily used internally but is exposed
/// as part of the public API for consistency.
pub const HOOK_PRE_TOOL_USE: &str = "pre_tool_use";

/// String constant for the PostToolUse hook event name.
///
/// See [`HOOK_PRE_TOOL_USE`] for usage details.
pub const HOOK_POST_TOOL_USE: &str = "post_tool_use";

/// String constant for the UserPromptSubmit hook event name.
///
/// See [`HOOK_PRE_TOOL_USE`] for usage details.
pub const HOOK_USER_PROMPT_SUBMIT: &str = "user_prompt_submit";

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_hook_decision_builders() {
        let continue_dec = HookDecision::continue_();
        assert!(continue_dec.continue_execution);
        assert!(continue_dec.reason.is_none());

        let block_dec = HookDecision::block("test");
        assert!(!block_dec.continue_execution);
        assert_eq!(block_dec.reason, Some("test".to_string()));

        let modify_dec = HookDecision::modify_input(json!({"test": 1}), "modified");
        assert!(modify_dec.continue_execution);
        assert!(modify_dec.modified_input.is_some());
    }

    #[tokio::test]
    async fn test_pre_tool_use_hook() {
        let hooks = Hooks::new().add_pre_tool_use(|event| async move {
            if event.tool_name == "dangerous" {
                return Some(HookDecision::block("blocked"));
            }
            None
        });

        let event = PreToolUseEvent::new(
            "dangerous".to_string(),
            json!({}),
            "id1".to_string(),
            vec![],
        );

        let decision = hooks.execute_pre_tool_use(event).await;
        assert!(decision.is_some());
        assert!(!decision.unwrap().continue_execution);
    }

    #[tokio::test]
    async fn test_post_tool_use_hook() {
        let hooks = Hooks::new().add_post_tool_use(|_event| async move { None });

        let event = PostToolUseEvent::new(
            "test".to_string(),
            json!({}),
            "id1".to_string(),
            json!({"result": "ok"}),
            vec![],
        );

        // Should not panic
        hooks.execute_post_tool_use(event).await;
    }

    #[tokio::test]
    async fn test_user_prompt_submit_hook() {
        let hooks = Hooks::new().add_user_prompt_submit(|event| async move {
            if event.prompt.contains("DELETE") {
                return Some(HookDecision::block("dangerous prompt"));
            }
            None
        });

        let event = UserPromptSubmitEvent::new("DELETE all files".to_string(), vec![]);

        let decision = hooks.execute_user_prompt_submit(event).await;
        assert!(decision.is_some());
        assert!(!decision.unwrap().continue_execution);
    }
}
