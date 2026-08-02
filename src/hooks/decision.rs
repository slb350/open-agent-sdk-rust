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
