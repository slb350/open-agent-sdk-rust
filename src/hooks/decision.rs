/// A hook decision to continue, block, or replace input.
///
/// Returning `Some(decision)` stops the remaining hooks for that event, even when
/// the decision is [`continue_`](Self::continue_). Return `None` to let the next
/// hook decide. `Default` blocks execution with no reason or replacement.
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
    /// Continues the operation without modifications, skipping subsequent hooks.
    ///
    /// Return `None` from the hook instead when later hooks should also run.
    pub fn continue_() -> Self {
        Self {
            continue_execution: true,
            modified_input: None,
            modified_prompt: None,
            reason: None,
        }
    }

    /// Blocks the prompt or tool call with the supplied reason.
    ///
    /// A blocked prompt becomes an error; a blocked automatic tool call becomes an
    /// error result sent back to the model.
    pub fn block(reason: impl Into<String>) -> Self {
        Self {
            continue_execution: false,
            modified_input: None,
            modified_prompt: None,
            reason: Some(reason.into()),
        }
    }

    /// Continues with replacement tool input.
    ///
    /// In a post-tool hook, the same field replaces the tool result instead.
    pub fn modify_input(input: Value, reason: impl Into<String>) -> Self {
        Self {
            continue_execution: true,
            modified_input: Some(input),
            modified_prompt: None,
            reason: Some(reason.into()),
        }
    }

    /// Continues with replacement prompt text in a user-prompt-submit hook.
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
