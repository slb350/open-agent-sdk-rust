//! Hooks system for intercepting and controlling agent execution
//!
//! This module provides a hooks system for monitoring and controlling agent
//! behavior at key lifecycle points.
//!
//! # Examples
//!
//! ```rust,no_run
//! use open_agent::{Client, AgentOptions, PreToolUseEvent, HookDecision};
//!
//! async fn approve_tool(event: PreToolUseEvent) -> Option<HookDecision> {
//!     if event.tool_name == "delete_file" {
//!         return Some(HookDecision {
//!             continue_execution: false,
//!             modified_input: None,
//!             modified_prompt: None,
//!             reason: Some("Dangerous operation blocked".to_string()),
//!         });
//!     }
//!     None // Continue normally
//! }
//! ```

use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Event fired before tool execution
#[derive(Debug, Clone)]
pub struct PreToolUseEvent {
    /// Name of the tool about to be executed
    pub tool_name: String,
    /// Input parameters for the tool
    pub tool_input: Value,
    /// Unique identifier for this tool use
    pub tool_use_id: String,
    /// Snapshot of conversation history (read-only)
    pub history: Vec<Value>,
}

impl PreToolUseEvent {
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

/// Event fired after tool execution
#[derive(Debug, Clone)]
pub struct PostToolUseEvent {
    /// Name of the tool that was executed
    pub tool_name: String,
    /// Input parameters that were used
    pub tool_input: Value,
    /// Unique identifier for this tool use
    pub tool_use_id: String,
    /// Result returned by the tool (success or error)
    pub tool_result: Value,
    /// Snapshot of conversation history (read-only)
    pub history: Vec<Value>,
}

impl PostToolUseEvent {
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

/// Event fired before processing user input
#[derive(Debug, Clone)]
pub struct UserPromptSubmitEvent {
    /// The user's input prompt
    pub prompt: String,
    /// Snapshot of conversation history (read-only)
    pub history: Vec<Value>,
}

impl UserPromptSubmitEvent {
    pub fn new(prompt: String, history: Vec<Value>) -> Self {
        Self { prompt, history }
    }
}

/// Decision returned by hook handler to control execution
#[derive(Debug, Clone, Default)]
pub struct HookDecision {
    /// Whether to continue execution (default: true)
    pub continue_execution: bool,
    /// For PreToolUse - modified tool input (overrides original)
    pub modified_input: Option<Value>,
    /// For UserPromptSubmit - modified prompt (overrides original)
    pub modified_prompt: Option<String>,
    /// Optional explanation for logging/debugging
    pub reason: Option<String>,
}

impl HookDecision {
    /// Create a decision to continue execution normally
    pub fn continue_() -> Self {
        Self {
            continue_execution: true,
            modified_input: None,
            modified_prompt: None,
            reason: None,
        }
    }

    /// Create a decision to block execution
    pub fn block(reason: impl Into<String>) -> Self {
        Self {
            continue_execution: false,
            modified_input: None,
            modified_prompt: None,
            reason: Some(reason.into()),
        }
    }

    /// Create a decision to modify tool input
    pub fn modify_input(input: Value, reason: impl Into<String>) -> Self {
        Self {
            continue_execution: true,
            modified_input: Some(input),
            modified_prompt: None,
            reason: Some(reason.into()),
        }
    }

    /// Create a decision to modify prompt
    pub fn modify_prompt(prompt: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            continue_execution: true,
            modified_input: None,
            modified_prompt: Some(prompt.into()),
            reason: Some(reason.into()),
        }
    }
}

/// Type alias for hook handler functions
///
/// Hook handlers are async functions that take an event and return an optional decision.
///
/// Returns:
/// - `None`: Continue normally with no modifications
/// - `Some(HookDecision)`: Control execution (continue/skip/modify)
///
/// Raising an error will abort execution entirely.
pub type PreToolUseHandler = Arc<
    dyn Fn(PreToolUseEvent) -> Pin<Box<dyn Future<Output = Option<HookDecision>> + Send>>
        + Send
        + Sync,
>;

pub type PostToolUseHandler = Arc<
    dyn Fn(PostToolUseEvent) -> Pin<Box<dyn Future<Output = Option<HookDecision>> + Send>>
        + Send
        + Sync,
>;

pub type UserPromptSubmitHandler = Arc<
    dyn Fn(UserPromptSubmitEvent) -> Pin<Box<dyn Future<Output = Option<HookDecision>> + Send>>
        + Send
        + Sync,
>;

/// Container for all hook types
#[derive(Clone, Default)]
pub struct Hooks {
    pub pre_tool_use: Vec<PreToolUseHandler>,
    pub post_tool_use: Vec<PostToolUseHandler>,
    pub user_prompt_submit: Vec<UserPromptSubmitHandler>,
}

impl Hooks {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a PreToolUse hook
    pub fn add_pre_tool_use<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(PreToolUseEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<HookDecision>> + Send + 'static,
    {
        self.pre_tool_use
            .push(Arc::new(move |event| Box::pin(handler(event))));
        self
    }

    /// Add a PostToolUse hook
    pub fn add_post_tool_use<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(PostToolUseEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<HookDecision>> + Send + 'static,
    {
        self.post_tool_use
            .push(Arc::new(move |event| Box::pin(handler(event))));
        self
    }

    /// Add a UserPromptSubmit hook
    pub fn add_user_prompt_submit<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(UserPromptSubmitEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<HookDecision>> + Send + 'static,
    {
        self.user_prompt_submit
            .push(Arc::new(move |event| Box::pin(handler(event))));
        self
    }

    /// Execute PreToolUse hooks and return the first non-None decision
    pub async fn execute_pre_tool_use(&self, event: PreToolUseEvent) -> Option<HookDecision> {
        for handler in &self.pre_tool_use {
            if let Some(decision) = handler(event.clone()).await {
                return Some(decision);
            }
        }
        None
    }

    /// Execute PostToolUse hooks and return the first non-None decision
    pub async fn execute_post_tool_use(&self, event: PostToolUseEvent) -> Option<HookDecision> {
        for handler in &self.post_tool_use {
            if let Some(decision) = handler(event.clone()).await {
                return Some(decision);
            }
        }
        None
    }

    /// Execute UserPromptSubmit hooks and return the first non-None decision
    pub async fn execute_user_prompt_submit(
        &self,
        event: UserPromptSubmitEvent,
    ) -> Option<HookDecision> {
        for handler in &self.user_prompt_submit {
            if let Some(decision) = handler(event.clone()).await {
                return Some(decision);
            }
        }
        None
    }
}

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

/// Hook event name constants
pub const HOOK_PRE_TOOL_USE: &str = "pre_tool_use";
pub const HOOK_POST_TOOL_USE: &str = "post_tool_use";
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
