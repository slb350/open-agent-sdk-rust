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

include!("hooks/events.rs");
include!("hooks/decision.rs");
include!("hooks/handlers.rs");
include!("hooks/registry.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    include!("hooks/tests.rs");
}
