//! Lifecycle hooks for prompts and automatic tool execution.
//!
//! Handlers run in registration order until one returns `Some(HookDecision)`.
//! Return `None` to let later handlers run, including audit handlers.
//!
//! `UserPromptSubmit` runs in [`Client::send`](crate::Client::send). Pre- and
//! post-tool hooks run only during automatic tool execution. Each event owns a
//! structured conversation-history snapshot; post-tool history includes the
//! unmodified result. Handle fallible work inside a hook and choose a decision;
//! hook futures return `Option<HookDecision>`, not `Result`.

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
