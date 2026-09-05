//! Tool definitions, schema conversion, and asynchronous execution.
//!
//! Use [`tool()`] for a fluent builder or [`Tool::new`] for direct construction.
//! The handler validates its own arguments; schemas describe the tool to the model.

use crate::Result;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

include!("tools/handler.rs");
include!("tools/tool.rs");
include!("tools/schema.rs");
include!("tools/builder.rs");
include!("tools/factory.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use serde_json::json;

    include!("tools/tests.rs");
}
