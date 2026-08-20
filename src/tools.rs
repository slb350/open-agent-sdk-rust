//! # Tool System for Open Agent SDK
//!
//! This module provides a comprehensive tool definition system. One tool definition serves
//! both wire protocols: it converts to OpenAI's function calling shape and to Anthropic's
//! `name`/`description`/`input_schema` shape.
//!
//! ## Architecture Overview
//!
//! The tool system is built around three core concepts:
//!
//! 1. **Tool Definition** - The [`Tool`] struct contains metadata (name, description, schema)
//!    and an async handler function that executes the tool's logic.
//!
//! 2. **Schema Flexibility** - Supports both simple type notation and full JSON Schema,
//!    automatically converting to whichever wire format the endpoint's protocol uses.
//!
//! 3. **Async Execution** - Tools run asynchronously with a pinned, boxed future pattern
//!    that enables dynamic dispatch and easy integration with async runtimes.
//!
//! ## Tool Lifecycle
//!
//! ```text
//! 1. Definition:   Create tool with name, description, schema, and handler
//! 2. Registration: Add tool to agent's tool registry
//! 3. Invocation:   LLM decides to call tool with specific arguments
//! 4. Execution:    Handler processes arguments and returns result
//! 5. Response:     Result is sent back to LLM for further processing
//! ```
//!
//! ## Schema Conversion
//!
//! The system intelligently handles multiple schema formats:
//!
//! - **Simple notation**: `{"location": "string", "units": "string"}`
//! - **Typed schema**: `{"param": {"type": "number", "description": "A value"}}`
//! - **Full JSON Schema**: Already valid JSON Schema with "type" and "properties"
//!
//! All formats are normalized to OpenAI's expected JSON Schema structure.
//!
//! ## Handler Pattern
//!
//! Tool handlers use `Pin<Box<dyn Future>>` for several critical reasons:
//!
//! - **Type Erasure**: Different async functions have different concrete types.
//!   Boxing allows storing handlers with varying types in a single collection.
//!
//! - **Pinning**: Futures in Rust must be pinned to a memory location before polling.
//!   Pin guarantees the future won't move, which is essential for self-referential types.
//!
//! - **Send + Sync**: These bounds ensure handlers can be safely shared across threads,
//!   crucial for concurrent agent operations.
//!
//! ## Examples
//!
//! ### Creating a Simple Tool
//!
//! ```rust,no_run
//! use open_agent::{tool, Result};
//! use serde_json::json;
//!
//! // Using the builder pattern
//! let weather_tool = tool("get_weather", "Get current weather for a location")
//!     .param("location", "string")
//!     .param("units", "string")
//!     .build(|args| async move {
//!         let location = args["location"].as_str().unwrap_or("Unknown");
//!         let units = args["units"].as_str().unwrap_or("celsius");
//!
//!         // Simulate API call
//!         Ok(json!({
//!             "location": location,
//!             "temperature": 22,
//!             "units": units
//!         }))
//!     });
//! ```
//!
//! ### Creating a Tool with Complex Schema
//!
//! ```rust,no_run
//! use open_agent::Tool;
//! use serde_json::json;
//!
//! let search_tool = Tool::new(
//!     "search",
//!     "Search the web for information",
//!     json!({
//!         "query": {
//!             "type": "string",
//!             "description": "Search query"
//!         },
//!         "max_results": {
//!             "type": "integer",
//!             "description": "Maximum number of results",
//!             "optional": true
//!         }
//!     }),
//!     |args| Box::pin(async move {
//!         // Implementation
//!         Ok(json!({"results": []}))
//!     })
//! );
//! ```

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
