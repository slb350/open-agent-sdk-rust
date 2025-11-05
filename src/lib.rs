//! Open Agent SDK - Rust implementation
//!
//! A clean, streaming API for working with OpenAI-compatible local model servers.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use open_agent::{query, AgentOptions, ContentBlock};
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let options = AgentOptions::builder()
//!         .system_prompt("You are a helpful assistant")
//!         .model("qwen2.5-32b-instruct")
//!         .base_url("http://localhost:1234/v1")
//!         .build()?;
//!
//!     let mut stream = query("What's the capital of France?", &options).await?;
//!
//!     while let Some(block) = stream.next().await {
//!         match block? {
//!             ContentBlock::Text(text_block) => {
//!                 print!("{}", text_block.text);
//!             }
//!             ContentBlock::ToolUse(tool_block) => {
//!                 println!("Tool called: {}", tool_block.name);
//!             }
//!             ContentBlock::ToolResult(_) => {
//!                 // Tool results can be ignored in simple queries
//!             }
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```

mod client;
mod config;
mod context;
mod error;
mod hooks;
mod tools;
mod types;
mod utils;

pub mod retry;

pub use client::{Client, query};
pub use config::{Provider, get_base_url, get_model};
pub use context::{estimate_tokens, is_approaching_limit, truncate_messages};
pub use error::{Error, Result};
pub use hooks::{
    HOOK_POST_TOOL_USE, HOOK_PRE_TOOL_USE, HOOK_USER_PROMPT_SUBMIT, HookDecision, Hooks,
    PostToolUseEvent, PreToolUseEvent, UserPromptSubmitEvent,
};
pub use tools::{Tool, ToolBuilder, tool};
pub use types::{
    AgentOptions, AgentOptionsBuilder, ContentBlock, Message, MessageRole, TextBlock,
    ToolResultBlock, ToolUseBlock,
};

/// Re-export common types for convenience
pub mod prelude {
    pub use crate::{
        AgentOptions, AgentOptionsBuilder, Client, ContentBlock, Error, HookDecision, Hooks,
        PostToolUseEvent, PreToolUseEvent, Result, TextBlock, Tool, ToolUseBlock,
        UserPromptSubmitEvent, query, tool,
    };
}
