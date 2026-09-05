//! Configuration, conversation messages, and wire-format types.
//!
//! [`AgentOptionsBuilder`] validates configuration at runtime. [`Message`] and
//! [`ContentBlock`] describe conversation history; OpenAI and Anthropic request
//! types handle the transport representation.

use crate::Error;
use crate::hooks::Hooks;
use crate::tools::Tool;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

include!("types/validated.rs");
include!("types/agent_options.rs");
include!("types/agent_options_builder.rs");
include!("types/message_blocks.rs");
include!("types/image.rs");
include!("types/message.rs");
include!("types/openai.rs");
// Real submodules rather than `include!` fragments, so `cargo-mutants` — which walks `mod`
// declarations but does not expand `include!` — can mutate the logic they hold. The
// re-exports keep every public path unchanged. The remaining fragments stay fragments
// because they reach into module-private fields of types their siblings define, which real
// modules could only reach by weakening that encapsulation.
mod anthropic;
mod anthropic_stream;
pub(crate) mod http_headers;
mod openai_stream;
mod protocol;
mod stream_event;

// Glob re-export: the wire types are consumed across the crate (and by unit tests) exactly
// as the `include!` fragment used to expose them, so this keeps every path identical.
pub use openai_stream::*;
pub use stream_event::{FinishReason, StreamEvent};

pub use anthropic::{AnthropicMessage, AnthropicRequest};
pub use anthropic_stream::{
    AnthropicBlockStart, AnthropicDelta, AnthropicErrorBody, AnthropicEvent, AnthropicMessageDelta,
    anthropic_finish_reason,
};
pub use protocol::ApiProtocol;

#[cfg(test)]
mod tests {
    use super::*;

    include!("types/tests/core.rs");
    include!("types/tests/image.rs");
}
