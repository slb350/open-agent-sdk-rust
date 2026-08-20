//! Wire types for the Anthropic messages streaming response.
//!
//! Anthropic streams a typed event sequence rather than OpenAI's uniform chunk-with-deltas:
//! blocks are opened, appended to, and closed by index, and the reason generation stopped
//! arrives on a `message_delta` near the end. Every event carries its own `type`, so the SSE
//! `event:` line is redundant and the SDK parses only the `data:` payload.
//!
//! Every enum here has an `Unknown` catch-all. Third-party Anthropic-compatible endpoints
//! emit events this SDK has never heard of, and a hard parse failure on one of them would
//! discard a response that was otherwise complete.

use serde::Deserialize;

use super::FinishReason;

/// One event from an Anthropic streaming response.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AnthropicEvent {
    /// Opens the response. Carries usage and the empty message envelope, neither of which
    /// the SDK needs.
    MessageStart {},

    /// Opens a content block at `index`, declaring what kind it is.
    ContentBlockStart {
        /// Position of the block within the response.
        index: u32,
        /// The block's kind, and its identity when it is a tool call.
        content_block: AnthropicBlockStart,
    },

    /// Appends to the block at `index`.
    ContentBlockDelta {
        /// Position of the block being appended to.
        index: u32,
        /// The fragment to append, tagged by which channel it belongs to.
        delta: AnthropicDelta,
    },

    /// Closes the block at `index`.
    ContentBlockStop {
        /// Position of the block being closed.
        index: u32,
    },

    /// Reports top-level message changes; this is where `stop_reason` arrives.
    MessageDelta {
        /// The changed fields.
        delta: AnthropicMessageDelta,
    },

    /// Ends the response.
    MessageStop {},

    /// Keep-alive. Carries nothing.
    Ping {},

    /// A mid-stream error, such as an overload. Terminates the response.
    Error {
        /// The error body.
        error: AnthropicErrorBody,
    },

    /// An event type this SDK does not recognise, ignored rather than fatal.
    #[serde(other)]
    Unknown,
}

/// The declaration that opens a content block.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AnthropicBlockStart {
    /// Assistant text.
    Text {
        /// Text present at the point the block opened. Normally empty; some compatible
        /// servers front-load the first fragment here rather than sending a delta for it,
        /// and dropping it would lose the opening characters of the answer.
        #[serde(default)]
        text: String,
    },

    /// Extended thinking, which belongs to the reasoning channel and never to content.
    Thinking {
        /// Thinking text present at the point the block opened, for the same reason as
        /// [`AnthropicBlockStart::Text`].
        #[serde(default)]
        thinking: String,
    },

    /// Thinking the server has redacted. Carries ciphertext, never plain reasoning, so it is
    /// tracked as a block kind and its payload discarded.
    RedactedThinking {},

    /// A tool call. Its arguments arrive later as `input_json_delta` fragments.
    ToolUse {
        /// Correlation id, echoed back with the tool result.
        id: String,
        /// Name of the tool the model wants to run.
        name: String,
    },

    /// A block kind this SDK does not recognise. It carries no id or name to emit under, so
    /// the block itself contributes nothing; its deltas are still routed by their own tag,
    /// which is what keeps an unrecognised channel out of assistant text.
    #[serde(other)]
    Unknown,
}

/// One fragment appended to an open block.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AnthropicDelta {
    /// A fragment of assistant text.
    TextDelta {
        /// The fragment.
        text: String,
    },

    /// A fragment of extended thinking.
    ThinkingDelta {
        /// The fragment.
        thinking: String,
    },

    /// The cryptographic signature over a thinking block. Not reasoning text and not
    /// content, so it is parsed and dropped.
    SignatureDelta {},

    /// A fragment of a tool call's JSON arguments. Split at arbitrary byte positions.
    InputJsonDelta {
        /// The fragment.
        partial_json: String,
    },

    /// A delta type this SDK does not recognise.
    #[serde(other)]
    Unknown,
}

/// Top-level message changes, carrying the reason generation stopped.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessageDelta {
    /// Why generation stopped. Null until the model actually stops.
    #[serde(default)]
    pub stop_reason: Option<String>,
}

/// The body of a mid-stream `error` event.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicErrorBody {
    /// The error's machine-readable kind, e.g. `"overloaded_error"`.
    #[serde(rename = "type", default)]
    pub error_type: Option<String>,

    /// Human-readable description.
    #[serde(default)]
    pub message: String,
}

/// Maps an Anthropic `stop_reason` onto the SDK's protocol-neutral [`FinishReason`].
///
/// Anthropic and OpenAI agree on none of the spellings, so the OpenAI-shaped
/// [`FinishReason::from_wire`] would file every one of these under
/// [`FinishReason::Other`] and callers branching on `Length` would never see a truncation.
///
/// `model_context_window_exceeded` maps to [`FinishReason::Length`] because it is a token
/// ceiling like any other, and a caller's correct response — send less, do not simply ask
/// again — is the same one `Length` already prescribes. `pause_turn` keeps its own name:
/// the turn is resumable, which no existing variant means, and inventing an equivalence
/// would tell a caller the response finished when it did not.
///
/// # Examples
///
/// ```rust
/// use open_agent::FinishReason;
/// use open_agent::anthropic_finish_reason;
///
/// assert_eq!(anthropic_finish_reason("end_turn"), FinishReason::Stop);
/// assert_eq!(anthropic_finish_reason("max_tokens"), FinishReason::Length);
/// ```
pub fn anthropic_finish_reason(raw: &str) -> FinishReason {
    match raw.to_ascii_lowercase().as_str() {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" | "model_context_window_exceeded" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "refusal" => FinishReason::ContentFilter,
        _ => FinishReason::Other(raw.to_string()),
    }
}

#[cfg(test)]
mod tests;
