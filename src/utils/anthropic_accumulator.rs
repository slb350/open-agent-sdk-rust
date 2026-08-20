//! Decoding of Anthropic streaming events into the shared stream buffers.
//!
//! The sibling of [`StreamAccumulator`](super::StreamAccumulator). Both hold the same
//! contract — `process` never returns [`StreamEvent::Finish`], `finalize` always ends with
//! exactly one — and both meet it the same way, by draining a
//! [`StreamBuffers`](super::buffers::StreamBuffers). What lives here is only the part that is
//! genuinely Anthropic's: which wire event means what.
//!
//! # Routing is by delta type, not by block kind
//!
//! Anthropic opens each block with its kind and then appends deltas by index, so there are
//! two things a fragment could be routed by. Every delta is self-describing — `text_delta`,
//! `thinking_delta`, `input_json_delta` — and routing by that tag is what makes the
//! reasoning separation a property of the parser rather than of bookkeeping that a missing
//! `content_block_start` could defeat. Block starts are tracked only for what deltas cannot
//! carry: a tool call's id and name.

use super::buffers::StreamBuffers;
use crate::types::{
    AnthropicBlockStart, AnthropicDelta, AnthropicEvent, StreamEvent, anthropic_finish_reason,
};
use crate::{Error, Result};

/// Aggregates [`AnthropicEvent`]s into completed [`StreamEvent`]s.
pub struct AnthropicAccumulator {
    /// Everything accumulated so far, and the policy for draining it.
    buffers: StreamBuffers,
}

impl AnthropicAccumulator {
    /// Creates an accumulator with empty buffers and reasoning capture disabled.
    pub fn new() -> Self {
        Self {
            buffers: StreamBuffers::new(),
        }
    }

    /// Sets whether thinking deltas are retained for emission.
    ///
    /// When disabled, thinking is still parsed off the wire and then dropped as each delta
    /// arrives, so a caller that did not ask for a chain of thought never pays to hold one.
    pub fn capture_reasoning(mut self, capture: bool) -> Self {
        self.buffers.set_capture_reasoning(capture);
        self
    }

    /// Processes one event and returns any events it completed.
    ///
    /// Returns an empty vector while generation is ongoing, and the drained buffers when the
    /// server reports `stop_reason`. Never returns [`StreamEvent::Finish`]; only
    /// [`AnthropicAccumulator::finalize`] emits that.
    ///
    /// # Errors
    ///
    /// Returns an error for a mid-stream `error` event, and for a tool call whose assembled
    /// arguments are not valid JSON.
    pub fn process_event(&mut self, event: AnthropicEvent) -> Result<Vec<StreamEvent>> {
        match event {
            AnthropicEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                self.open_block(index, content_block);
                Ok(Vec::new())
            }

            AnthropicEvent::ContentBlockDelta { index, delta } => {
                self.append_delta(index, delta);
                Ok(Vec::new())
            }

            AnthropicEvent::MessageDelta { delta } => match delta.stop_reason {
                Some(raw) => {
                    self.buffers.record_finish(anthropic_finish_reason(&raw));
                    self.buffers.flush()
                }
                None => Ok(Vec::new()),
            },

            AnthropicEvent::Error { error } => Err(stream_error(&error)),

            // `content_block_stop` closes a block whose content is already buffered, and
            // `message_stop` follows the `message_delta` that carried the stop reason.
            // Neither adds information, and flushing on them would emit content twice.
            AnthropicEvent::MessageStart {}
            | AnthropicEvent::ContentBlockStop { .. }
            | AnthropicEvent::MessageStop {}
            | AnthropicEvent::Ping {}
            | AnthropicEvent::Unknown => Ok(Vec::new()),
        }
    }

    /// Records a block's identity, and any content it already carried.
    fn open_block(&mut self, index: u32, block: AnthropicBlockStart) {
        match block {
            AnthropicBlockStart::Text { text } => self.buffers.push_text(&text),
            AnthropicBlockStart::Thinking { thinking } => self.buffers.push_reasoning(&thinking),
            AnthropicBlockStart::ToolUse { id, name } => {
                let call = self.buffers.tool_call(index);
                call.id = Some(id);
                call.name = Some(name);
            }
            AnthropicBlockStart::RedactedThinking {} | AnthropicBlockStart::Unknown => {}
        }
    }

    /// Routes one fragment to the channel its own tag names.
    fn append_delta(&mut self, index: u32, delta: AnthropicDelta) {
        match delta {
            AnthropicDelta::TextDelta { text } => self.buffers.push_text(&text),
            AnthropicDelta::ThinkingDelta { thinking } => self.buffers.push_reasoning(&thinking),
            AnthropicDelta::InputJsonDelta { partial_json } => {
                // A fragment for a block that never opened has no id or name to emit under,
                // so there is nothing to attach it to. Dropping it is the only option that
                // does not invent a tool call.
                if let Some(call) = self.buffers.open_tool_call(index) {
                    call.arguments.push_str(&partial_json);
                }
            }
            AnthropicDelta::SignatureDelta {} | AnthropicDelta::Unknown => {}
        }
    }

    /// Drains anything left and emits the terminating [`StreamEvent::Finish`].
    ///
    /// Called once when the transport ends. A response that stopped without the server ever
    /// sending `stop_reason` still yields its content and finishes as
    /// [`FinishReason::Unspecified`](crate::FinishReason::Unspecified), which stays distinct
    /// from [`FinishReason::Stop`](crate::FinishReason::Stop) because the SDK has no evidence
    /// the model finished cleanly.
    pub fn finalize(&mut self) -> Result<Vec<StreamEvent>> {
        self.buffers.finalize()
    }
}

impl Default for AnthropicAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Turns a mid-stream `error` event into a retry-classifiable [`Error`].
///
/// The SDK's retry layer reads [`Error::status_code`], and a mid-stream error arrives with
/// no HTTP status of its own because the response already returned 200. Mapping the two
/// error kinds that are genuinely transient onto the statuses they would have carried had
/// they arrived before the stream opened is what lets `retry_with_backoff_conditional` treat
/// them as retryable; anything else stays a stream error, which is not retried.
fn stream_error(error: &crate::types::AnthropicErrorBody) -> Error {
    let (described, status) = match error.error_type.as_deref() {
        Some(kind) => {
            let status = match kind {
                "overloaded_error" => Some(529),
                "rate_limit_error" => Some(429),
                "api_error" => Some(500),
                _ => None,
            };
            (format!("{kind}: {}", error.message), status)
        }
        None => (error.message.clone(), None),
    };

    match status {
        Some(status) => Error::api_status(status, described),
        None => Error::stream(described),
    }
}

impl super::EventAccumulator for AnthropicAccumulator {
    type Event = AnthropicEvent;

    fn process(&mut self, event: Self::Event) -> Result<Vec<StreamEvent>> {
        self.process_event(event)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        self.finalize()
    }
}

#[cfg(test)]
mod tests;
