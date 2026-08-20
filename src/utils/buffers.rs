//! The buffers both protocols drain, and the policy for draining them.
//!
//! [`StreamAccumulator`](super::StreamAccumulator) and
//! [`AnthropicAccumulator`](super::AnthropicAccumulator) decode different wire vocabularies,
//! but what they do with the results is the same: forward text and reasoning to the caller as
//! each fragment lands, assemble tool calls by index, remember the first stop reason, and
//! drain what is left in one fixed order. Only the decoding differs, so only the decoding
//! lives per protocol.
//!
//! Text and reasoning pass straight through because a fragment of either is already
//! meaningful. Tool calls cannot: their arguments arrive split at arbitrary byte positions and
//! are not valid JSON until the last fragment lands, so they are the only thing still held to
//! the drain.
//!
//! Four of the crate's documented invariants are decided here rather than twice over:
//! `Finish` is emitted exactly once and only by [`StreamBuffers::finalize`],
//! [`FinishReason::Unspecified`] stays distinct from [`FinishReason::Stop`], reasoning never
//! reaches the content channel, and parallel tool calls emit in ascending index order.

use std::collections::BTreeMap;

use crate::types::{ContentBlock, FinishReason, StreamEvent, TextBlock, ToolUseBlock};
use crate::{Error, Result};

/// A tool call being assembled from fragments spread across several wire events.
///
/// `id` and `name` are optional because OpenAI sends them in whichever delta it likes and can
/// stop streaming before either arrives; Anthropic names both when it opens the block. A call
/// still missing either at drain time has nothing to emit under and is dropped.
#[derive(Debug, Default)]
pub struct PartialToolCall {
    /// Correlation id, echoed back with the tool result.
    pub id: Option<String>,

    /// Name of the tool the model wants to run.
    pub name: Option<String>,

    /// Accumulated JSON arguments, split at arbitrary byte positions across events.
    pub arguments: String,
}

/// The accumulated state of one streaming response.
pub struct StreamBuffers {
    /// Whether reasoning fragments are emitted or dropped on arrival.
    capture_reasoning: bool,

    /// Tool calls under construction, keyed by the index the wire gave them. Ordered rather
    /// than hashed, so draining yields parallel calls in the order the model opened them.
    tool_calls: BTreeMap<u32, PartialToolCall>,

    /// The first stop reason the server reported, replayed by [`StreamBuffers::finalize`].
    finish_reason: Option<FinishReason>,
}

impl StreamBuffers {
    /// Creates empty buffers with reasoning capture disabled.
    pub fn new() -> Self {
        Self {
            capture_reasoning: false,
            tool_calls: BTreeMap::new(),
            finish_reason: None,
        }
    }

    /// Sets whether reasoning fragments are emitted.
    pub fn set_capture_reasoning(&mut self, capture: bool) {
        self.capture_reasoning = capture;
    }

    /// Turns a fragment of assistant text into the event that carries it.
    ///
    /// `None` for an empty fragment: the first chunk of an OpenAI stream routinely carries an
    /// empty content string alongside the role, and a block per such chunk would hand callers
    /// a stream of empty strings to filter out.
    pub fn push_text(&self, text: impl Into<String>) -> Option<StreamEvent> {
        let text = text.into();
        (!text.is_empty()).then(|| StreamEvent::Block(ContentBlock::Text(TextBlock::new(text))))
    }

    /// Turns a fragment of reasoning into an event, or drops it when capture is disabled.
    ///
    /// The check lives here so that no decoder can route reasoning into the content channel by
    /// forgetting it, which is what makes the separation a property of this type.
    pub fn push_reasoning(&self, reasoning: impl Into<String>) -> Option<StreamEvent> {
        // Converted only when capture is on, so a discarded trace costs nothing to discard.
        self.capture_reasoning
            .then(|| reasoning.into())
            .filter(|reasoning: &String| !reasoning.is_empty())
            .map(StreamEvent::Reasoning)
    }

    /// The tool call at `index`, opening one if this is the first fragment for it.
    ///
    /// OpenAI never announces a tool call before streaming its parts, so the first fragment
    /// to mention an index is what opens it.
    pub fn tool_call(&mut self, index: u32) -> &mut PartialToolCall {
        self.tool_calls.entry(index).or_default()
    }

    /// The tool call at `index`, or `None` when nothing opened one.
    ///
    /// Anthropic opens every tool call with a `content_block_start` carrying its id and name,
    /// so a fragment for an index that never opened has nothing to attach to. Dropping it is
    /// the only option that does not invent a tool call.
    pub fn open_tool_call(&mut self, index: u32) -> Option<&mut PartialToolCall> {
        self.tool_calls.get_mut(&index)
    }

    /// Records why generation stopped, keeping the first reason reported.
    ///
    /// A server that keeps sending after its own stop reason — or reports a different one per
    /// choice — cannot overwrite what it said first.
    pub fn record_finish(&mut self, reason: FinishReason) {
        if self.finish_reason.is_none() {
            self.finish_reason = Some(reason);
        }
    }

    /// Drains the assembled tool calls, in ascending index order.
    ///
    /// Text and reasoning have already reached the caller by the time this runs. Empty when
    /// nothing has accumulated since the last drain, which is what makes it safe to call
    /// unconditionally without double-emitting.
    ///
    /// # Errors
    ///
    /// Returns an error when a tool call's assembled arguments are not valid JSON, which
    /// means the stream was truncated or corrupted mid-call.
    pub fn flush(&mut self) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();

        for partial in std::mem::take(&mut self.tool_calls).into_values() {
            // A call missing either half has nothing to emit under. It should not happen
            // against a well-behaved server, and inventing an id would be worse than the gap.
            let (Some(id), Some(name)) = (partial.id, partial.name) else {
                continue;
            };

            let input = if partial.arguments.is_empty() {
                serde_json::json!({})
            } else {
                serde_json::from_str(&partial.arguments).map_err(|error| {
                    Error::stream(format!(
                        "Failed to parse tool call arguments for '{name}': {error}"
                    ))
                })?
            };

            events.push(StreamEvent::Block(ContentBlock::ToolUse(
                ToolUseBlock::new(id, name, input),
            )));
        }

        Ok(events)
    }

    /// Drains anything left and emits the terminating [`StreamEvent::Finish`].
    ///
    /// A response that ended without the server ever reporting a stop reason still yields its
    /// content and finishes as [`FinishReason::Unspecified`], which stays distinct from
    /// [`FinishReason::Stop`] because the SDK has no evidence the model finished cleanly.
    pub fn finalize(&mut self) -> Result<Vec<StreamEvent>> {
        let mut events = self.flush()?;
        events.push(StreamEvent::Finish(
            self.finish_reason
                .take()
                .unwrap_or(FinishReason::Unspecified),
        ));
        Ok(events)
    }
}

impl Default for StreamBuffers {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
