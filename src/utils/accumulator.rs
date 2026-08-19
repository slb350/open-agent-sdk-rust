//! Aggregation of streaming deltas into completed [`StreamEvent`]s.

use crate::types::{ContentBlock, FinishReason, OpenAIChunk, StreamEvent, TextBlock, ToolUseBlock};
use crate::{Error, Result};
use std::collections::BTreeMap;

/// Aggregates streaming deltas into completed [`StreamEvent`]s.
///
/// This is a **stateful accumulator** that processes [`OpenAIChunk`] objects one at a time,
/// building up complete text, reasoning, and tool call content over multiple chunks. It
/// returns completed events when a `finish_reason` is encountered, and the stream driver must
/// call [`StreamAccumulator::finalize`] once the transport ends so that servers which never
/// send a `finish_reason` do not have their content silently discarded — and so that every
/// stream terminates with exactly one [`StreamEvent::Finish`].
///
/// # State Management
///
/// The accumulator maintains four pieces of state:
///
/// 1. **Text Buffer** (`text_buffer`): Accumulates text content across chunks. Text deltas
///    are concatenated as they arrive. When generation finishes, the complete text is
///    emitted as a [`ContentBlock::Text`].
///
/// 2. **Reasoning Buffer** (`reasoning_buffer`): Accumulates chain-of-thought deltas from the
///    provider's side channel, kept strictly apart from `text_buffer`. Only populated when
///    reasoning capture is enabled; otherwise reasoning deltas are read and dropped.
///
/// 3. **Tool Call Map** (`tool_calls`): A HashMap indexed by tool call index (provided by
///    the API) that tracks partially-received tool calls. Each entry accumulates the tool's
///    ID, name, and JSON argument string. When generation finishes, all tool calls are
///    emitted as [`ContentBlock::ToolUse`] blocks in ascending index order.
///
/// 4. **Finish Reason** (`finish_reason`): The first `finish_reason` observed on the stream,
///    replayed as the terminating [`StreamEvent::Finish`] by [`StreamAccumulator::finalize`].
///
/// # Why Index-Based Storage?
///
/// The API can return multiple tool calls in a single response, and they arrive interleaved:
///
/// ```text
/// Chunk 1: tool_calls[0] = { id: "call_1", name: "search" }
/// Chunk 2: tool_calls[1] = { id: "call_2", name: "calculate" }
/// Chunk 3: tool_calls[0] = { arguments: "{\"q\"" }
/// Chunk 4: tool_calls[1] = { arguments: "{\"expr\"" }
/// Chunk 5: tool_calls[0] = { arguments: ":\"rust\"}" }
/// Chunk 6: tool_calls[1] = { arguments: ":\"2+2\"}" }
/// ```
///
/// The HashMap keyed by index allows us to correctly accumulate each tool call independently.
///
/// # Usage Pattern
///
/// ```rust,ignore
/// let mut accumulator = StreamAccumulator::new();
///
/// for chunk in stream {
///     let events = accumulator.process_chunk(chunk)?;
///     // events is empty until finish_reason is encountered
///     handle_events(events);
/// }
///
/// // The transport ended. Emit anything the server left unterminated, then Finish.
/// handle_events(accumulator.finalize()?);
/// ```
///
/// # Important Invariants
///
/// - **Buffers are cleared after finish**: Once a `finish_reason` is seen, the text, reasoning,
///   and tool call buffers are drained. A subsequent [`StreamAccumulator::finalize`] therefore
///   emits no duplicate content, so end-of-stream finalization never double-emits.
///
/// - **`Finish` is emitted once, last**: `process_chunk` records the reason but never emits
///   the event; only `finalize` does. This holds the ordering guarantee even for servers that
///   keep sending after their own `finish_reason`.
///
/// - **Reasoning never becomes content**: reasoning deltas are read through
///   [`OpenAIDelta::reasoning_delta`] and routed to `reasoning_buffer`. They cannot reach
///   `text_buffer` by any path.
///
/// - **Partial JSON accumulation**: Tool call arguments are accumulated as raw strings and
///   only parsed as JSON when the tool call is complete. This allows JSON to be split at
///   arbitrary boundaries across chunks.
pub struct StreamAccumulator {
    /// Buffer for accumulating text content deltas across chunks.
    /// Drained when a finish_reason is encountered or the transport ends.
    text_buffer: String,

    /// Buffer for accumulating reasoning deltas across chunks.
    /// Stays empty unless `capture_reasoning` is set.
    reasoning_buffer: String,

    /// Whether reasoning deltas are retained for emission or read and discarded.
    capture_reasoning: bool,

    /// Map of partially-received tool calls, keyed by their API-provided index.
    /// Each entry accumulates ID, name, and argument deltas. Ordered rather than hashed so
    /// that draining it yields parallel tool calls in the order the model requested them.
    tool_calls: BTreeMap<u32, PartialToolCall>,

    /// The first `finish_reason` the server reported, replayed by `finalize`.
    finish_reason: Option<FinishReason>,
}

/// Represents an in-progress tool call that is being assembled from deltas.
///
/// Tool calls arrive fragmented across multiple chunks. This struct accumulates the pieces
/// until we have a complete tool call ready to be converted into a [`ToolUseBlock`].
///
/// # Field Evolution
///
/// As chunks arrive, fields are populated incrementally:
///
/// ```text
/// Initial state:     { id: None, name: None, arguments: "" }
/// After chunk 1:     { id: Some("call_123"), name: Some("search"), arguments: "" }
/// After chunk 2:     { id: Some("call_123"), name: Some("search"), arguments: "{\"q" }
/// After chunk 3:     { id: Some("call_123"), name: Some("search"), arguments: "{\"q\":\"rust\"}" }
/// ```
///
/// # Completion Criteria
///
/// A `PartialToolCall` is considered **complete** when:
/// 1. A `finish_reason` is encountered, or the transport ends
/// 2. Both `id` and `name` are `Some(_)`
/// 3. The `arguments` string is valid JSON (validated during parsing)
///
/// Incomplete tool calls (missing ID or name) are silently dropped during aggregation.
#[derive(Debug, Default)]
struct PartialToolCall {
    /// Unique identifier for the tool call. Usually arrives in the first chunk.
    id: Option<String>,

    /// Name of the tool/function to call. Usually arrives in the first chunk.
    name: Option<String>,

    /// Accumulated JSON argument string. Built up incrementally across chunks.
    /// May be split at arbitrary byte positions across chunks.
    arguments: String,
}

impl StreamAccumulator {
    /// Creates a new accumulator with empty buffers and reasoning capture disabled.
    pub fn new() -> Self {
        Self {
            text_buffer: String::new(),
            reasoning_buffer: String::new(),
            capture_reasoning: false,
            tool_calls: BTreeMap::new(),
            finish_reason: None,
        }
    }

    /// Sets whether reasoning deltas are retained for emission.
    ///
    /// When disabled (the default), reasoning is still parsed off the wire but discarded as
    /// each delta arrives rather than accumulated. A caller that did not ask for a chain of
    /// thought never retains it, so the cost is O(1) rather than O(trace length). The delta
    /// itself is still allocated one layer down by serde and freed immediately.
    pub fn capture_reasoning(mut self, capture: bool) -> Self {
        self.capture_reasoning = capture;
        self
    }

    /// Processes a single chunk and returns any events it completed.
    ///
    /// # Arguments
    ///
    /// * `chunk` - A single [`OpenAIChunk`] from the streaming response
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<StreamEvent>)` - Empty while generation is ongoing, or the drained buffers
    ///   when `finish_reason` is encountered. Never contains [`StreamEvent::Finish`]; that is
    ///   emitted exclusively by [`StreamAccumulator::finalize`].
    /// * `Err(Error)` - If tool call argument JSON is invalid
    ///
    /// # Errors
    ///
    /// Returns an error if tool call arguments cannot be parsed as valid JSON. This indicates
    /// either a streaming error or malformed data from the API.
    pub fn process_chunk(&mut self, chunk: OpenAIChunk) -> Result<Vec<StreamEvent>> {
        // Vector to collect completed events. Will be empty unless finish_reason is present.
        let mut events = Vec::new();

        // A chunk can contain multiple choices, though typically there's only one.
        // Each choice represents a separate generation path (used in n>1 scenarios).
        for choice in chunk.choices {
            // === PHASE 1: ROUTE REASONING DELTAS ===
            // Read before `content` is moved out of the delta. Reasoning goes to its own
            // buffer or nowhere at all; it must never reach `text_buffer`, because a caller
            // parsing the text as JSON would get deliberation prose spliced into its payload.
            if let Some(reasoning) = choice.delta.reasoning_delta() {
                if self.capture_reasoning {
                    self.reasoning_buffer.push_str(reasoning);
                }
            }

            // === PHASE 2: ACCUMULATE TEXT DELTAS ===
            // If this chunk contains text content, append it to our buffer.
            // Text arrives as incremental strings: "Hello", " ", "world", etc.
            if let Some(content) = choice.delta.content {
                self.text_buffer.push_str(&content);
            }

            // === PHASE 3: ACCUMULATE TOOL CALL DELTAS ===
            // Tool calls are more complex - they can arrive as multiple interleaved deltas.
            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    // Get or create the partial tool call for this index.
                    // The index is provided by the API and identifies which tool call
                    // this delta belongs to (important when multiple tools are called).
                    let entry = self.tool_calls.entry(tool_call.index).or_default();

                    // Update the ID if present. Usually only in the first delta for this tool.
                    if let Some(id) = tool_call.id {
                        entry.id = Some(id);
                    }

                    // Handle function/tool details
                    if let Some(function) = tool_call.function {
                        // Update the name if present. Usually only in the first delta.
                        if let Some(name) = function.name {
                            entry.name = Some(name);
                        }

                        // Append argument delta. This is where JSON gets assembled character by
                        // character. The API may split JSON at any position, even mid-string:
                        // Chunk 1: "{\"loc"
                        // Chunk 2: "ation\":"
                        // Chunk 3: "\"Paris\"}"
                        if let Some(args) = function.arguments {
                            entry.arguments.push_str(&args);
                        }
                    }
                }
            }

            // === PHASE 4: CHECK FOR COMPLETION ===
            // finish_reason indicates that generation is complete. The reason is recorded for
            // `finalize` to replay; only the accumulated content is emitted here. The first
            // reason wins, so a second choice reporting a different one cannot overwrite it.
            if let Some(raw) = choice.finish_reason {
                if self.finish_reason.is_none() {
                    self.finish_reason = Some(FinishReason::from_wire(&raw));
                }
                events.extend(self.flush()?);
            }
        }

        Ok(events)
    }

    /// Drains all accumulated content into completed events.
    ///
    /// Emits reasoning first (it precedes the answer it produced), then the text block, then
    /// tool calls in ascending index order. Called internally whenever a `finish_reason` is
    /// observed, and by [`StreamAccumulator::finalize`] at end of transport.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<StreamEvent>)` - Buffered reasoning, text, and tool calls. Empty if nothing
    ///   was accumulated since the last drain, which makes the method safe to call
    ///   unconditionally without double-emitting content.
    /// * `Err(Error)` - If a tool call's accumulated arguments are not valid JSON, which
    ///   indicates the stream was truncated or corrupted mid-tool-call.
    fn flush(&mut self) -> Result<Vec<StreamEvent>> {
        let mut events = Vec::new();

        // === FLUSH REASONING BUFFER ===
        // Always empty unless capture was requested, so this costs nothing by default.
        if !self.reasoning_buffer.is_empty() {
            events.push(StreamEvent::Reasoning(std::mem::take(
                &mut self.reasoning_buffer,
            )));
        }

        // === FLUSH TEXT BUFFER ===
        // If we accumulated any text, emit it as a TextBlock. `take` moves the buffer into the
        // block and leaves an empty String behind, so the whole response body is not copied.
        if !self.text_buffer.is_empty() {
            events.push(StreamEvent::Block(ContentBlock::Text(TextBlock::new(
                std::mem::take(&mut self.text_buffer),
            ))));
        }

        // === FLUSH AND VALIDATE TOOL CALLS ===
        // Taking the BTreeMap gives us ownership of all partial tool calls in ascending index
        // order, so parallel tool calls are emitted in the order the model requested them.
        for partial in std::mem::take(&mut self.tool_calls).into_values() {
            // Only emit tool calls that have both ID and name.
            // Incomplete tool calls are silently dropped (shouldn't happen with valid API).
            if let (Some(id), Some(name)) = (partial.id, partial.name) {
                // Parse the accumulated JSON argument string.
                // If arguments is empty, default to an empty object {}.
                let input: serde_json::Value = if partial.arguments.is_empty() {
                    serde_json::json!({})
                } else {
                    // This is where we validate that all the assembled JSON is valid.
                    // If the streaming was corrupted or incomplete, this will error.
                    serde_json::from_str(&partial.arguments).map_err(|e| {
                        Error::stream(format!("Failed to parse tool arguments: {}", e))
                    })?
                };

                events.push(StreamEvent::Block(ContentBlock::ToolUse(
                    ToolUseBlock::new(id, name, input),
                )));
            }
        }

        Ok(events)
    }

    /// Signals end of transport: drains remaining content, then emits the terminating event.
    ///
    /// The stream driver must call this when the underlying transport terminates. Not every
    /// OpenAI-compatible server sets `finish_reason` on its final content chunk — llama.cpp,
    /// vLLM, and several local gateways stream content and then send `data: [DONE]` (or simply
    /// close the connection) with `finish_reason` still null. Without this call, everything
    /// accumulated so far would be discarded silently and the caller would see an empty
    /// successful response.
    ///
    /// # Returns
    ///
    /// Any content still buffered, followed by exactly one [`StreamEvent::Finish`]. The reason
    /// is whatever the server reported, or [`FinishReason::Unspecified`] when it reported
    /// nothing — which is deliberately not conflated with [`FinishReason::Stop`].
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

#[cfg(test)]
mod tests {
    use super::*;

    include!("tests/accumulator.rs");
}
