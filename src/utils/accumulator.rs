//! Decoding of OpenAI streaming deltas into the shared stream buffers.

use super::buffers::StreamBuffers;
use crate::Result;
use crate::types::{FinishReason, OpenAIChunk, StreamEvent};

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
/// The state itself — text buffer, reasoning buffer, tool call map and first-seen
/// `finish_reason` — lives in [`StreamBuffers`](super::buffers::StreamBuffers), shared with
/// the Anthropic accumulator, because only the decoding below differs between the two
/// protocols. What remains here is the mapping from OpenAI's delta shape onto those buffers.
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
/// Keying by the API-provided index accumulates each tool call independently, and the map is
/// ordered so draining it yields them in the order the model requested.
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
///   [`OpenAIDelta::reasoning_delta`](crate::types::OpenAIDelta::reasoning_delta) and handed
///   to the reasoning channel. They cannot reach the text buffer by any path.
///
/// - **Partial JSON accumulation**: Tool call arguments are accumulated as raw strings and
///   only parsed as JSON when the tool call is complete. This allows JSON to be split at
///   arbitrary boundaries across chunks.
pub struct StreamAccumulator {
    /// Everything accumulated so far, and the policy for draining it.
    buffers: StreamBuffers,
}

impl StreamAccumulator {
    /// Creates a new accumulator with empty buffers and reasoning capture disabled.
    pub fn new() -> Self {
        Self {
            buffers: StreamBuffers::new(),
        }
    }

    /// Sets whether reasoning deltas are retained for emission.
    ///
    /// When disabled (the default), reasoning is still parsed off the wire but discarded as
    /// each delta arrives rather than accumulated. A caller that did not ask for a chain of
    /// thought never retains it, so the cost is O(1) rather than O(trace length). The delta
    /// itself is still allocated one layer down by serde and freed immediately.
    pub fn capture_reasoning(mut self, capture: bool) -> Self {
        self.buffers.set_capture_reasoning(capture);
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
            // buffer or nowhere at all; it must never reach the text buffer, because a caller
            // parsing the text as JSON would get deliberation prose spliced into its payload.
            if let Some(reasoning) = choice.delta.reasoning_delta() {
                self.buffers.push_reasoning(reasoning);
            }

            // === PHASE 2: ACCUMULATE TEXT DELTAS ===
            // If this chunk contains text content, append it to our buffer.
            // Text arrives as incremental strings: "Hello", " ", "world", etc.
            if let Some(content) = choice.delta.content {
                self.buffers.push_text(&content);
            }

            // === PHASE 3: ACCUMULATE TOOL CALL DELTAS ===
            // Tool calls are more complex - they can arrive as multiple interleaved deltas.
            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    // Get or create the partial tool call for this index. The index is
                    // provided by the API and identifies which tool call this delta belongs
                    // to (important when multiple tools are called).
                    let entry = self.buffers.tool_call(tool_call.index);

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
                self.buffers.record_finish(FinishReason::from_wire(&raw));
                events.extend(self.buffers.flush()?);
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
        self.buffers.finalize()
    }
}

impl super::EventAccumulator for StreamAccumulator {
    type Event = OpenAIChunk;

    fn process(&mut self, event: Self::Event) -> Result<Vec<StreamEvent>> {
        self.process_chunk(event)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        self.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    include!("tests/accumulator.rs");
}
