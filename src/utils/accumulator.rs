//! Decoding of OpenAI streaming deltas into the shared stream buffers.

use super::buffers::StreamBuffers;
use crate::Result;
use crate::types::{FinishReason, OpenAIChunk, StreamEvent};

/// Aggregates streaming deltas into completed [`StreamEvent`]s.
///
/// This is a **stateful accumulator** that processes [`OpenAIChunk`] objects one at a time.
/// Text and reasoning fragments are forwarded as each chunk decodes them; tool call arguments
/// accumulate, because they are split at arbitrary byte positions and are not valid JSON
/// until the last fragment lands. The stream driver must call
/// [`StreamAccumulator::finalize`] once the transport ends, so that a tool call left
/// unterminated by a server which never sends a `finish_reason` is not silently discarded —
/// and so that every stream terminates with exactly one [`StreamEvent::Finish`].
///
/// # State Management
///
/// The state itself — the tool call map and the first-seen
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
///     // Text and reasoning arrive here, one event per fragment.
///     handle_events(events);
/// }
///
/// // The transport ended. Emit anything the server left unterminated, then Finish.
/// handle_events(accumulator.finalize()?);
/// ```
///
/// # Important Invariants
///
/// - **The drain empties itself**: Once a `finish_reason` is seen, the tool call map is
///   drained. A subsequent [`StreamAccumulator::finalize`] therefore emits no duplicate
///   content, so end-of-stream finalization never double-emits.
///
/// - **`Finish` is emitted once, last**: `process_chunk` records the reason but never emits
///   the event; only `finalize` does. This holds the ordering guarantee even for servers that
///   keep sending after their own `finish_reason`.
///
/// - **Reasoning never becomes content**: reasoning deltas are read through
///   [`OpenAIDelta::reasoning_delta`](crate::types::OpenAIDelta::reasoning_delta) and handed
///   to the reasoning channel. They cannot reach the content channel by any path.
///
/// - **Partial JSON accumulation**: Tool call arguments are accumulated as raw strings and
///   only parsed as JSON when the tool call is complete. This allows JSON to be split at
///   arbitrary boundaries across chunks.
pub struct StreamAccumulator {
    /// The tool calls under assembly, and the policy for draining them.
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
    /// When disabled (the default), reasoning is still parsed off the wire but dropped as
    /// each delta arrives rather than forwarded. A caller that did not ask for a chain of
    /// thought never sees one. The delta itself is still allocated one layer down by serde
    /// and freed immediately.
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
    /// * `Ok(Vec<StreamEvent>)` - The text and reasoning fragments this chunk carried, plus
    ///   the assembled tool calls when `finish_reason` is encountered. Never contains
    ///   [`StreamEvent::Finish`]; that is emitted exclusively by
    ///   [`StreamAccumulator::finalize`].
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
            // channel or nowhere at all; it must never reach the content channel, because a
            // caller parsing the text as JSON would get deliberation prose in its payload.
            if let Some(reasoning) = choice.delta.reasoning_delta() {
                events.extend(self.buffers.push_reasoning(reasoning));
            }

            // === PHASE 2: FORWARD TEXT DELTAS ===
            // Text arrives as incremental strings — "Hello", " ", "world" — and each one goes
            // straight to the caller.
            if let Some(content) = choice.delta.content {
                events.extend(self.buffers.push_text(content));
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
            // `finalize` to replay; only the assembled tool calls are emitted here. The first
            // reason wins, so a second choice reporting a different one cannot overwrite it.
            if let Some(raw) = choice.finish_reason {
                self.buffers.record_finish(FinishReason::from_wire(&raw));
                events.extend(self.buffers.flush()?);
            }
        }

        Ok(events)
    }

    /// Signals end of transport: drains any assembled tool calls, then emits the terminating
    /// event.
    ///
    /// The stream driver must call this when the underlying transport terminates. Not every
    /// OpenAI-compatible server sets `finish_reason` on its final content chunk — llama.cpp,
    /// vLLM, and several local gateways stream content and then send `data: [DONE]` (or simply
    /// close the connection) with `finish_reason` still null. Without this call, everything
    /// assembled so far would be discarded silently and the caller would never learn the
    /// model asked for a tool.
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
