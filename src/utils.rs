//! Streaming utilities for SSE parsing and stream accumulation.
//!
//! Three stages sit between the raw HTTP response and the [`StreamEvent`](crate::StreamEvent)s
//! a caller sees:
//!
//! ```text
//! Raw HTTP body (SSE)  ──sse──▶  Stream<wire event>
//!                      ──accumulator──▶  Vec<StreamEvent>
//!                      ──driver──▶  Stream<StreamEvent>
//! ```
//!
//! - [`sse`] decodes the SSE wire format, buffering across arbitrary HTTP transport chunk
//!   boundaries so an event split mid-JSON (or mid-UTF-8) still parses.
//! - [`accumulator`] and [`anthropic_accumulator`] decode the per-event deltas, forwarding
//!   text and reasoning as each fragment lands and reassembling tool call arguments, which are
//!   not valid JSON until the last one does. One per wire protocol, since only the vocabulary
//!   differs.
//! - [`driver`] owns everything that does not: the end-of-transport sentinel, threading the
//!   accumulator through the stream, and flattening its batches.
//!
//! The rationale for each stage lives with its code rather than being restated here.

// Real submodules rather than `include!` fragments: `cargo-mutants` walks `mod`
// declarations but does not expand `include!`, so fragment-backed code is invisible to the
// mutation gate. Re-exporting keeps `crate::utils::{StreamAccumulator, parse_sse_stream}`
// unchanged for callers.
mod accumulator;
mod anthropic_accumulator;
mod buffers;
mod coalesce;
mod driver;
mod sse;

#[cfg(test)]
mod test_support;

pub use accumulator::StreamAccumulator;
pub use anthropic_accumulator::AnthropicAccumulator;
pub use coalesce::coalesce_text_blocks;
pub use driver::{EventAccumulator, drive};
pub use sse::{parse_anthropic_sse_stream, parse_sse_stream};
