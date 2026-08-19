//! Streaming utilities for SSE parsing and stream accumulation.
//!
//! Two stages sit between the raw HTTP response and the [`StreamEvent`](crate::StreamEvent)s
//! a caller sees:
//!
//! ```text
//! Raw HTTP body (SSE)  ──parse_sse_stream──▶  Stream<OpenAIChunk>
//!                      ──StreamAccumulator──▶  Vec<StreamEvent>
//! ```
//!
//! - [`sse`] decodes the SSE wire format, buffering across arbitrary HTTP transport chunk
//!   boundaries so an event split mid-JSON (or mid-UTF-8) still parses.
//! - [`accumulator`] reassembles the per-chunk deltas — text, reasoning, and tool call
//!   arguments all arrive fragmented — and decides when a complete block can be emitted.
//!
//! The rationale for each stage lives with its code rather than being restated here.

// Real submodules rather than `include!` fragments: `cargo-mutants` walks `mod`
// declarations but does not expand `include!`, so fragment-backed code is invisible to the
// mutation gate. Re-exporting keeps `crate::utils::{StreamAccumulator, parse_sse_stream}`
// unchanged for callers.
mod accumulator;
mod sse;

pub use accumulator::StreamAccumulator;
pub use sse::parse_sse_stream;
