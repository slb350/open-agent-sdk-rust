//! Assertions shared by the streaming unit tests.
//!
//! Both accumulators and the buffers beneath them are checked against the same structural
//! guarantees, so the assertions live once. A per-module copy of [`sole_finish`] in
//! particular would be three chances to weaken the guarantee for one protocol without any
//! test noticing.

use crate::types::{FinishReason, StreamEvent};

/// Concatenates every text block in an event batch.
pub fn text_of(events: &[StreamEvent]) -> String {
    events.iter().filter_map(StreamEvent::as_text).collect()
}

/// Concatenates every reasoning event in a batch.
pub fn reasoning_of(events: &[StreamEvent]) -> String {
    events
        .iter()
        .filter_map(StreamEvent::as_reasoning)
        .collect()
}

/// The reason from the batch's single [`StreamEvent::Finish`], asserting it is also last.
pub fn sole_finish(events: &[StreamEvent]) -> FinishReason {
    let reasons: Vec<&FinishReason> = events
        .iter()
        .filter_map(StreamEvent::finish_reason)
        .collect();
    assert_eq!(reasons.len(), 1, "expected one Finish, got {events:?}");
    assert!(
        events
            .last()
            .is_some_and(|event| event.finish_reason().is_some()),
        "Finish must be last, got {events:?}"
    );
    reasons[0].clone()
}
