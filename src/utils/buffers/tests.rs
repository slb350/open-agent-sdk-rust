//! Unit tests for [`StreamBuffers`].
//!
//! The protocol accumulators test their own decoding. What is tested here is the policy they
//! share, at the level it now lives at rather than once per protocol.

use super::*;
use crate::utils::test_support::sole_finish;

#[test]
fn nothing_accumulated_flushes_to_nothing() {
    let mut buffers = StreamBuffers::new();
    assert!(buffers.flush().expect("flush").is_empty());
}

#[test]
fn reasoning_precedes_text_which_precedes_tool_calls() {
    let mut buffers = StreamBuffers::new();
    buffers.set_capture_reasoning(true);
    buffers.push_reasoning("thinking");
    buffers.push_text("answer");
    let call = buffers.tool_call(0);
    call.id = Some("call_1".to_string());
    call.name = Some("search".to_string());

    let events = buffers.flush().expect("flush");

    assert!(matches!(events[0], StreamEvent::Reasoning(_)), "{events:?}");
    assert!(
        matches!(events[1], StreamEvent::Block(ContentBlock::Text(_))),
        "{events:?}"
    );
    assert!(
        matches!(events[2], StreamEvent::Block(ContentBlock::ToolUse(_))),
        "{events:?}"
    );
    assert_eq!(events.len(), 3, "{events:?}");
}

#[test]
fn tool_calls_emit_in_ascending_index_order_regardless_of_arrival() {
    let mut buffers = StreamBuffers::new();
    for (index, name) in [(2, "third"), (0, "first"), (1, "second")] {
        let call = buffers.tool_call(index);
        call.id = Some(format!("call_{index}"));
        call.name = Some(name.to_string());
    }

    let names: Vec<String> = buffers
        .flush()
        .expect("flush")
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Block(ContentBlock::ToolUse(call)) => Some(call.name().to_string()),
            _ => None,
        })
        .collect();

    assert_eq!(names, vec!["first", "second", "third"]);
}

#[test]
fn reasoning_is_dropped_on_arrival_unless_capture_is_enabled() {
    let mut buffers = StreamBuffers::new();
    buffers.push_reasoning("deliberation");
    buffers.push_text("answer");

    let events = buffers.flush().expect("flush");

    assert!(
        !events
            .iter()
            .any(|event| matches!(event, StreamEvent::Reasoning(_))),
        "{events:?}"
    );
    // The critical half: dropping it must not mean merging it into the answer.
    assert_eq!(events.len(), 1, "{events:?}");
    match &events[0] {
        StreamEvent::Block(ContentBlock::Text(text)) => assert_eq!(text.text, "answer"),
        other => panic!("expected text, got {other:?}"),
    }
}

#[test]
fn the_first_finish_reason_wins() {
    let mut buffers = StreamBuffers::new();
    buffers.record_finish(FinishReason::ToolCalls);
    buffers.record_finish(FinishReason::Stop);

    assert_eq!(
        sole_finish(&buffers.finalize().expect("finalize")),
        FinishReason::ToolCalls
    );
}

#[test]
fn an_unreported_finish_reason_is_unspecified_and_not_stop() {
    let mut buffers = StreamBuffers::new();

    let events = buffers.finalize().expect("finalize");

    assert_eq!(sole_finish(&events), FinishReason::Unspecified);
    assert_ne!(sole_finish(&events), FinishReason::Stop);
}

#[test]
fn finalize_after_a_flush_does_not_re_emit_content() {
    let mut buffers = StreamBuffers::new();
    buffers.push_text("answer");

    let flushed = buffers.flush().expect("flush");
    assert_eq!(flushed.len(), 1, "{flushed:?}");

    let finalized = buffers.finalize().expect("finalize");
    assert_eq!(finalized.len(), 1, "only Finish remains: {finalized:?}");
    assert_eq!(sole_finish(&finalized), FinishReason::Unspecified);
}

#[test]
fn a_tool_call_missing_its_id_or_name_is_dropped() {
    let mut buffers = StreamBuffers::new();
    buffers.tool_call(0).name = Some("no_id".to_string());
    buffers.tool_call(1).id = Some("no_name".to_string());
    let complete = buffers.tool_call(2);
    complete.id = Some("call_2".to_string());
    complete.name = Some("kept".to_string());

    let events = buffers.flush().expect("flush");

    assert_eq!(events.len(), 1, "{events:?}");
    match &events[0] {
        StreamEvent::Block(ContentBlock::ToolUse(call)) => assert_eq!(call.name(), "kept"),
        other => panic!("expected a tool call, got {other:?}"),
    }
}

#[test]
fn empty_arguments_become_an_empty_object() {
    let mut buffers = StreamBuffers::new();
    let call = buffers.tool_call(0);
    call.id = Some("call_1".to_string());
    call.name = Some("now".to_string());

    let events = buffers.flush().expect("flush");

    match &events[0] {
        StreamEvent::Block(ContentBlock::ToolUse(call)) => {
            assert_eq!(call.input(), &serde_json::json!({}));
        }
        other => panic!("expected a tool call, got {other:?}"),
    }
}

#[test]
fn truncated_arguments_error_and_name_the_tool() {
    let mut buffers = StreamBuffers::new();
    let call = buffers.tool_call(0);
    call.id = Some("call_1".to_string());
    call.name = Some("search".to_string());
    call.arguments.push_str("{\"q\": \"unter");

    let error = buffers.flush().expect_err("invalid JSON must error");

    assert!(
        error
            .to_string()
            .contains("Failed to parse tool call arguments for 'search'"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_fragment_for_a_block_that_never_opened_has_nowhere_to_go() {
    let mut buffers = StreamBuffers::new();
    assert!(buffers.open_tool_call(0).is_none());

    // `tool_call` opens one; `open_tool_call` then finds it.
    buffers.tool_call(0);
    assert!(buffers.open_tool_call(0).is_some());
}
