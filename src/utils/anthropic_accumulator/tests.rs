//! Unit tests for [`AnthropicAccumulator`].

use super::*;
use crate::types::{AnthropicErrorBody, AnthropicMessageDelta, ContentBlock, FinishReason};
use crate::utils::test_support::{reasoning_of, sole_finish, text_of};

/// Feeds `events` through a fresh accumulator and returns everything it emitted, finalize
/// included. `capture` selects reasoning capture.
fn run(events: Vec<AnthropicEvent>, capture: bool) -> Result<Vec<StreamEvent>> {
    let mut accumulator = AnthropicAccumulator::new().capture_reasoning(capture);
    let mut out = Vec::new();
    for event in events {
        out.extend(accumulator.process_event(event)?);
    }
    out.extend(accumulator.finalize()?);
    Ok(out)
}

fn text_start(index: u32) -> AnthropicEvent {
    AnthropicEvent::ContentBlockStart {
        index,
        content_block: AnthropicBlockStart::Text {
            text: String::new(),
        },
    }
}

fn text_delta(index: u32, text: &str) -> AnthropicEvent {
    AnthropicEvent::ContentBlockDelta {
        index,
        delta: AnthropicDelta::TextDelta {
            text: text.to_string(),
        },
    }
}

fn thinking_delta(index: u32, thinking: &str) -> AnthropicEvent {
    AnthropicEvent::ContentBlockDelta {
        index,
        delta: AnthropicDelta::ThinkingDelta {
            thinking: thinking.to_string(),
        },
    }
}

fn json_delta(index: u32, partial: &str) -> AnthropicEvent {
    AnthropicEvent::ContentBlockDelta {
        index,
        delta: AnthropicDelta::InputJsonDelta {
            partial_json: partial.to_string(),
        },
    }
}

fn tool_start(index: u32, id: &str, name: &str) -> AnthropicEvent {
    AnthropicEvent::ContentBlockStart {
        index,
        content_block: AnthropicBlockStart::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
        },
    }
}

fn stop(reason: &str) -> AnthropicEvent {
    AnthropicEvent::MessageDelta {
        delta: AnthropicMessageDelta {
            stop_reason: Some(reason.to_string()),
        },
    }
}

#[test]
fn text_deltas_concatenate_into_one_block() {
    let events = run(
        vec![
            AnthropicEvent::MessageStart {},
            text_start(0),
            text_delta(0, "Hel"),
            text_delta(0, "lo"),
            AnthropicEvent::ContentBlockStop { index: 0 },
            stop("end_turn"),
            AnthropicEvent::MessageStop {},
        ],
        false,
    )
    .expect("no error");

    assert_eq!(text_of(&events), "Hello");
    assert_eq!(sole_finish(&events), FinishReason::Stop);
}

#[test]
fn text_present_on_the_opening_block_is_not_lost() {
    let events = run(
        vec![
            AnthropicEvent::ContentBlockStart {
                index: 0,
                content_block: AnthropicBlockStart::Text {
                    text: "He".to_string(),
                },
            },
            text_delta(0, "llo"),
            stop("end_turn"),
        ],
        false,
    )
    .expect("no error");

    assert_eq!(text_of(&events), "Hello");
}

#[test]
fn thinking_never_reaches_assistant_text() {
    let events = run(
        vec![
            AnthropicEvent::ContentBlockStart {
                index: 0,
                content_block: AnthropicBlockStart::Thinking {
                    thinking: String::new(),
                },
            },
            thinking_delta(0, "deliberating"),
            AnthropicEvent::ContentBlockStop { index: 0 },
            text_start(1),
            text_delta(1, "answer"),
            stop("end_turn"),
        ],
        true,
    )
    .expect("no error");

    assert_eq!(text_of(&events), "answer");
    assert_eq!(reasoning_of(&events), "deliberating");
}

#[test]
fn thinking_is_dropped_when_capture_is_off() {
    let events = run(
        vec![thinking_delta(0, "deliberating"), stop("end_turn")],
        false,
    )
    .expect("no error");

    assert_eq!(reasoning_of(&events), "");
    assert_eq!(text_of(&events), "", "and it never became content either");
}

#[test]
fn thinking_present_on_the_opening_block_is_captured() {
    let events = run(
        vec![
            AnthropicEvent::ContentBlockStart {
                index: 0,
                content_block: AnthropicBlockStart::Thinking {
                    thinking: "seed".to_string(),
                },
            },
            thinking_delta(0, "-more"),
            stop("end_turn"),
        ],
        true,
    )
    .expect("no error");

    assert_eq!(reasoning_of(&events), "seed-more");
}

#[test]
fn reasoning_precedes_the_text_it_produced() {
    let events = run(
        vec![
            thinking_delta(0, "think"),
            text_delta(1, "say"),
            stop("end_turn"),
        ],
        true,
    )
    .expect("no error");

    let reasoning_at = events
        .iter()
        .position(|event| event.as_reasoning().is_some())
        .expect("a reasoning event");
    let text_at = events
        .iter()
        .position(|event| event.as_text().is_some())
        .expect("a text event");
    assert!(reasoning_at < text_at, "got {events:?}");
}

#[test]
fn a_tool_call_assembles_from_its_start_and_json_fragments() {
    let events = run(
        vec![
            tool_start(0, "toolu_1", "get_weather"),
            json_delta(0, r#"{"location":"#),
            json_delta(0, r#""SF"}"#),
            stop("tool_use"),
        ],
        false,
    )
    .expect("no error");

    let block = events
        .iter()
        .find_map(StreamEvent::as_block)
        .expect("a block");
    match block {
        ContentBlock::ToolUse(call) => {
            assert_eq!(call.id(), "toolu_1");
            assert_eq!(call.name(), "get_weather");
            assert_eq!(*call.input(), serde_json::json!({ "location": "SF" }));
        }
        other => panic!("expected a tool call, got {other:?}"),
    }
    assert_eq!(sole_finish(&events), FinishReason::ToolCalls);
}

#[test]
fn a_tool_call_with_no_arguments_gets_an_empty_object() {
    let events = run(vec![tool_start(0, "t", "ping"), stop("tool_use")], false).expect("no error");

    match events
        .iter()
        .find_map(StreamEvent::as_block)
        .expect("a block")
    {
        ContentBlock::ToolUse(call) => assert_eq!(*call.input(), serde_json::json!({})),
        other => panic!("expected a tool call, got {other:?}"),
    }
}

#[test]
fn malformed_tool_arguments_fail_the_stream() {
    let error = run(
        vec![
            tool_start(0, "t", "search"),
            json_delta(0, "{not json"),
            stop("tool_use"),
        ],
        false,
    )
    .expect_err("truncated tool arguments are an error");

    assert!(
        error.to_string().contains("search"),
        "the message names the tool: {error}"
    );
}

#[test]
fn parallel_tool_calls_are_emitted_in_block_order() {
    let events = run(
        vec![
            // Opened out of order to prove the ordering comes from the index, not arrival.
            tool_start(1, "second", "b"),
            tool_start(0, "first", "a"),
            stop("tool_use"),
        ],
        false,
    )
    .expect("no error");

    let ids: Vec<String> = events
        .iter()
        .filter_map(StreamEvent::as_block)
        .filter_map(|block| match block {
            ContentBlock::ToolUse(call) => Some(call.id().to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(ids, vec!["first".to_string(), "second".to_string()]);
}

#[test]
fn a_json_fragment_for_a_block_that_never_opened_invents_no_tool_call() {
    let events = run(vec![json_delta(7, r#"{"a":1}"#), stop("end_turn")], false).expect("no error");

    assert!(
        events.iter().all(|event| event.as_block().is_none()),
        "got {events:?}"
    );
}

#[test]
fn a_stream_that_never_reports_a_stop_reason_still_yields_its_content() {
    let events = run(vec![text_start(0), text_delta(0, "orphan")], false).expect("no error");

    assert_eq!(text_of(&events), "orphan");
    assert_eq!(
        sole_finish(&events),
        FinishReason::Unspecified,
        "no evidence the model finished cleanly"
    );
}

#[test]
fn the_first_stop_reason_wins() {
    let events = run(
        vec![text_delta(0, "x"), stop("max_tokens"), stop("end_turn")],
        false,
    )
    .expect("no error");

    assert_eq!(sole_finish(&events), FinishReason::Length);
}

#[test]
fn content_is_not_emitted_twice_when_the_stream_keeps_going_after_stopping() {
    let events = run(
        vec![
            text_delta(0, "once"),
            stop("end_turn"),
            AnthropicEvent::ContentBlockStop { index: 0 },
            AnthropicEvent::MessageStop {},
        ],
        false,
    )
    .expect("no error");

    assert_eq!(text_of(&events), "once");
    assert_eq!(
        events.iter().filter(|e| e.as_text().is_some()).count(),
        1,
        "got {events:?}"
    );
}

#[test]
fn a_message_delta_without_a_stop_reason_does_not_flush() {
    let mut accumulator = AnthropicAccumulator::new();
    accumulator
        .process_event(text_delta(0, "partial"))
        .expect("no error");

    let emitted = accumulator
        .process_event(AnthropicEvent::MessageDelta {
            delta: AnthropicMessageDelta { stop_reason: None },
        })
        .expect("no error");

    assert!(emitted.is_empty(), "got {emitted:?}");
}

#[test]
fn ping_and_unknown_events_are_ignored() {
    let events = run(
        vec![
            AnthropicEvent::Ping {},
            AnthropicEvent::Unknown,
            text_delta(0, "kept"),
            AnthropicEvent::Ping {},
            stop("end_turn"),
        ],
        false,
    )
    .expect("no error");

    assert_eq!(text_of(&events), "kept");
}

#[test]
fn a_signature_delta_contributes_nothing_to_either_channel() {
    let events = run(
        vec![
            AnthropicEvent::ContentBlockDelta {
                index: 0,
                delta: AnthropicDelta::SignatureDelta {},
            },
            stop("end_turn"),
        ],
        true,
    )
    .expect("no error");

    assert_eq!(text_of(&events), "");
    assert_eq!(reasoning_of(&events), "");
}

#[test]
fn an_unrecognised_delta_does_not_become_assistant_text() {
    let events = run(
        vec![
            AnthropicEvent::ContentBlockDelta {
                index: 0,
                delta: AnthropicDelta::Unknown,
            },
            text_delta(0, "real"),
            stop("end_turn"),
        ],
        false,
    )
    .expect("no error");

    assert_eq!(text_of(&events), "real");
}

#[test]
fn a_redacted_thinking_block_contributes_nothing() {
    let events = run(
        vec![
            AnthropicEvent::ContentBlockStart {
                index: 0,
                content_block: AnthropicBlockStart::RedactedThinking {},
            },
            stop("end_turn"),
        ],
        true,
    )
    .expect("no error");

    assert_eq!(text_of(&events), "");
    assert_eq!(reasoning_of(&events), "");
}

#[test]
fn an_overload_error_is_retryable_by_status() {
    let error = run(
        vec![AnthropicEvent::Error {
            error: AnthropicErrorBody {
                error_type: Some("overloaded_error".to_string()),
                message: "Overloaded".to_string(),
            },
        }],
        false,
    )
    .expect_err("an error event fails the stream");

    assert_eq!(
        error.status_code(),
        Some(529),
        "the retry layer reads the status, not the message: {error}"
    );
}

#[test]
fn a_rate_limit_error_carries_429() {
    let error = run(
        vec![AnthropicEvent::Error {
            error: AnthropicErrorBody {
                error_type: Some("rate_limit_error".to_string()),
                message: "slow down".to_string(),
            },
        }],
        false,
    )
    .expect_err("an error event fails the stream");

    assert_eq!(error.status_code(), Some(429));
}

#[test]
fn a_server_side_api_error_carries_500() {
    let error = run(
        vec![AnthropicEvent::Error {
            error: AnthropicErrorBody {
                error_type: Some("api_error".to_string()),
                message: "internal".to_string(),
            },
        }],
        false,
    )
    .expect_err("an error event fails the stream");

    assert_eq!(
        error.status_code(),
        Some(500),
        "a server-side fault is transient and must reach the retry layer as one: {error}"
    );
}

#[test]
fn an_unclassifiable_error_is_not_given_a_retryable_status() {
    let error = run(
        vec![AnthropicEvent::Error {
            error: AnthropicErrorBody {
                error_type: Some("invalid_request_error".to_string()),
                message: "bad model".to_string(),
            },
        }],
        false,
    )
    .expect_err("an error event fails the stream");

    assert_eq!(
        error.status_code(),
        None,
        "asking again cannot fix a malformed request: {error}"
    );
    assert!(error.to_string().contains("bad model"), "{error}");
}

#[test]
fn an_error_message_names_its_kind() {
    let error = run(
        vec![AnthropicEvent::Error {
            error: AnthropicErrorBody {
                error_type: Some("overloaded_error".to_string()),
                message: "Overloaded".to_string(),
            },
        }],
        false,
    )
    .expect_err("an error event fails the stream");

    assert!(error.to_string().contains("overloaded_error"), "{error}");
}

#[test]
fn an_error_without_a_kind_still_reports_its_message() {
    let error = run(
        vec![AnthropicEvent::Error {
            error: AnthropicErrorBody {
                error_type: None,
                message: "something went wrong".to_string(),
            },
        }],
        false,
    )
    .expect_err("an error event fails the stream");

    assert!(
        error.to_string().contains("something went wrong"),
        "{error}"
    );
    assert_eq!(error.status_code(), None);
}

#[test]
fn content_emits_on_arrival_and_is_never_re_emitted() {
    let mut accumulator = AnthropicAccumulator::new();
    let delta = accumulator
        .process_event(text_delta(0, "body"))
        .expect("no error");
    let flushed = accumulator
        .process_event(stop("end_turn"))
        .expect("no error");
    let finalized = accumulator.finalize().expect("no error");

    assert_eq!(text_of(&delta), "body");
    assert_eq!(text_of(&flushed), "", "the drain must not repeat it");
    assert_eq!(text_of(&finalized), "");
    assert_eq!(finalized.len(), 1, "only Finish remains: {finalized:?}");
}

#[test]
fn default_matches_new() {
    let mut from_default = AnthropicAccumulator::default();
    let events = from_default.finalize().expect("no error");

    assert_eq!(sole_finish(&events), FinishReason::Unspecified);
}
