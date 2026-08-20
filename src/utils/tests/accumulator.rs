use crate::types::{
    ContentBlock, OpenAIChoice, OpenAIDelta, OpenAIFunctionDelta, OpenAIToolCallDelta,
};
use crate::utils::test_support::text_of;

/// Builds a chunk carrying one choice with the given delta and finish reason.
fn chunk(delta: OpenAIDelta, finish_reason: Option<&str>) -> OpenAIChunk {
    OpenAIChunk {
        id: "test".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 0,
        model: "test".to_string(),
        choices: vec![OpenAIChoice {
            index: 0,
            delta,
            finish_reason: finish_reason.map(str::to_string),
        }],
    }
}

/// An otherwise-empty delta, so each test names only the fields it exercises.
fn delta() -> OpenAIDelta {
    OpenAIDelta {
        role: None,
        content: None,
        tool_calls: None,
        reasoning_content: None,
        reasoning: None,
    }
}

fn text_delta(content: &str) -> OpenAIDelta {
    OpenAIDelta {
        content: Some(content.to_string()),
        ..delta()
    }
}

fn tool_call_delta(index: u32, id: &str, name: &str, arguments: &str) -> OpenAIDelta {
    OpenAIDelta {
        tool_calls: Some(vec![OpenAIToolCallDelta {
            index,
            id: Some(id.to_string()),
            call_type: Some("function".to_string()),
            function: Some(OpenAIFunctionDelta {
                name: Some(name.to_string()),
                arguments: Some(arguments.to_string()),
            }),
        }]),
        ..delta()
    }
}


// ============================================================================
// FINISH REASON
// ============================================================================

#[test]
fn process_chunk_never_emits_the_finish_event_itself() {
    let mut accumulator = StreamAccumulator::new();

    let events = accumulator
        .process_chunk(chunk(text_delta("done"), Some("stop")))
        .expect("chunk processes");

    assert_eq!(text_of(&events), "done");
    assert!(
        events.iter().all(|event| event.finish_reason().is_none()),
        "Finish belongs to finalize(), not process_chunk(): {events:?}"
    );
}

#[test]
fn finalize_replays_the_reported_finish_reason() {
    let mut accumulator = StreamAccumulator::new();
    accumulator
        .process_chunk(chunk(text_delta("x"), Some("length")))
        .expect("chunk processes");

    let events = accumulator.finalize().expect("finalize");
    assert_eq!(
        events.last().and_then(StreamEvent::finish_reason),
        Some(&FinishReason::Length)
    );
}

#[test]
fn finalize_reports_unspecified_when_the_server_never_said() {
    let mut accumulator = StreamAccumulator::new();
    let mut events = accumulator
        .process_chunk(chunk(text_delta("stranded"), None))
        .expect("chunk processes");
    events.extend(accumulator.finalize().expect("finalize"));

    // The 0.7.0 flush fix, now carried by delivery on arrival: content must survive a stream
    // that never reports a reason.
    assert_eq!(text_of(&events), "stranded");
    assert_eq!(
        events.last().and_then(StreamEvent::finish_reason),
        Some(&FinishReason::Unspecified)
    );
}

#[test]
fn finalize_emits_exactly_one_finish_event() {
    let mut accumulator = StreamAccumulator::new();
    accumulator
        .process_chunk(chunk(text_delta("x"), Some("stop")))
        .expect("chunk processes");

    let events = accumulator.finalize().expect("finalize");
    assert_eq!(
        events
            .iter()
            .filter(|event| event.finish_reason().is_some())
            .count(),
        1,
        "{events:?}"
    );
    assert!(events.last().unwrap().finish_reason().is_some());
}

#[test]
fn the_first_finish_reason_wins_over_a_later_one() {
    // A second choice reporting a different reason must not overwrite the first.
    let mut accumulator = StreamAccumulator::new();
    accumulator
        .process_chunk(chunk(delta(), Some("length")))
        .expect("first finish");
    accumulator
        .process_chunk(chunk(delta(), Some("stop")))
        .expect("second finish");

    assert_eq!(
        accumulator
            .finalize()
            .expect("finalize")
            .last()
            .and_then(StreamEvent::finish_reason),
        Some(&FinishReason::Length)
    );
}

#[test]
fn finalize_after_a_finish_reason_does_not_repeat_content() {
    let mut accumulator = StreamAccumulator::new();
    let flushed = accumulator
        .process_chunk(chunk(text_delta("once"), Some("stop")))
        .expect("chunk processes");
    assert_eq!(text_of(&flushed), "once");

    let events = accumulator.finalize().expect("finalize");
    assert_eq!(text_of(&events), "", "content must not be emitted twice");
}

// ============================================================================
// REASONING CHANNEL
// ============================================================================

#[test]
fn reasoning_is_discarded_by_default() {
    let mut accumulator = StreamAccumulator::new();
    let reasoning = OpenAIDelta {
        reasoning_content: Some("deliberation".to_string()),
        ..delta()
    };

    accumulator
        .process_chunk(chunk(reasoning, None))
        .expect("chunk processes");
    let events = accumulator
        .process_chunk(chunk(text_delta("answer"), Some("stop")))
        .expect("chunk processes");

    assert_eq!(text_of(&events), "answer");
    assert!(
        events.iter().all(|event| event.as_reasoning().is_none()),
        "{events:?}"
    );
}

#[test]
fn captured_reasoning_is_emitted_before_the_text_it_produced() {
    let mut accumulator = StreamAccumulator::new().capture_reasoning(true);
    let both = OpenAIDelta {
        content: Some("answer".to_string()),
        reasoning_content: Some("because".to_string()),
        ..delta()
    };

    let events = accumulator
        .process_chunk(chunk(both, Some("stop")))
        .expect("chunk processes");

    assert_eq!(events[0].as_reasoning(), Some("because"));
    assert_eq!(events[1].as_text(), Some("answer"));
}

#[test]
fn reasoning_is_never_concatenated_into_text() {
    let mut accumulator = StreamAccumulator::new().capture_reasoning(true);
    let reasoning = OpenAIDelta {
        reasoning: Some("SHOULD NOT APPEAR".to_string()),
        ..delta()
    };

    accumulator
        .process_chunk(chunk(reasoning, None))
        .expect("chunk processes");
    let events = accumulator
        .process_chunk(chunk(text_delta("{\"ok\":true}"), Some("stop")))
        .expect("chunk processes");

    assert_eq!(text_of(&events), "{\"ok\":true}");
}

#[test]
fn a_gateway_mirroring_both_channels_is_counted_once() {
    let mut accumulator = StreamAccumulator::new().capture_reasoning(true);
    let mirrored = OpenAIDelta {
        reasoning: Some("trace".to_string()),
        reasoning_content: Some("trace".to_string()),
        ..delta()
    };

    let events = accumulator
        .process_chunk(chunk(mirrored, Some("stop")))
        .expect("chunk processes");

    assert_eq!(events[0].as_reasoning(), Some("trace"));
}

#[test]
fn the_openrouter_channel_is_read_when_deepseeks_is_absent() {
    let mut accumulator = StreamAccumulator::new().capture_reasoning(true);
    let openrouter = OpenAIDelta {
        reasoning: Some("or-trace".to_string()),
        ..delta()
    };

    let events = accumulator
        .process_chunk(chunk(openrouter, Some("stop")))
        .expect("chunk processes");

    assert_eq!(events[0].as_reasoning(), Some("or-trace"));
}

#[test]
fn reasoning_arrives_fragment_by_fragment_in_order() {
    let mut accumulator = StreamAccumulator::new().capture_reasoning(true);
    let mut reasoning = Vec::new();
    for fragment in ["one ", "two ", "three"] {
        let events = accumulator
            .process_chunk(chunk(
                OpenAIDelta {
                    reasoning_content: Some(fragment.to_string()),
                    ..delta()
                },
                None,
            ))
            .expect("chunk processes");
        reasoning.extend(
            events
                .iter()
                .filter_map(|event| event.as_reasoning().map(str::to_string)),
        );
    }

    assert_eq!(reasoning, vec!["one ", "two ", "three"]);
    assert_eq!(reasoning.concat(), "one two three");
}

#[test]
fn a_reasoning_only_response_produces_no_content_block() {
    let mut accumulator = StreamAccumulator::new().capture_reasoning(true);
    accumulator
        .process_chunk(chunk(
            OpenAIDelta {
                reasoning_content: Some("all budget spent here".to_string()),
                ..delta()
            },
            Some("length"),
        ))
        .expect("chunk processes");

    let events = accumulator.finalize().expect("finalize");
    assert!(
        events.iter().all(|event| event.as_block().is_none()),
        "{events:?}"
    );
}

// ============================================================================
// TOOL CALLS
// ============================================================================

#[test]
fn parallel_tool_calls_are_emitted_in_index_order() {
    let mut accumulator = StreamAccumulator::new();

    // Deliver the higher index first: emission order must follow the index, not arrival.
    for (index, id, name) in [(2, "call_c", "third"), (0, "call_a", "first"), (1, "call_b", "second")] {
        accumulator
            .process_chunk(chunk(tool_call_delta(index, id, name, "{}"), None))
            .expect("chunk processes");
    }

    let events = accumulator.finalize().expect("finalize");
    let names: Vec<&str> = events
        .iter()
        .filter_map(StreamEvent::as_block)
        .filter_map(|block| match block {
            ContentBlock::ToolUse(tool_use) => Some(tool_use.name()),
            _ => None,
        })
        .collect();

    assert_eq!(names, ["first", "second", "third"]);
}

#[test]
fn a_tool_call_missing_its_name_is_dropped() {
    let mut accumulator = StreamAccumulator::new();
    accumulator
        .process_chunk(chunk(
            OpenAIDelta {
                tool_calls: Some(vec![OpenAIToolCallDelta {
                    index: 0,
                    id: Some("call_1".to_string()),
                    call_type: None,
                    function: None,
                }]),
                ..delta()
            },
            None,
        ))
        .expect("chunk processes");

    let events = accumulator.finalize().expect("finalize");
    assert!(
        events.iter().all(|event| event.as_block().is_none()),
        "{events:?}"
    );
}

#[test]
fn truncated_tool_arguments_surface_as_a_stream_error() {
    let mut accumulator = StreamAccumulator::new();
    accumulator
        .process_chunk(chunk(
            tool_call_delta(0, "call_1", "search", "{\"q\": \"unter"),
            None,
        ))
        .expect("chunk processes");

    let error = accumulator.finalize().expect_err("invalid JSON must error");
    // The message names the tool, so a response with several calls says which one failed.
    assert!(
        error
            .to_string()
            .contains("Failed to parse tool call arguments for 'search'"),
        "unexpected error: {error}"
    );
}
