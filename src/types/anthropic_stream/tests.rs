//! Parsing tests for the Anthropic streaming vocabulary.
//!
//! Every payload here is copied from the shape the API documentation publishes, so a rename
//! on the wire fails a test rather than silently deserializing into a catch-all.

use super::*;

/// Parses one event, failing the test with the payload when it does not deserialize.
fn parse(payload: &str) -> AnthropicEvent {
    serde_json::from_str(payload).unwrap_or_else(|error| panic!("{payload} should parse: {error}"))
}

#[test]
fn message_start_parses_and_its_payload_is_ignored() {
    let event = parse(
        r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant",
            "content":[],"model":"claude-opus-5","stop_reason":null,"stop_sequence":null,
            "usage":{"input_tokens":25,"output_tokens":1}}}"#,
    );

    assert!(
        matches!(event, AnthropicEvent::MessageStart {}),
        "{event:?}"
    );
}

#[test]
fn a_text_block_opens_with_its_index() {
    let event = parse(
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
    );

    match event {
        AnthropicEvent::ContentBlockStart {
            index,
            content_block: AnthropicBlockStart::Text { text },
        } => {
            assert_eq!(index, 0);
            assert!(text.is_empty());
        }
        other => panic!("expected a text block start, got {other:?}"),
    }
}

#[test]
fn a_text_block_keeps_text_present_at_open() {
    // A server that front-loads the first fragment here rather than sending a delta for it.
    let event = parse(
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"He"}}"#,
    );

    match event {
        AnthropicEvent::ContentBlockStart {
            content_block: AnthropicBlockStart::Text { text },
            ..
        } => assert_eq!(text, "He"),
        other => panic!("expected a text block start, got {other:?}"),
    }
}

#[test]
fn a_text_block_start_without_a_text_field_still_parses() {
    let event =
        parse(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text"}}"#);

    assert!(
        matches!(
            event,
            AnthropicEvent::ContentBlockStart {
                content_block: AnthropicBlockStart::Text { .. },
                ..
            }
        ),
        "{event:?}"
    );
}

#[test]
fn a_thinking_block_is_its_own_kind() {
    let event = parse(
        r#"{"type":"content_block_start","index":0,
            "content_block":{"type":"thinking","thinking":""}}"#,
    );

    assert!(
        matches!(
            event,
            AnthropicEvent::ContentBlockStart {
                content_block: AnthropicBlockStart::Thinking { .. },
                ..
            }
        ),
        "{event:?}"
    );
}

#[test]
fn a_redacted_thinking_block_parses_without_reading_its_payload() {
    let event = parse(
        r#"{"type":"content_block_start","index":0,
            "content_block":{"type":"redacted_thinking","data":"EncryptedBlob"}}"#,
    );

    assert!(
        matches!(
            event,
            AnthropicEvent::ContentBlockStart {
                content_block: AnthropicBlockStart::RedactedThinking {},
                ..
            }
        ),
        "{event:?}"
    );
}

#[test]
fn a_tool_use_block_carries_its_id_and_name() {
    let event = parse(
        r#"{"type":"content_block_start","index":1,
            "content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}"#,
    );

    match event {
        AnthropicEvent::ContentBlockStart {
            index,
            content_block: AnthropicBlockStart::ToolUse { id, name },
        } => {
            assert_eq!(index, 1);
            assert_eq!(id, "toolu_1");
            assert_eq!(name, "get_weather");
        }
        other => panic!("expected a tool_use block start, got {other:?}"),
    }
}

#[test]
fn an_unrecognised_block_kind_is_not_mistaken_for_text() {
    let event = parse(
        r#"{"type":"content_block_start","index":0,
            "content_block":{"type":"server_tool_use","id":"x","name":"y"}}"#,
    );

    assert!(
        matches!(
            event,
            AnthropicEvent::ContentBlockStart {
                content_block: AnthropicBlockStart::Unknown,
                ..
            }
        ),
        "{event:?}"
    );
}

#[test]
fn a_text_delta_carries_its_fragment() {
    let event = parse(
        r#"{"type":"content_block_delta","index":0,
            "delta":{"type":"text_delta","text":"ello frien"}}"#,
    );

    match event {
        AnthropicEvent::ContentBlockDelta {
            index,
            delta: AnthropicDelta::TextDelta { text },
        } => {
            assert_eq!(index, 0);
            assert_eq!(text, "ello frien");
        }
        other => panic!("expected a text delta, got {other:?}"),
    }
}

#[test]
fn a_thinking_delta_reads_the_thinking_field_not_the_text_field() {
    let event = parse(
        r#"{"type":"content_block_delta","index":0,
            "delta":{"type":"thinking_delta","thinking":"1071 = 2 x 462 + 147"}}"#,
    );

    match event {
        AnthropicEvent::ContentBlockDelta {
            delta: AnthropicDelta::ThinkingDelta { thinking },
            ..
        } => assert_eq!(thinking, "1071 = 2 x 462 + 147"),
        other => panic!("expected a thinking delta, got {other:?}"),
    }
}

#[test]
fn a_signature_delta_parses_and_is_carried_as_nothing() {
    let event = parse(
        r#"{"type":"content_block_delta","index":0,
            "delta":{"type":"signature_delta","signature":"EqQBCgIYAhIM1gbcDa9GJwZA"}}"#,
    );

    assert!(
        matches!(
            event,
            AnthropicEvent::ContentBlockDelta {
                delta: AnthropicDelta::SignatureDelta {},
                ..
            }
        ),
        "{event:?}"
    );
}

#[test]
fn an_input_json_delta_carries_its_partial_json() {
    let event = parse(
        r#"{"type":"content_block_delta","index":1,
            "delta":{"type":"input_json_delta","partial_json":"{\"location\": \"San Fra"}}"#,
    );

    match event {
        AnthropicEvent::ContentBlockDelta {
            delta: AnthropicDelta::InputJsonDelta { partial_json },
            ..
        } => assert_eq!(partial_json, r#"{"location": "San Fra"#),
        other => panic!("expected an input_json delta, got {other:?}"),
    }
}

#[test]
fn an_unrecognised_delta_type_does_not_fail_the_stream() {
    let event = parse(
        r#"{"type":"content_block_delta","index":0,
            "delta":{"type":"citations_delta","citation":{}}}"#,
    );

    assert!(
        matches!(
            event,
            AnthropicEvent::ContentBlockDelta {
                delta: AnthropicDelta::Unknown,
                ..
            }
        ),
        "{event:?}"
    );
}

#[test]
fn a_block_stop_carries_its_index() {
    let event = parse(r#"{"type":"content_block_stop","index":2}"#);

    match event {
        AnthropicEvent::ContentBlockStop { index } => assert_eq!(index, 2),
        other => panic!("expected a block stop, got {other:?}"),
    }
}

#[test]
fn a_message_delta_carries_the_stop_reason() {
    let event = parse(
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},
            "usage":{"output_tokens":15}}"#,
    );

    match event {
        AnthropicEvent::MessageDelta { delta } => {
            assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
        }
        other => panic!("expected a message delta, got {other:?}"),
    }
}

#[test]
fn a_message_delta_with_no_stop_reason_yet_parses_as_none() {
    let event = parse(r#"{"type":"message_delta","delta":{"stop_sequence":null}}"#);

    match event {
        AnthropicEvent::MessageDelta { delta } => assert!(delta.stop_reason.is_none()),
        other => panic!("expected a message delta, got {other:?}"),
    }
}

#[test]
fn message_stop_and_ping_parse() {
    assert!(
        matches!(
            parse(r#"{"type":"message_stop"}"#),
            AnthropicEvent::MessageStop {}
        ),
        "message_stop"
    );
    assert!(
        matches!(parse(r#"{"type":"ping"}"#), AnthropicEvent::Ping {}),
        "ping"
    );
}

#[test]
fn an_error_event_carries_its_kind_and_message() {
    let event =
        parse(r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#);

    match event {
        AnthropicEvent::Error { error } => {
            assert_eq!(error.error_type.as_deref(), Some("overloaded_error"));
            assert_eq!(error.message, "Overloaded");
        }
        other => panic!("expected an error, got {other:?}"),
    }
}

#[test]
fn an_unrecognised_event_type_is_ignored_rather_than_fatal() {
    let event = parse(r#"{"type":"message_limit","limit":{"remaining":3}}"#);

    assert!(matches!(event, AnthropicEvent::Unknown), "{event:?}");
}

#[test]
fn every_documented_stop_reason_maps_to_a_finish_reason() {
    let table = [
        ("end_turn", FinishReason::Stop),
        ("stop_sequence", FinishReason::Stop),
        ("max_tokens", FinishReason::Length),
        ("model_context_window_exceeded", FinishReason::Length),
        ("tool_use", FinishReason::ToolCalls),
        ("refusal", FinishReason::ContentFilter),
    ];

    for (raw, expected) in table {
        assert_eq!(anthropic_finish_reason(raw), expected, "mapping {raw}");
    }
}

#[test]
fn a_pause_turn_keeps_its_own_name() {
    // Resumable, which no existing variant means. Flattening it to `Stop` would tell a
    // caller the response finished.
    assert_eq!(
        anthropic_finish_reason("pause_turn"),
        FinishReason::Other("pause_turn".to_string())
    );
}

#[test]
fn an_unknown_stop_reason_is_preserved_verbatim() {
    assert_eq!(
        anthropic_finish_reason("quota_exhausted"),
        FinishReason::Other("quota_exhausted".to_string())
    );
}

#[test]
fn stop_reason_matching_is_case_insensitive_but_preserves_case_when_unknown() {
    assert_eq!(anthropic_finish_reason("END_TURN"), FinishReason::Stop);
    assert_eq!(
        anthropic_finish_reason("Weird_Reason"),
        FinishReason::Other("Weird_Reason".to_string())
    );
}

#[test]
fn an_anthropic_stop_reason_is_not_what_the_openai_mapper_would_produce() {
    // The regression this function exists for: `from_wire` files every Anthropic spelling
    // under `Other`, so a caller branching on `Length` never sees a truncation.
    assert_eq!(
        FinishReason::from_wire("max_tokens"),
        FinishReason::Other("max_tokens".to_string())
    );
    assert_eq!(anthropic_finish_reason("max_tokens"), FinishReason::Length);
}
