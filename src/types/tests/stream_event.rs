#[test]
fn finish_reason_maps_every_well_known_wire_value() {
    assert_eq!(FinishReason::from_wire("stop"), FinishReason::Stop);
    assert_eq!(FinishReason::from_wire("length"), FinishReason::Length);
    assert_eq!(FinishReason::from_wire("tool_calls"), FinishReason::ToolCalls);
    assert_eq!(
        FinishReason::from_wire("content_filter"),
        FinishReason::ContentFilter
    );
}

#[test]
fn finish_reason_matching_ignores_case() {
    assert_eq!(FinishReason::from_wire("STOP"), FinishReason::Stop);
    assert_eq!(FinishReason::from_wire("Length"), FinishReason::Length);
}

#[test]
fn an_unknown_reason_keeps_its_original_casing() {
    assert_eq!(
        FinishReason::from_wire("ERR_Budget"),
        FinishReason::Other("ERR_Budget".to_string())
    );
}

#[test]
fn an_empty_reason_is_not_mistaken_for_stop() {
    assert_eq!(
        FinishReason::from_wire(""),
        FinishReason::Other(String::new())
    );
}

#[test]
fn finish_reason_round_trips_through_as_str() {
    for reason in [
        FinishReason::Stop,
        FinishReason::Length,
        FinishReason::ToolCalls,
        FinishReason::ContentFilter,
    ] {
        assert_eq!(FinishReason::from_wire(reason.as_str()), reason);
    }
}

#[test]
fn unspecified_renders_distinctly_from_stop() {
    assert_eq!(FinishReason::Unspecified.as_str(), "unspecified");
    assert_ne!(
        FinishReason::Unspecified.as_str(),
        FinishReason::Stop.as_str()
    );
    assert_eq!(FinishReason::Other("x".to_string()).as_str(), "x");
}

#[test]
fn display_matches_as_str() {
    assert_eq!(FinishReason::Length.to_string(), "length");
    assert_eq!(FinishReason::Unspecified.to_string(), "unspecified");
    assert_eq!(FinishReason::Other("weird".to_string()).to_string(), "weird");
}

#[test]
fn only_length_counts_as_truncated() {
    assert!(FinishReason::Length.is_truncated());
    for reason in [
        FinishReason::Stop,
        FinishReason::ToolCalls,
        FinishReason::ContentFilter,
        FinishReason::Unspecified,
        FinishReason::Other("length_ish".to_string()),
    ] {
        assert!(!reason.is_truncated(), "{reason} must not be truncated");
    }
}

#[test]
fn stream_event_accessors_only_answer_for_their_own_variant() {
    let text = StreamEvent::Block(ContentBlock::Text(TextBlock::new("hello")));
    let tool = StreamEvent::Block(ContentBlock::ToolUse(ToolUseBlock::new(
        "id",
        "name",
        serde_json::json!({}),
    )));
    let reasoning = StreamEvent::Reasoning("thinking".to_string());
    let finish = StreamEvent::Finish(FinishReason::Stop);

    assert_eq!(text.as_text(), Some("hello"));
    assert!(text.as_reasoning().is_none());
    assert!(text.finish_reason().is_none());
    assert!(text.as_block().is_some());

    // A tool-use block is a block, but it is not text.
    assert!(tool.as_block().is_some());
    assert!(tool.as_text().is_none());

    assert_eq!(reasoning.as_reasoning(), Some("thinking"));
    assert!(reasoning.as_block().is_none());
    assert!(reasoning.as_text().is_none());

    assert_eq!(finish.finish_reason(), Some(&FinishReason::Stop));
    assert!(finish.as_block().is_none());
}

#[test]
fn into_block_yields_the_block_and_nothing_else() {
    let block = StreamEvent::Block(ContentBlock::Text(TextBlock::new("hi")))
        .into_block()
        .expect("a block event yields its block");
    assert!(matches!(block, ContentBlock::Text(text) if text.text == "hi"));

    assert!(StreamEvent::Reasoning("r".to_string()).into_block().is_none());
    assert!(
        StreamEvent::Finish(FinishReason::Unspecified)
            .into_block()
            .is_none()
    );
}
