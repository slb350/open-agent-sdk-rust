//! Exact tool-content arithmetic, threshold and truncation boundaries.

mod common;

use common::message_text;
use open_agent::{
    ContentBlock, Message, MessageRole, TextBlock, ToolResultBlock, ToolUseBlock, estimate_tokens,
    is_approaching_limit, truncate_messages,
};
use serde_json::json;

#[test]
fn tool_use_blocks_accumulate_name_id_and_input_lengths() {
    let input = json!({"q": "rust"});
    assert_eq!(
        input.to_string().len(),
        12,
        "{{\"q\":\"rust\"}} is 12 chars"
    );

    let message = Message::new(
        MessageRole::Assistant,
        vec![ContentBlock::ToolUse(ToolUseBlock::new(
            "call_12", "search", input,
        ))],
    );

    // 8 role + "search" (6) + "call_12" (7) + input (12) + 16 conversation = 49 chars,
    // and 49 / 4 rounds up to 13 tokens.
    assert_eq!(estimate_tokens(&[message]), 13);
}

#[test]
fn tool_result_blocks_accumulate_id_and_content_lengths() {
    let content = json!({"hits": 3});
    assert_eq!(content.to_string().len(), 10, "{{\"hits\":3}} is 10 chars");

    let message = Message::new(
        MessageRole::User,
        vec![ContentBlock::ToolResult(ToolResultBlock::new(
            "call_12", content,
        ))],
    );

    // 8 role + "call_12" (7) + content (10) + 16 conversation = 41 chars -> 11 tokens.
    assert_eq!(estimate_tokens(&[message]), 11);
}

/// Builds a single message whose estimate is exactly 600 tokens.
///
/// 8 role + 2376 text + 16 conversation = 2400 chars, and 2400 / 4 = 600 exactly. The exact
/// figure is what lets the threshold tests below distinguish `>` from `>=`.
fn six_hundred_token_message() -> Vec<Message> {
    let messages = vec![Message::new(
        MessageRole::User,
        vec![ContentBlock::Text(TextBlock::new("x".repeat(2376)))],
    )];

    assert_eq!(estimate_tokens(&messages), 600);
    messages
}

#[test]
fn approaching_limit_scales_the_limit_by_the_margin() {
    let messages = six_hundred_token_message();

    // 1000 * 0.5 = 500, and 600 > 500. Addition (1000.5 -> 1000) or division (2000) would
    // both put the threshold above the estimate and report false.
    assert!(is_approaching_limit(&messages, 1000, 0.5));
}

#[test]
fn approaching_limit_is_strict_at_the_threshold() {
    let messages = six_hundred_token_message();

    // 1000 * 0.6 = 600, exactly the estimate. The check is `>`, not `>=`, so an estimate that
    // merely reaches the threshold is not yet "approaching" it.
    assert!(!is_approaching_limit(&messages, 1000, 0.6));
    // A slightly smaller limit does cross it.
    assert!(is_approaching_limit(&messages, 999, 0.6));
}

#[test]
fn truncation_with_keep_zero_drops_everything_but_the_system_prompt() {
    let messages = vec![
        Message::system("system"),
        Message::user("first"),
        Message::user("second"),
    ];

    let preserved = truncate_messages(&messages, 0, true);
    assert_eq!(preserved.len(), 1);
    assert_eq!(preserved[0].role, MessageRole::System);

    assert!(truncate_messages(&messages, 0, false).is_empty());
}

#[test]
fn truncation_keeps_the_most_recent_messages_after_the_system_prompt() {
    let messages = vec![
        Message::system("system"),
        Message::user("first"),
        Message::user("second"),
        Message::user("third"),
    ];

    // Identity, not just length: the tail must be the *most recent* messages, and the system
    // prompt must not be duplicated into it.
    let kept = truncate_messages(&messages, 2, true);
    assert_eq!(kept.len(), 3);
    assert_eq!(kept[0].role, MessageRole::System);
    assert_eq!(message_text(&kept[1]), "second");
    assert_eq!(message_text(&kept[2]), "third");

    // Without system preservation the tail stands alone.
    let tail = truncate_messages(&messages, 2, false);
    assert_eq!(tail.len(), 2);
    assert_eq!(message_text(&tail[0]), "second");
    assert_eq!(message_text(&tail[1]), "third");
}

#[test]
fn truncation_returns_everything_when_keep_covers_the_history() {
    let messages = vec![Message::system("system"), Message::user("only")];

    assert_eq!(truncate_messages(&messages, 2, true).len(), 2);
    assert_eq!(truncate_messages(&messages, 5, true).len(), 2);
    assert!(truncate_messages(&[], 5, true).is_empty());
}
