use super::coalesce_text_blocks;
use crate::types::{ContentBlock, TextBlock, ToolResultBlock, ToolUseBlock};
use serde_json::json;

fn text(value: &str) -> ContentBlock {
    ContentBlock::Text(TextBlock::new(value))
}

fn tool_use(id: &str) -> ContentBlock {
    ContentBlock::ToolUse(ToolUseBlock::new(id, "add", json!({})))
}

fn texts_of(blocks: &[ContentBlock]) -> Vec<&str> {
    blocks
        .iter()
        .map(|block| match block {
            ContentBlock::Text(text) => text.text.as_str(),
            ContentBlock::ToolUse(_) => "<tool_use>",
            ContentBlock::ToolResult(_) => "<tool_result>",
            _ => "<other>",
        })
        .collect()
}

#[test]
fn adjacent_text_is_joined_into_one_block() {
    let joined = coalesce_text_blocks(&[text("Hel"), text("lo, "), text("world")]);

    assert_eq!(texts_of(&joined), vec!["Hello, world"]);
}

#[test]
fn text_separated_by_a_tool_call_stays_separate() {
    // The two runs are different parts of the turn: the model spoke, called a tool, and spoke
    // again. Joining across the call would misreport the order it did those things in.
    let joined =
        coalesce_text_blocks(&[text("before"), tool_use("call_1"), text("af"), text("ter")]);

    assert_eq!(
        texts_of(&joined),
        vec!["before", "<tool_use>", "after"],
        "runs on either side of a tool call must not merge"
    );
}

#[test]
fn non_text_blocks_pass_through_untouched() {
    let joined = coalesce_text_blocks(&[
        tool_use("call_1"),
        ContentBlock::ToolResult(ToolResultBlock::new("call_1", json!("42"))),
    ]);

    assert_eq!(texts_of(&joined), vec!["<tool_use>", "<tool_result>"]);
}

#[test]
fn an_empty_sequence_stays_empty() {
    assert!(coalesce_text_blocks(&[]).is_empty());
}

#[test]
fn a_lone_text_block_is_returned_unchanged() {
    let joined = coalesce_text_blocks(&[text("solo")]);

    assert_eq!(texts_of(&joined), vec!["solo"]);
}
