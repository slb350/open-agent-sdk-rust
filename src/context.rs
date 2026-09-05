//! Opt-in estimates and truncation for conversation history.
//!
//! The SDK never applies these helpers automatically. Token estimates are fixed
//! heuristics, not tokenizer output or a guaranteed upper bound.

use crate::types::Message;
use std::fmt::{self, Write};

fn json_len(value: &serde_json::Value) -> usize {
    struct ByteCount(usize);

    impl Write for ByteCount {
        fn write_str(&mut self, text: &str) -> fmt::Result {
            self.0 += text.len();
            Ok(())
        }
    }

    let mut count = ByteCount(0);
    write!(count, "{value}").expect("counting JSON bytes cannot fail");
    count.0
}

/// Estimates tokens using UTF-8 byte lengths and fixed image allowances.
///
/// Text, tool names/IDs, and serialized tool JSON contribute one token per four
/// bytes, rounded up after adding message/conversation overhead. Images contribute
/// 85 (`Low`), 300 (`High`), or 200 (`Auto`) tokens before overhead. Empty history
/// estimates to zero.
///
/// These heuristics do not account for the provider's tokenizer, image dimensions,
/// or model-specific costs. Use a provider tokenizer when exact limits matter.
pub fn estimate_tokens(messages: &[Message]) -> usize {
    // String lengths are UTF-8 bytes; divide the combined estimate only once.

    if messages.is_empty() {
        return 0;
    }

    let mut total_chars = 0;

    for message in messages {
        // Count role overhead (approximately 1-2 tokens)
        total_chars += 8; // ~2 tokens for role formatting

        // Count content
        for block in &message.content {
            match block {
                crate::types::ContentBlock::Text(text) => {
                    total_chars += text.text.len();
                }
                crate::types::ContentBlock::Image(image) => {
                    // Fixed allowances because image dimensions and tokenizer are unavailable.
                    use crate::types::ImageDetail;
                    let token_estimate = match image.detail() {
                        ImageDetail::Low => 85 * 4,   // Fixed low-detail allowance
                        ImageDetail::High => 300 * 4, // Fixed heuristic; dimensions are unavailable
                        ImageDetail::Auto => 200 * 4, // Middle ground default
                    };
                    total_chars += token_estimate;
                }
                crate::types::ContentBlock::ToolUse(tool) => {
                    // Tool calls add significant overhead
                    total_chars += tool.name().len();
                    total_chars += tool.id().len();
                    total_chars += json_len(tool.input());
                }
                crate::types::ContentBlock::ToolResult(result) => {
                    // Tool results add overhead
                    total_chars += result.tool_use_id().len();
                    total_chars += json_len(result.content());
                }
            }
        }
    }

    // Add conversation-level overhead (~2-4 tokens)
    total_chars += 16;

    // Round up after adding all byte-equivalent allowances.
    total_chars.div_ceil(4)
}

/// Clones the most recent `keep` messages, optionally preserving the first system message.
///
/// When `preserve_system` is true and the first message is a system message, it is
/// kept in addition to the tail. The original slice is unchanged. This does not
/// preserve tool-call/result pairs; callers must choose suitable truncation points.
///
/// ```rust
/// use open_agent::{Client, is_approaching_limit, truncate_messages};
///
/// fn trim_history(client: &mut Client) {
///     if is_approaching_limit(client.history(), 32_000, 0.9) {
///         *client.history_mut() = truncate_messages(client.history(), 10, true);
///     }
/// }
/// ```
pub fn truncate_messages(messages: &[Message], keep: usize, preserve_system: bool) -> Vec<Message> {
    if messages.is_empty() {
        return Vec::new();
    }

    if messages.len() <= keep {
        return messages.to_vec();
    }

    // Check if first message is system prompt. The empty case returned above, so indexing is
    // safe here without a further emptiness check.
    let has_system = preserve_system && messages[0].role == crate::types::MessageRole::System;

    // The early returns above guarantee `keep < messages.len()`, so `start` is at least 1 and
    // the tail slice never re-includes the system prompt. `keep == 0` yields `start ==
    // messages.len()`, i.e. an empty tail — no separate zero case is needed.
    let start = messages.len() - keep;

    if has_system {
        // Keep system + last N messages
        let mut result = vec![messages[0].clone()];
        result.extend_from_slice(&messages[start..]);
        result
    } else {
        // Just keep last N messages
        messages[start..].to_vec()
    }
}

/// Returns whether [`estimate_tokens`] is strictly greater than `limit * margin`.
///
/// The floating-point threshold is converted to `usize` before comparison.
/// For example, `margin = 0.9` checks against 90% of the supplied limit.
pub fn is_approaching_limit(messages: &[Message], limit: usize, margin: f32) -> bool {
    let estimated = estimate_tokens(messages);
    let threshold = (limit as f32 * margin) as usize;
    estimated > threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentBlock, ImageBlock, ImageDetail, Message, MessageRole};

    #[test]
    fn text_estimates_include_overhead_and_round_up() {
        assert_eq!(estimate_tokens(&[]), 0);
        // 11 text bytes + 8 message bytes + 16 conversation bytes = 35; ceil(35 / 4) = 9.
        assert_eq!(estimate_tokens(&[Message::user("Hello world")]), 9);
    }

    #[test]
    fn image_estimates_include_detail_allowance_and_message_overhead() {
        for (detail, expected) in [
            (ImageDetail::Low, 91),
            (ImageDetail::High, 306),
            (ImageDetail::Auto, 206),
        ] {
            let image = ImageBlock::from_url("https://example.com/img.jpg")
                .unwrap()
                .with_detail(detail);
            let message = Message::new(MessageRole::User, vec![ContentBlock::Image(image)]);
            assert_eq!(estimate_tokens(&[message]), expected, "{detail:?}");
        }
    }
}
