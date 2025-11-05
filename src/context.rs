//! Context management utilities for manual history management
//!
//! This module provides low-level helpers for managing conversation history.
//! These are opt-in utilities - nothing is automatic. You decide when and how
//! to manage context.
//!
//! # Features
//!
//! - Token estimation (character-based approximation)
//! - Message truncation with system prompt preservation
//! - Manual history management patterns
//!
//! # Examples
//!
//! ```rust
//! use open_agent::{estimate_tokens, truncate_messages};
//!
//! // Estimate tokens
//! let messages = vec![/* your messages */];
//! let tokens = estimate_tokens(&messages);
//! println!("Estimated tokens: {}", tokens);
//!
//! // Truncate when needed
//! if tokens > 28000 {
//!     let truncated = truncate_messages(&messages, 10, true);
//!     // Use truncated messages...
//! }
//! ```

use crate::types::Message;

/// Estimate token count for message list
///
/// Uses character-based approximation (1 token ≈ 4 characters).
///
/// # Arguments
///
/// * `messages` - List of messages to estimate tokens for
///
/// # Returns
///
/// Estimated token count
///
/// # Note
///
/// This is an APPROXIMATION. Actual token counts vary by model family:
/// - GPT models: ~70-85% accurate (different tokenizers)
/// - Llama, Qwen, Mistral: ~70-85% accurate
/// - Always include 10-20% safety margin when checking limits
///
/// For more accurate estimation, consider using tiktoken bindings
/// (not included to keep dependencies minimal).
///
/// # Examples
///
/// ```rust
/// use open_agent::{Message, MessageRole, estimate_tokens};
///
/// let messages = vec![
///     Message::system("You are a helpful assistant"),
///     Message::user("Hello!"),
/// ];
///
/// let tokens = estimate_tokens(&messages);
/// println!("Estimated tokens: {}", tokens);
///
/// // Check if approaching context limit
/// if tokens > 28000 {
///     println!("Need to truncate!");
/// }
/// ```
pub fn estimate_tokens(messages: &[Message]) -> usize {
    // Character-based approximation: 1 token ≈ 4 characters
    // This is a conservative estimate that works across model families

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
                crate::types::ContentBlock::ToolUse(tool) => {
                    // Tool calls add significant overhead
                    total_chars += tool.name.len();
                    total_chars += tool.id.len();
                    total_chars += tool.input.to_string().len();
                }
                crate::types::ContentBlock::ToolResult(result) => {
                    // Tool results add overhead
                    total_chars += result.tool_use_id.len();
                    total_chars += result.content.to_string().len();
                }
            }
        }
    }

    // Add conversation-level overhead (~2-4 tokens)
    total_chars += 16;

    // Convert characters to tokens (4 chars ≈ 1 token, round up for safety)
    (total_chars + 3) / 4 // Ceiling division
}

/// Truncate message history, keeping recent messages
///
/// Always preserves the system prompt (if present) and keeps the most
/// recent N messages. This is a simple truncation - it does NOT attempt
/// to preserve tool chains or important context.
///
/// # Arguments
///
/// * `messages` - List of messages to truncate
/// * `keep` - Number of recent messages to keep (default: 10)
/// * `preserve_system` - Keep system message if present (default: true)
///
/// # Returns
///
/// Truncated message list (new Vec, original unchanged)
///
/// # Examples
///
/// ```rust
/// use open_agent::{Message, Client, truncate_messages, estimate_tokens};
///
/// # async fn example(mut client: Client) {
/// // Manual truncation when needed
/// let tokens = estimate_tokens(client.history());
/// if tokens > 28000 {
///     let truncated = truncate_messages(client.history(), 10, true);
///     *client.history_mut() = truncated;
/// }
/// # }
/// ```
///
/// # Note
///
/// This is a SIMPLE truncation. For domain-specific needs (e.g.,
/// preserving tool call chains, keeping important context), implement
/// your own logic or use this as a starting point.
///
/// Warning: Truncating mid-conversation may remove context that the
/// model needs to properly respond. Use judiciously at natural breakpoints.
pub fn truncate_messages(messages: &[Message], keep: usize, preserve_system: bool) -> Vec<Message> {
    if messages.is_empty() {
        return Vec::new();
    }

    if messages.len() <= keep {
        return messages.to_vec();
    }

    // Check if first message is system prompt
    let has_system = preserve_system
        && !messages.is_empty()
        && messages[0].role == crate::types::MessageRole::System;

    if has_system {
        // Keep system + last N messages
        let mut result = vec![messages[0].clone()];
        if keep > 0 && messages.len() > 1 {
            let start = messages.len().saturating_sub(keep);
            result.extend_from_slice(&messages[start..]);
        }
        result
    } else {
        // Just keep last N messages
        if keep > 0 {
            let start = messages.len().saturating_sub(keep);
            messages[start..].to_vec()
        } else {
            Vec::new()
        }
    }
}

/// Check if history is approaching a token limit
///
/// Convenience function that combines estimation with a threshold check.
///
/// # Arguments
///
/// * `messages` - Messages to check
/// * `limit` - Token limit (e.g., 32000 for a 32k context window)
/// * `margin` - Safety margin as a percentage (default: 0.9 = 90%)
///
/// # Returns
///
/// `true` if estimated tokens exceed limit * margin
///
/// # Examples
///
/// ```rust
/// use open_agent::{is_approaching_limit, Message};
///
/// # fn example(messages: Vec<Message>) {
/// if is_approaching_limit(&messages, 32000, 0.9) {
///     println!("Time to truncate!");
/// }
/// # }
/// ```
pub fn is_approaching_limit(messages: &[Message], limit: usize, margin: f32) -> bool {
    let estimated = estimate_tokens(messages);
    let threshold = (limit as f32 * margin) as usize;
    estimated > threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentBlock, Message, MessageRole, TextBlock};

    #[test]
    fn test_estimate_tokens_empty() {
        let messages: Vec<Message> = vec![];
        assert_eq!(estimate_tokens(&messages), 0);
    }

    #[test]
    fn test_estimate_tokens_simple() {
        let messages = vec![Message::new(
            MessageRole::User,
            vec![ContentBlock::Text(TextBlock::new("Hello world"))],
        )];

        let tokens = estimate_tokens(&messages);
        // "Hello world" = 11 chars + overhead ≈ 5-8 tokens
        assert!(tokens >= 3 && tokens <= 10);
    }

    #[test]
    fn test_truncate_messages_empty() {
        let messages: Vec<Message> = vec![];
        let truncated = truncate_messages(&messages, 10, true);
        assert_eq!(truncated.len(), 0);
    }

    #[test]
    fn test_truncate_messages_preserve_system() {
        let messages = vec![
            Message::system("System prompt"),
            Message::user("Message 1"),
            Message::user("Message 2"),
            Message::user("Message 3"),
            Message::user("Message 4"),
        ];

        let truncated = truncate_messages(&messages, 2, true);

        // Should have system + last 2 = 3 messages
        assert_eq!(truncated.len(), 3);
        assert_eq!(truncated[0].role, MessageRole::System);
    }

    #[test]
    fn test_truncate_messages_no_preserve() {
        let messages = vec![
            Message::system("System prompt"),
            Message::user("Message 1"),
            Message::user("Message 2"),
            Message::user("Message 3"),
        ];

        let truncated = truncate_messages(&messages, 2, false);

        // Should have only last 2 messages
        assert_eq!(truncated.len(), 2);
        assert_eq!(truncated[0].role, MessageRole::User);
    }

    #[test]
    fn test_truncate_messages_keep_all() {
        let messages = vec![Message::user("Message 1"), Message::user("Message 2")];

        let truncated = truncate_messages(&messages, 10, true);
        assert_eq!(truncated.len(), 2);
    }

    #[test]
    fn test_is_approaching_limit() {
        let messages = vec![Message::user("x".repeat(1000))];

        // ~250 tokens, should not exceed 90% of 1000
        assert!(!is_approaching_limit(&messages, 1000, 0.9));

        // Should exceed 90% of 200
        assert!(is_approaching_limit(&messages, 200, 0.9));
    }
}
