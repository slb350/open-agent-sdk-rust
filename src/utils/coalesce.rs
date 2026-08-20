//! Joining the text fragments a stream delivered back into whole blocks.
//!
//! Callers see text as it arrives, one [`StreamEvent`](crate::StreamEvent) per fragment. What
//! goes into conversation history is the other shape: a turn the model produced in one breath
//! is one block. Writing the fragments straight to history would replay a dozen text blocks to
//! the server on the next request in place of the one it sent.
//!
//! Only *adjacent* text is joined. A text block on either side of a tool call is a separate
//! block, because the tool call sits between them in the turn.

use crate::types::ContentBlock;

/// Joins runs of adjacent [`ContentBlock::Text`], leaving every other block untouched.
///
/// Takes a slice because the common caller still needs its fragments afterwards: the client
/// hands them to the caller one at a time while writing the joined turn to history.
///
/// # Examples
///
/// ```
/// use open_agent::{ContentBlock, TextBlock, coalesce_text_blocks};
///
/// let fragments = vec![
///     ContentBlock::Text(TextBlock::new("Hel")),
///     ContentBlock::Text(TextBlock::new("lo")),
/// ];
///
/// let joined = coalesce_text_blocks(&fragments);
///
/// assert_eq!(joined.len(), 1);
/// match &joined[0] {
///     ContentBlock::Text(text) => assert_eq!(text.text, "Hello"),
///     _ => unreachable!(),
/// }
/// ```
pub fn coalesce_text_blocks(blocks: &[ContentBlock]) -> Vec<ContentBlock> {
    // Sized for what comes out, not what goes in: a response with no tool calls is one block
    // however many fragments carried it.
    let mut joined: Vec<ContentBlock> = Vec::new();

    for block in blocks {
        match (joined.last_mut(), block) {
            (Some(ContentBlock::Text(previous)), ContentBlock::Text(next)) => {
                previous.text.push_str(&next.text);
            }
            (_, other) => joined.push(other.clone()),
        }
    }

    joined
}

#[cfg(test)]
mod tests;
