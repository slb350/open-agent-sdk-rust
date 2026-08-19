//! Items yielded by the model stream: content, reasoning, and the terminating finish reason.

use super::ContentBlock;

/// Why the model stopped generating.
///
/// OpenAI-compatible servers report this as a `finish_reason` string on the final streaming
/// chunk. The SDK maps the well-known values onto variants and preserves anything else
/// verbatim in [`FinishReason::Other`], so a provider-specific reason is never silently
/// flattened into a generic one.
///
/// # Why `Unspecified` exists
///
/// `finish_reason` is optional in practice. llama.cpp, vLLM, and several local gateways stream
/// content and then close the connection (or send `data: [DONE]`) with `finish_reason` still
/// null. Reporting that as [`FinishReason::Stop`] would claim the model finished cleanly when
/// the SDK has no evidence either way, so it is reported as
/// [`FinishReason::Unspecified`] instead — a distinct, checkable state.
///
/// # Examples
///
/// The distinction that matters for callers parsing structured output:
///
/// ```rust
/// use open_agent::FinishReason;
///
/// // A truncated response is worth retrying with a larger budget.
/// assert!(FinishReason::Length.is_truncated());
/// // A clean stop that produced unparseable output is a model behaviour problem.
/// assert!(!FinishReason::Stop.is_truncated());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishReason {
    /// Generation completed naturally (`"stop"`).
    Stop,

    /// Generation was cut off at the token limit (`"length"`).
    ///
    /// The content delivered before this point is a prefix of what the model intended to say.
    Length,

    /// The model finished in order to call tools (`"tool_calls"`).
    ToolCalls,

    /// Generation was halted by a content filter (`"content_filter"`).
    ContentFilter,

    /// A reason the SDK does not recognise, preserved exactly as the server sent it.
    Other(String),

    /// The SDK's automatic tool-execution loop stopped at `max_tool_iterations`.
    ///
    /// Unlike every other variant this does not come from the server: generation was cut
    /// short by the client, and the model's own `finish_reason` for the last round would
    /// have been `ToolCalls` — an accurate answer to a different question than "why did this
    /// operation stop?". Reported only by
    /// [`Client::finish_reason()`](crate::Client::finish_reason) in auto-execution mode; it
    /// never appears in a [`StreamEvent::Finish`], and [`FinishReason::from_wire`] never
    /// produces it.
    MaxToolIterations,

    /// The stream ended without the server ever reporting a reason.
    ///
    /// This is not an error — it is the normal behaviour of several OpenAI-compatible
    /// servers. It means "no information", which is deliberately distinct from
    /// [`FinishReason::Stop`].
    Unspecified,
}

impl FinishReason {
    /// Maps a raw `finish_reason` string from the wire onto a variant.
    ///
    /// Matching is ASCII-case-insensitive because servers are inconsistent about casing.
    /// Unrecognised values are preserved verbatim (original casing intact) in
    /// [`FinishReason::Other`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use open_agent::FinishReason;
    ///
    /// assert_eq!(FinishReason::from_wire("length"), FinishReason::Length);
    /// assert_eq!(
    ///     FinishReason::from_wire("ERR"),
    ///     FinishReason::Other("ERR".to_string())
    /// );
    /// ```
    pub fn from_wire(raw: &str) -> Self {
        match raw.to_ascii_lowercase().as_str() {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_calls" => Self::ToolCalls,
            "content_filter" => Self::ContentFilter,
            _ => Self::Other(raw.to_string()),
        }
    }

    /// Returns the canonical wire string for this reason.
    ///
    /// [`FinishReason::Unspecified`] has no wire representation — it is reported as
    /// `"unspecified"` so it can be logged without being mistaken for `"stop"`.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::ContentFilter => "content_filter",
            Self::Other(raw) => raw,
            Self::MaxToolIterations => "max_tool_iterations",
            Self::Unspecified => "unspecified",
        }
    }

    /// Returns `true` when generation was cut short by the token limit.
    ///
    /// This is the signal that a partial or unparseable response is the SDK caller's budget
    /// problem rather than a model that refused to answer in the requested format.
    pub fn is_truncated(&self) -> bool {
        matches!(self, Self::Length)
    }
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One item in the stream returned by [`query()`](crate::query).
///
/// Before 0.8.0 the stream yielded bare [`ContentBlock`]s, which left no room for anything
/// that is not content — most importantly the reason generation stopped. `StreamEvent` makes
/// that explicit: content arrives as [`StreamEvent::Block`], and the stream always ends with
/// exactly one [`StreamEvent::Finish`].
///
/// # Guarantees
///
/// - Exactly one [`StreamEvent::Finish`] is emitted per stream, and it is the final event.
/// - [`StreamEvent::Reasoning`] is emitted only when
///   [`AgentOptions::include_reasoning`](crate::AgentOptions::include_reasoning) is enabled,
///   and never carries text that also appears in a [`ContentBlock::Text`].
///
/// The enum is `#[non_exhaustive]`: future channels can be added without another breaking
/// release, so match with a `_` arm.
///
/// # Examples
///
/// ```rust,no_run
/// use futures::StreamExt;
/// use open_agent::{AgentOptions, ContentBlock, FinishReason, StreamEvent, query};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let options = AgentOptions::builder()
///     .model("deepseek-reasoner")
///     .base_url("http://localhost:1234/v1")
///     .build()?;
///
/// let mut answer = String::new();
/// let mut stream = query("Reply with JSON.", &options).await?;
///
/// while let Some(event) = stream.next().await {
///     match event? {
///         StreamEvent::Block(ContentBlock::Text(text)) => answer.push_str(&text.text),
///         StreamEvent::Finish(FinishReason::Length) => {
///             // Truncated at the token cap: retry with a larger budget rather than
///             // treating the unparseable body as a refusal.
///         }
///         _ => {}
///     }
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StreamEvent {
    /// A completed content block: assistant text, or a fully assembled tool call.
    Block(ContentBlock),

    /// Accumulated reasoning/chain-of-thought text from the model's side channel.
    ///
    /// Opt in with
    /// [`AgentOptions::builder().include_reasoning(true)`](crate::AgentOptionsBuilder::include_reasoning).
    /// This text is never merged into a [`ContentBlock::Text`] and never enters conversation
    /// history.
    Reasoning(String),

    /// The stream has ended; carries why generation stopped.
    Finish(FinishReason),
}

impl StreamEvent {
    /// Returns the content block if this event carries one.
    pub fn as_block(&self) -> Option<&ContentBlock> {
        match self {
            Self::Block(block) => Some(block),
            _ => None,
        }
    }

    /// Consumes the event, returning its content block if it carries one.
    ///
    /// This is the shortest migration path from the pre-0.8.0 block stream:
    ///
    /// ```rust,no_run
    /// # use futures::StreamExt;
    /// # use open_agent::{AgentOptions, ContentBlock, query};
    /// # async fn example(options: AgentOptions) -> Result<(), Box<dyn std::error::Error>> {
    /// let mut stream = query("hi", &options).await?;
    /// while let Some(event) = stream.next().await {
    ///     if let Some(ContentBlock::Text(text)) = event?.into_block() {
    ///         print!("{}", text.text);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_block(self) -> Option<ContentBlock> {
        match self {
            Self::Block(block) => Some(block),
            _ => None,
        }
    }

    /// Returns the text if this event carries a [`ContentBlock::Text`].
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Block(ContentBlock::Text(text)) => Some(&text.text),
            _ => None,
        }
    }

    /// Returns the reasoning text if this event carries one.
    pub fn as_reasoning(&self) -> Option<&str> {
        match self {
            Self::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        }
    }

    /// Returns the finish reason if this is the terminating event.
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        match self {
            Self::Finish(reason) => Some(reason),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TextBlock, ToolUseBlock};

    include!("tests/stream_event.rs");
}
