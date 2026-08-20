//! Transport-safe decoding of an SSE response body into one wire protocol's events.
//!
//! The framing, the `[DONE]` sentinel and the error classification are identical for both
//! protocols, so they live once in `parse_events`; the two public entry points differ only in
//! the payload type they ask it to deserialize.

use crate::types::{AnthropicEvent, OpenAIChunk};
use crate::{Error, Result};
use eventsource_stream::{EventStreamError, Eventsource};
use futures::stream::{Stream, StreamExt};
use serde::de::DeserializeOwned;
use std::pin::Pin;

/// Parses a raw HTTP response body as a Server-Sent Events (SSE) stream.
///
/// Transforms an HTTP streaming response into a stream of parsed [`OpenAIChunk`] objects.
/// This function handles the SSE protocol details, extracting JSON data from the SSE format.
///
/// # SSE Format
///
/// Server-Sent Events is a standard protocol for server-to-client streaming. The format is:
///
/// ```text
/// data: {"id":"msg_123","object":"chat.completion.chunk",...}
///
/// data: {"id":"msg_123","object":"chat.completion.chunk",...}
///
/// data: [DONE]
///
/// ```
///
/// Key characteristics:
/// - Each message starts with `data: `
/// - Messages are separated by double newlines (`\n\n`)
/// - The stream ends with `data: [DONE]`
/// - Everything after `data: ` (until newline) is the payload
///
/// # Arguments
///
/// * `body` - The raw HTTP response from the API request
///
/// # Returns
///
/// A pinned, boxed stream that yields `Result<OpenAIChunk>` for each successfully parsed event.
/// The stream is `Send` to allow use across thread boundaries.
///
/// # Error Handling
///
/// Each stream item can be an error:
/// - **HTTP errors**: Network issues, connection drops (wrapped as [`Error::Http`])
/// - **Parse errors**: Invalid JSON in the SSE data field (wrapped as [`Error::Stream`])
/// - **Protocol errors**: Invalid UTF-8 or malformed SSE fields (wrapped as [`Error::Stream`])
///
/// Errors are yielded as stream items rather than panicking the parser. Consumers should handle
/// them explicitly.
///
/// # Example Flow
///
/// ```text
/// Raw HTTP bytes: b"data: {\"id\":\"123\"}\n\ndata: [DONE]\n\n"
///        ↓
/// bytes_stream() yields arbitrary transport chunks
///        ↓
/// Eventsource buffers chunks into complete SSE events
///        ↓
/// Skip "[DONE]" and parse event data into OpenAIChunk
///        ↓
/// Stream<Result<OpenAIChunk>>
/// ```
///
/// # Protocol Notes
///
/// - **`[DONE]` sentinel**: OpenAI's SSE streams end with `data: [DONE]`. This is not valid
///   JSON, so we skip it rather than attempting to parse.
///
/// - **Chunk boundaries**: HTTP streaming can split data at arbitrary byte positions. The
///   eventsource decoder buffers partial events and emits every complete event, including
///   multiple events received in a single transport chunk.
///
/// - **UTF-8 handling**: Split multi-byte characters are buffered until complete. Invalid
///   UTF-8 is reported as a stream error instead of being replaced with lossy characters.
///
/// # Usage
///
/// ```rust,ignore
/// let response = client.post(url).send().await?;
/// let mut stream = parse_sse_stream(response);
///
/// while let Some(result) = stream.next().await {
///     match result {
///         Ok(chunk) => process_chunk(chunk),
///         Err(e) => eprintln!("Stream error: {}", e),
///     }
/// }
/// ```
pub fn parse_sse_stream(
    body: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<OpenAIChunk>> + Send>> {
    parse_events(body)
}

/// Parses an SSE response body as a stream of [`AnthropicEvent`]s.
///
/// The Anthropic sibling of [`parse_sse_stream`]. Anthropic labels each frame with an
/// `event:` line as well as its `data:` payload, but the payload repeats the type in its own
/// `type` field, so only the payload is parsed and the two can never disagree about what an
/// event is.
///
/// Anthropic does not terminate with `data: [DONE]` — `message_stop` ends the response and
/// the connection closes — but the sentinel is skipped here as it is for OpenAI, because
/// some compatible third-party endpoints send both.
pub fn parse_anthropic_sse_stream(
    body: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<AnthropicEvent>> + Send>> {
    parse_events(body)
}

/// Decodes an SSE body into `T`, one item per complete event.
///
/// The shared half of both public parsers: the SSE framing, the `[DONE]` sentinel and the
/// error classification are protocol-independent, and only the payload type differs. Written
/// once so a fix to the transport handling cannot land in one protocol and miss the other.
fn parse_events<T>(body: reqwest::Response) -> Pin<Box<dyn Stream<Item = Result<T>> + Send>>
where
    T: DeserializeOwned + Send + 'static,
{
    let stream = body
        .bytes_stream()
        .eventsource()
        .filter_map(|event_result| async move {
            match event_result {
                Ok(event) if event.data == "[DONE]" => None,
                Ok(event) => Some(serde_json::from_str(&event.data).map_err(|error| {
                    Error::stream(format!("Failed to parse SSE event data: {error}"))
                })),
                Err(EventStreamError::Transport(error)) => Some(Err(Error::Http(error))),
                Err(error) => Some(Err(Error::stream(format!(
                    "Failed to parse SSE event: {error}"
                )))),
            }
        });

    // Pin the stream to the heap and box it for dynamic dispatch.
    // This allows the function to return a uniform type regardless of the
    // concrete stream implementation.
    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    include!("tests/sse.rs");
}
