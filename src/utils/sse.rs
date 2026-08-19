//! Transport-safe decoding of an SSE response body into [`OpenAIChunk`]s.

use crate::types::OpenAIChunk;
use crate::{Error, Result};
use eventsource_stream::{EventStreamError, Eventsource};
use futures::stream::{Stream, StreamExt};
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
