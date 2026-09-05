//! Streaming requests and conversation state.
//!
//! [`query`] streams a single prompt. [`Client`] also manages history and supports
//! manual or automatic tool execution. Manual mode yields content immediately;
//! auto mode executes tool rounds and then yields the final response.
//!
//! ```rust,no_run
//! use open_agent::{AgentOptions, Client, ContentBlock};
//! # async fn example() -> open_agent::Result<()> {
//! let options = AgentOptions::builder()
//!     .model("local-model").base_url("http://localhost:1234/v1").build()?;
//! let mut client = Client::new(options)?;
//! client.send("Hello").await?;
//! while let Some(block) = client.receive().await? {
//!     if let ContentBlock::Text(text) = block { print!("{}", text.text); }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! For manual tools, execute each `ToolUse`, supply it with [`Client::add_tool_result`],
//! and call `send("")` to continue. For auto mode, register tools and enable
//! [`AgentOptions::auto_execute_tools`]. Context truncation is always caller controlled.
//!
//! Cancellation is checked between stream events. Share [`Client::interrupt_handle`]
//! rather than locking a client across an awaited receive. New sends discard pending
//! output; completed history remains until [`Client::clear_history`] is called.

mod request;

use crate::types::{
    AgentOptions, AnthropicRequest, ApiProtocol, ContentBlock, FinishReason, Message, MessageRole,
    OpenAIContent, OpenAIMessage, OpenAIRequest, StreamEvent,
};
use crate::utils::{
    AnthropicAccumulator, StreamAccumulator, coalesce_text_blocks, drive,
    parse_anthropic_sse_stream, parse_sse_stream,
};
use crate::{Error, Result};
use futures::stream::{Stream, StreamExt};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

fn serialize_history_message(message: &Message) -> Result<serde_json::Value> {
    serde_json::to_value(message).map_err(|error| {
        Error::other(format!(
            "Failed to serialize conversation history for hook: {error}"
        ))
    })
}

fn serialize_history_snapshot(history: &[Message]) -> Result<Vec<serde_json::Value>> {
    history.iter().map(serialize_history_message).collect()
}

/// The API version header every Anthropic messages request must carry.
///
/// A dated constant rather than a configurable field: it names the request/response schema
/// this SDK was written against, so it changes when the code does.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Builds the shared HTTP-client policy for model requests.
///
/// Redirects are rejected even when they stay on the same origin. The configured base URL
/// names the exact model endpoint, and caller-supplied credentials must never be replayed to a
/// destination selected by an HTTP response.
fn model_http_client_builder(options: &AgentOptions) -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(options.timeout()))
        .redirect(reqwest::redirect::Policy::none())
}

/// Sends the request over whichever protocol `options` selects and returns its event stream.
///
/// The one place the two protocols differ. Both call sites build the same protocol-neutral
/// [`OpenAIRequest`]; the translation, the auth header and the streaming vocabulary are all
/// resolved here, so neither caller has to know which endpoint it is talking to.
async fn stream_request(
    http_client: &reqwest::Client,
    options: &AgentOptions,
    request: &OpenAIRequest,
) -> Result<EventStream> {
    let protocol = options.protocol();
    let url = format!("{}{}", options.base_url(), protocol.path());
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    if !options.api_key().is_empty() {
        let (name, value) = match protocol {
            ApiProtocol::OpenAiChat => (
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", options.api_key())).map_err(|_| {
                    Error::config("api_key cannot be encoded as an HTTP header value")
                })?,
            ),
            ApiProtocol::Anthropic => (
                HeaderName::from_static("x-api-key"),
                HeaderValue::from_str(options.api_key()).map_err(|_| {
                    Error::config("api_key cannot be encoded as an HTTP header value")
                })?,
            ),
        };
        headers.insert(name, value);
    }

    if protocol == ApiProtocol::Anthropic {
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
    }

    // `HeaderMap::insert` replaces case-insensitively. Applying caller values last is what
    // makes custom authentication possible without dropping unrelated protocol defaults.
    crate::types::http_headers::insert_all(&mut headers, options.headers())?;

    let pending = http_client.post(&url).headers(headers);

    let pending = match protocol {
        ApiProtocol::OpenAiChat => pending.json(request),
        ApiProtocol::Anthropic => pending.json(&AnthropicRequest::from_openai(request)),
    };

    let response = pending.send().await.map_err(Error::Http)?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_else(|error| {
            eprintln!("WARNING: Failed to read error response body: {error}");
            "Unknown error (failed to read response body)".to_string()
        });
        return Err(Error::api_status(status, body));
    }

    let capture = options.include_reasoning();
    Ok(match protocol {
        ApiProtocol::OpenAiChat => drive(
            parse_sse_stream(response),
            StreamAccumulator::new().capture_reasoning(capture),
        ),
        ApiProtocol::Anthropic => drive(
            parse_anthropic_sse_stream(response),
            AnthropicAccumulator::new().capture_reasoning(capture),
        ),
    })
}

include!("client/query.rs");
include!("client/state.rs");
include!("client/setup.rs");
include!("client/send.rs");
include!("client/streaming.rs");
include!("client/send_message.rs");
include!("client/receive.rs");
include!("client/history.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TextBlock;

    include!("client/tests.rs");
}
