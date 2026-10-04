//! Errors a server reports inside an OpenAI-protocol stream.
//!
//! The status line is final once the headers are sent, so a failure after that point reaches
//! the client as an SSE event. Three servers this SDK lists document three spellings of it:
//! vLLM sends `{"error": {"message", "type", "param", "code"}}` and llama.cpp
//! `{"error": {"code", "message", "type"}}`, each as a whole payload, and OpenRouter sends a
//! complete chunk whose top-level `error` sits beside a choice ending in
//! `finish_reason: "error"`. All three carry the HTTP status the failure would have had as an
//! integer `code`.
//!
//! None of that is an [`OpenAIChunk`], so left to the generic decoder the first two failed as
//! "missing field `id`" — a retryable stream error that hid the server's message — and the
//! third was swallowed as an ordinary finish, which a `Client` recorded as a completed turn.
//! [`decode_chunk`] turns each into an [`Error::Api`], so the message survives and
//! [`is_retryable_error`](crate::retry::is_retryable_error) can tell a rate limit from a
//! prompt that will never fit.

use serde::Deserialize;

use super::sse::decode_json;
use crate::types::OpenAIChunk;
use crate::{Error, Result};

/// The `finish_reason` OpenRouter puts on the chunk that carries its error object.
const ERROR_FINISH_REASON: &str = "error";

/// Reported when a server sends an `error` object with neither a type nor a message.
const NO_DETAIL: &str = "the server reported an error without details";

/// The part of a payload that matters here; every other field is ignored.
#[derive(Deserialize)]
struct Envelope {
    error: Option<ErrorBody>,
}

#[derive(Deserialize)]
struct ErrorBody {
    message: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    /// An HTTP status for vLLM, llama.cpp and OpenRouter, but a string such as
    /// `"context_length_exceeded"` for OpenAI itself, so it is read as a bare value.
    code: Option<serde_json::Value>,
}

/// Decodes one OpenAI SSE payload, reporting a server-sent error as [`Error::Api`].
///
/// A payload that parses is a chunk unless it ends with `finish_reason: "error"` and carries an
/// `error` object. A chunk with an `error` beside content the server did not call failed is
/// delivered, since dropping it would discard a response nobody said was lost. A payload that
/// does not parse is searched for an `error` object before it is called malformed.
///
/// The common case pays for one parse: the `error` object is only looked for on a chunk that
/// finishes with `"error"`, or on one that did not parse at all.
pub(super) fn decode_chunk(data: &str) -> Result<OpenAIChunk> {
    let parsed = decode_json::<OpenAIChunk>(data);
    let may_carry_error = match &parsed {
        Ok(chunk) => chunk
            .choices
            .iter()
            .any(|choice| choice.finish_reason.as_deref() == Some(ERROR_FINISH_REASON)),
        Err(_) => true,
    };

    match may_carry_error.then(|| reported_error(data)).flatten() {
        Some(error) => Err(error),
        None => parsed,
    }
}

/// The error a payload's top-level `error` object describes, if it has a usable one.
fn reported_error(data: &str) -> Option<Error> {
    let body = serde_json::from_str::<Envelope>(data).ok()?.error?;
    let message = describe(body.kind, body.message);

    Some(match body.code.as_ref().and_then(http_status) {
        Some(status) => Error::api_status(status, message),
        None => Error::api(message),
    })
}

/// An error status in `400..=599`; anything else in `code` is a name or a number that is not
/// a status, and a statusless error is never retried.
fn http_status(code: &serde_json::Value) -> Option<u16> {
    code.as_u64()
        .and_then(|code| u16::try_from(code).ok())
        .filter(|status| (400..=599).contains(status))
}

/// Joins type and message as `type: message`, the form the Anthropic path reports.
fn describe(kind: Option<String>, message: Option<String>) -> String {
    match (
        kind.filter(|kind| !kind.is_empty()),
        message.filter(|message| !message.is_empty()),
    ) {
        (Some(kind), Some(message)) => format!("{kind}: {message}"),
        (None, Some(message)) => message,
        (Some(kind), None) => kind,
        (None, None) => NO_DETAIL.to_string(),
    }
}

#[cfg(test)]
mod tests;
