//! Wire types for the Anthropic messages API, and the translation into them.
//!
//! [`OpenAIRequest`] stays the SDK's single internal request representation. Both call sites
//! that build one — `query()` and `Client::start_request()` — are protocol-agnostic, and the
//! translation happens once, at the transport boundary. Branching earlier would mean two
//! parallel request builders drifting apart, which is the defect this arrangement exists to
//! prevent.
//!
//! Three shape differences carry real logic rather than field renaming:
//!
//! - **The system prompt is not a message.** OpenAI puts it in the `messages` array with
//!   `role: "system"`; Anthropic takes it as a top-level `system` field. Multiple system
//!   messages are joined rather than dropped, because the auto-execution loop can append one.
//! - **Tool results are user turns.** OpenAI sends `role: "tool"` with a `tool_call_id`;
//!   Anthropic sends a `user` message whose content opens with `tool_result` blocks. Runs of
//!   consecutive tool results merge into one user turn, because the API rejects a turn that
//!   answers only some of the outstanding calls.
//! - **Tool schemas are flat.** OpenAI nests under `function` and calls the schema
//!   `parameters`; Anthropic puts `name`/`description`/`input_schema` at the top level.

use serde::Serialize;
use serde_json::{Value, json};

use super::{OpenAIContent, OpenAIContentPart, OpenAIMessage, OpenAIRequest};

/// Request payload for `POST {base_url}/messages`.
///
/// Optional fields are omitted when `None` so the server applies its own defaults, matching
/// [`OpenAIRequest`]'s treatment of `max_tokens` and `temperature`.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicRequest {
    /// Model identifier (e.g. `"claude-opus-5"`, `"k3"`, `"MiniMax-M3"`).
    pub model: String,

    /// Conversation turns, alternating user and assistant. Never carries the system prompt.
    pub messages: Vec<AnthropicMessage>,

    /// The system prompt, as a top-level field rather than a message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Whether to stream the response. The SDK always sets this.
    pub stream: bool,

    /// Maximum tokens to generate.
    ///
    /// Optional in the current Anthropic API, and omitted when unset so a long-context model
    /// is not truncated by a client-invented ceiling. Some Anthropic-compatible third-party
    /// endpoints still require it; that is the caller's to set, not this layer's to invent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature. Anthropic's accepted range is 0.0 to 1.0, narrower than
    /// OpenAI's 0.0 to 2.0, and several compatible endpoints reject any value at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Tool definitions in Anthropic's flat shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
}

/// One conversation turn.
///
/// `content` is deliberately a [`Value`]: Anthropic accepts either a bare string or an array
/// of typed blocks, and the SDK emits whichever the turn actually needs.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessage {
    /// `"user"` or `"assistant"`. Anthropic has no `"system"` or `"tool"` role.
    pub role: String,

    /// A string for text-only turns, or an array of content blocks.
    pub content: Value,
}

impl AnthropicRequest {
    /// Translates the SDK's canonical request into Anthropic's shape.
    ///
    /// Field-for-field except where the module documentation says otherwise.
    pub fn from_openai(request: &OpenAIRequest) -> Self {
        let mut system = Vec::new();
        let mut messages: Vec<AnthropicMessage> = Vec::new();

        for message in &request.messages {
            match message.role.as_str() {
                "system" => {
                    if let Some(text) = plain_text(message) {
                        system.push(text);
                    }
                }
                "tool" => push_tool_result(&mut messages, message),
                _ => messages.push(convert_turn(message)),
            }
        }

        Self {
            model: request.model.clone(),
            messages,
            system: (!system.is_empty()).then(|| system.join("\n\n")),
            stream: request.stream,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            tools: request
                .tools
                .as_ref()
                .map(|tools| tools.iter().map(to_anthropic_tool).collect()),
        }
    }
}

/// The message's content as a plain string, if it has any.
///
/// A system message carrying image parts has no meaning in either API, so only the text
/// parts survive.
fn plain_text(message: &OpenAIMessage) -> Option<String> {
    match message.content.as_ref()? {
        OpenAIContent::Text(text) => Some(text.clone()),
        OpenAIContent::Parts(parts) => {
            let text: String = parts
                .iter()
                .filter_map(|part| match part {
                    OpenAIContentPart::Text { text } => Some(text.as_str()),
                    OpenAIContentPart::ImageUrl { .. } => None,
                })
                .collect();
            (!text.is_empty()).then_some(text)
        }
    }
}

/// Converts a user or assistant turn.
///
/// An assistant turn carrying tool calls becomes a block array: its text first (when it said
/// anything alongside the calls), then one `tool_use` block per call, in the order the model
/// requested them.
fn convert_turn(message: &OpenAIMessage) -> AnthropicMessage {
    // A text-only turn serializes as a bare string, which is what the API's own examples show
    // and what keeps a simple request readable on the wire. The decision reads the source
    // message rather than re-inspecting the JSON a sibling function just built, so it does not
    // depend on the shape of a literal no type checks.
    if let Some(text) = bare_text(message) {
        return AnthropicMessage {
            role: message.role.clone(),
            content: Value::String(text),
        };
    }

    let mut blocks = content_blocks(message);

    if let Some(tool_calls) = &message.tool_calls {
        for call in tool_calls {
            blocks.push(json!({
                "type": "tool_use",
                "id": call.id,
                "name": call.function.name,
                // Arguments cross the OpenAI wire as a JSON *string*; Anthropic wants the
                // parsed object. An unparseable string means the model emitted malformed
                // arguments, and an empty object is the same fallback the OpenAI-side
                // accumulator applies rather than failing the whole turn.
                "input": serde_json::from_str::<Value>(&call.function.arguments)
                    .unwrap_or_else(|_| json!({})),
            }));
        }
    }

    AnthropicMessage {
        role: message.role.clone(),
        content: Value::Array(blocks),
    }
}

/// The turn's content as a bare string, when it is text and nothing else.
///
/// A turn carrying tool calls, images, or several parts needs the block array, and a turn
/// with no content at all becomes an empty one.
fn bare_text(message: &OpenAIMessage) -> Option<String> {
    if message.tool_calls.is_some() {
        return None;
    }

    match message.content.as_ref()? {
        OpenAIContent::Text(text) => Some(text.clone()),
        OpenAIContent::Parts(parts) => match parts.as_slice() {
            [OpenAIContentPart::Text { text }] => Some(text.clone()),
            _ => None,
        },
    }
}

/// The typed content blocks of a message, excluding tool calls.
fn content_blocks(message: &OpenAIMessage) -> Vec<Value> {
    match message.content.as_ref() {
        None => Vec::new(),
        Some(OpenAIContent::Text(text)) => vec![json!({ "type": "text", "text": text })],
        Some(OpenAIContent::Parts(parts)) => parts.iter().map(convert_part).collect(),
    }
}

/// Converts one OpenAI content part.
///
/// A `data:` URI carries the bytes inline and becomes a `base64` source; anything else is
/// passed through as a `url` source. Splitting on the scheme rather than fetching keeps this
/// a pure transform.
fn convert_part(part: &OpenAIContentPart) -> Value {
    match part {
        OpenAIContentPart::Text { text } => json!({ "type": "text", "text": text }),
        OpenAIContentPart::ImageUrl { image_url } => match parse_data_uri(&image_url.url) {
            Some((media_type, data)) => json!({
                "type": "image",
                "source": { "type": "base64", "media_type": media_type, "data": data },
            }),
            None => json!({
                "type": "image",
                "source": { "type": "url", "url": image_url.url },
            }),
        },
    }
}

/// Splits `data:<media-type>[;<parameter>...];base64,<data>` into its media type and payload.
///
/// Returns `None` for any other URI, including a `data:` URI that is not base64-encoded —
/// Anthropic's `base64` source would misread the payload, and a `url` source at least fails
/// visibly.
///
/// The media type ends at the first `;`; anything between there and `;base64,` is a
/// parameter such as `charset`. [`ImageBlock::from_url`](crate::ImageBlock::from_url) accepts
/// such a URI and reads the media type the same way, so treating the parameters as part of it
/// here would put a media type Anthropic rejects on the wire for an image the SDK had already
/// accepted.
fn parse_data_uri(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(";base64,")?;
    let media_type = match meta.split_once(';') {
        Some((media_type, _parameters)) => media_type,
        None => meta,
    };
    (!media_type.is_empty()).then(|| (media_type.to_string(), data.to_string()))
}

/// Appends a tool result, merging it into the preceding user turn when there is one.
///
/// Anthropic requires every outstanding `tool_use` to be answered within a single user turn.
/// Parallel tool calls arrive here as consecutive OpenAI `tool` messages, so a run of them
/// has to collapse into one message rather than becoming several.
fn push_tool_result(messages: &mut Vec<AnthropicMessage>, message: &OpenAIMessage) {
    let block = json!({
        "type": "tool_result",
        "tool_use_id": message.tool_call_id.clone().unwrap_or_default(),
        "content": plain_text(message).unwrap_or_default(),
    });

    // Tool results carry whole command outputs and file contents, so the block is moved into
    // the merge and handed back only when there was no open turn to absorb it.
    let Some(block) = merge_into_open_tool_turn(messages, block) else {
        return;
    };

    messages.push(AnthropicMessage {
        role: "user".to_string(),
        content: json!([block]),
    });
}

/// Appends `block` to the last message when that message is a tool-result turn.
///
/// Returns `None` when it merged, and gives `block` back untouched when it did not, so a
/// payload that can be arbitrarily large is never copied. Merging only into a turn this module
/// itself built is the point:
/// a user turn the caller supplied is a real conversational turn, and appending a tool result
/// to it would reorder the conversation.
///
/// The role is not checked, and checking it would be unfalsifiable code. A content array
/// whose first block is a `tool_result` is only ever produced here, and this function always
/// gives it `role: "user"` — so "opens with a tool result" already implies "is a user turn
/// this module built", and a role test could never take its false branch.
///
/// A standalone function rather than a let-chain inside the caller, because let-chains are
/// stable only from Rust 1.88 and this crate's MSRV is 1.85.
fn merge_into_open_tool_turn(messages: &mut [AnthropicMessage], block: Value) -> Option<Value> {
    let Some(last) = messages.last_mut() else {
        return Some(block);
    };
    // A bare-string turn is text, and an array that opens with anything else is a real user
    // turn — an image, say. Neither may absorb a tool result.
    let Value::Array(blocks) = &mut last.content else {
        return Some(block);
    };
    let opens_with_tool_result = blocks
        .first()
        .and_then(|block| block.get("type"))
        .and_then(Value::as_str)
        == Some("tool_result");
    if !opens_with_tool_result {
        return Some(block);
    }

    blocks.push(block);
    None
}

/// Flattens an OpenAI tool definition into Anthropic's shape.
///
/// Anything that is not the expected `{"type":"function","function":{...}}` envelope is
/// passed through untouched: a caller who hand-wrote a definition in Anthropic's shape
/// already gets what it asked for, and mangling it would be worse than leaving it alone.
fn to_anthropic_tool(tool: &Value) -> Value {
    let Some(function) = tool.get("function") else {
        return tool.clone();
    };

    let mut out = serde_json::Map::new();
    if let Some(name) = function.get("name") {
        out.insert("name".to_string(), name.clone());
    }
    if let Some(description) = function.get("description") {
        out.insert("description".to_string(), description.clone());
    }
    out.insert(
        "input_schema".to_string(),
        function
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
    );
    Value::Object(out)
}

#[cfg(test)]
mod tests;
