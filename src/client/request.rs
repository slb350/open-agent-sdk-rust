//! Shared request assembly; protocol translation happens in `stream_request`.
use crate::types::{
    AgentOptions, ContentBlock, Message, MessageRole, OpenAIContent, OpenAIContentPart,
    OpenAIFunction, OpenAIMessage, OpenAIRequest, OpenAIToolCall,
};

pub(super) fn build_request(options: &AgentOptions, messages: Vec<OpenAIMessage>) -> OpenAIRequest {
    OpenAIRequest {
        model: options.model().to_string(),
        messages,
        stream: true,
        max_tokens: options.max_tokens(),
        temperature: options.temperature(),
        tools: (!options.tools().is_empty()).then(|| {
            options
                .tools()
                .iter()
                .map(|tool| tool.to_openai_format())
                .collect()
        }),
    }
}

pub(super) fn history_messages(options: &AgentOptions, history: &[Message]) -> Vec<OpenAIMessage> {
    let mut messages = Vec::new();
    if !options.system_prompt().is_empty() {
        messages.push(message(
            "system",
            OpenAIContent::Text(options.system_prompt().to_string()),
        ));
    }
    for entry in history {
        let mut text = Vec::new();
        let mut tool_calls = Vec::new();
        let mut tool_results = Vec::new();
        let mut has_images = false;
        for block in &entry.content {
            match block {
                ContentBlock::Text(block) => text.push(block.text.as_str()),
                ContentBlock::Image(_) => has_images = true,
                ContentBlock::ToolUse(call) => tool_calls.push(OpenAIToolCall {
                    id: call.id().to_string(),
                    call_type: "function".to_string(),
                    function: OpenAIFunction {
                        name: call.name().to_string(),
                        arguments: call.input().to_string(),
                    },
                }),
                ContentBlock::ToolResult(result) => tool_results.push(OpenAIMessage {
                    tool_call_id: Some(result.tool_use_id().to_string()),
                    ..message("tool", OpenAIContent::Text(result.content().to_string()))
                }),
            }
        }
        if !tool_results.is_empty() {
            messages.extend(tool_results);
        } else if !tool_calls.is_empty() {
            // Tool-only assistant messages retain an explicit empty content field.
            messages.push(OpenAIMessage {
                tool_calls: Some(tool_calls),
                ..message("assistant", OpenAIContent::Text(text.join("\n")))
            });
        } else {
            let role = match entry.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::Tool => "tool",
            };
            let content = if has_images {
                OpenAIContent::Parts(
                    entry
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text(text) => Some(OpenAIContentPart::text(&text.text)),
                            ContentBlock::Image(image) => {
                                if log::log_enabled!(log::Level::Debug) {
                                    let url = image.url();
                                    let mut end = url.len().min(100);
                                    while !url.is_char_boundary(end) {
                                        end -= 1;
                                    }
                                    log::debug!(
                                        "  - Image: {}{} ({} bytes) (detail: {})",
                                        &url[..end],
                                        if end < url.len() { "..." } else { "" },
                                        url.len(),
                                        image.detail()
                                    );
                                }
                                Some(OpenAIContentPart::from_image(image))
                            }
                            ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) => None,
                        })
                        .collect(),
                )
            } else {
                OpenAIContent::Text(text.join("\n"))
            };
            messages.push(message(role, content));
        }
    }
    messages
}

fn message(role: &str, content: OpenAIContent) -> OpenAIMessage {
    OpenAIMessage {
        role: role.to_string(),
        content: Some(content),
        tool_calls: None,
        tool_call_id: None,
    }
}
