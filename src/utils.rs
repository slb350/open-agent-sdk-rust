//! Utility functions for streaming and tool call aggregation

use crate::types::{ContentBlock, OpenAIChunk, TextBlock, ToolUseBlock};
use crate::{Error, Result};
use futures::stream::{Stream, StreamExt};
use std::collections::HashMap;
use std::pin::Pin;

/// Aggregates streaming chunks into content blocks
pub struct ToolCallAggregator {
    text_buffer: String,
    tool_calls: HashMap<u32, PartialToolCall>,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl ToolCallAggregator {
    pub fn new() -> Self {
        Self {
            text_buffer: String::new(),
            tool_calls: HashMap::new(),
        }
    }

    /// Process a chunk and return any completed content blocks
    pub fn process_chunk(&mut self, chunk: OpenAIChunk) -> Result<Vec<ContentBlock>> {
        let mut blocks = Vec::new();

        for choice in chunk.choices {
            // Handle text content
            if let Some(content) = choice.delta.content {
                self.text_buffer.push_str(&content);
            }

            // Handle tool calls
            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    let entry = self.tool_calls.entry(tool_call.index).or_default();

                    if let Some(id) = tool_call.id {
                        entry.id = Some(id);
                    }

                    if let Some(function) = tool_call.function {
                        if let Some(name) = function.name {
                            entry.name = Some(name);
                        }
                        if let Some(args) = function.arguments {
                            entry.arguments.push_str(&args);
                        }
                    }
                }
            }

            // Check for finish reason
            if choice.finish_reason.is_some() {
                // Flush text buffer
                if !self.text_buffer.is_empty() {
                    blocks.push(ContentBlock::Text(TextBlock::new(self.text_buffer.clone())));
                    self.text_buffer.clear();
                }

                // Flush completed tool calls
                for (_, partial) in self.tool_calls.drain() {
                    if let (Some(id), Some(name)) = (partial.id, partial.name) {
                        let input: serde_json::Value = if partial.arguments.is_empty() {
                            serde_json::json!({})
                        } else {
                            serde_json::from_str(&partial.arguments).map_err(|e| {
                                Error::stream(format!("Failed to parse tool arguments: {}", e))
                            })?
                        };

                        blocks.push(ContentBlock::ToolUse(ToolUseBlock::new(id, name, input)));
                    }
                }
            }
        }

        Ok(blocks)
    }
}

/// Parse SSE (Server-Sent Events) stream
pub fn parse_sse_stream(
    body: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<OpenAIChunk>> + Send>> {
    let stream = body.bytes_stream().map(move |result| {
        let bytes = result.map_err(Error::Http)?;
        let text = String::from_utf8_lossy(&bytes);

        // Parse SSE format: "data: {...}\n\n"
        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    continue;
                }

                let chunk: OpenAIChunk = serde_json::from_str(data)
                    .map_err(|e| Error::stream(format!("Failed to parse chunk: {}", e)))?;

                return Ok(chunk);
            }
        }

        Err(Error::stream("No data in SSE chunk"))
    });

    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OpenAIChoice, OpenAIDelta, OpenAIFunctionDelta, OpenAIToolCallDelta};

    #[test]
    fn test_tool_call_aggregator_text() {
        let mut aggregator = ToolCallAggregator::new();

        let chunk = OpenAIChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: Some("Hello ".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        };

        let blocks = aggregator.process_chunk(chunk).unwrap();
        assert_eq!(blocks.len(), 0); // Not finished yet

        let chunk2 = OpenAIChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: Some("world".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };

        let blocks = aggregator.process_chunk(chunk2).unwrap();
        assert_eq!(blocks.len(), 1);

        match &blocks[0] {
            ContentBlock::Text(text) => assert_eq!(text.text, "Hello world"),
            _ => panic!("Expected text block"),
        }
    }

    #[test]
    fn test_tool_call_aggregator_tool() {
        let mut aggregator = ToolCallAggregator::new();

        let chunk = OpenAIChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(vec![OpenAIToolCallDelta {
                        index: 0,
                        id: Some("call_123".to_string()),
                        call_type: Some("function".to_string()),
                        function: Some(OpenAIFunctionDelta {
                            name: Some("get_weather".to_string()),
                            arguments: Some(r#"{"location":"#.to_string()),
                        }),
                    }]),
                },
                finish_reason: None,
            }],
        };

        let blocks = aggregator.process_chunk(chunk).unwrap();
        assert_eq!(blocks.len(), 0);

        let chunk2 = OpenAIChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                delta: OpenAIDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(vec![OpenAIToolCallDelta {
                        index: 0,
                        id: None,
                        call_type: None,
                        function: Some(OpenAIFunctionDelta {
                            name: None,
                            arguments: Some(r#""Paris"}"#.to_string()),
                        }),
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
        };

        let blocks = aggregator.process_chunk(chunk2).unwrap();
        assert_eq!(blocks.len(), 1);

        match &blocks[0] {
            ContentBlock::ToolUse(tool) => {
                assert_eq!(tool.id, "call_123");
                assert_eq!(tool.name, "get_weather");
                assert_eq!(tool.input["location"], "Paris");
            }
            _ => panic!("Expected tool use block"),
        }
    }
}
