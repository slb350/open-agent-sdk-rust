impl Client {
    /// Creates a new client with the specified configuration.
    ///
    /// This constructor initializes all state fields and creates a reusable HTTP client
    /// configured with the timeout from `AgentOptions`.
    ///
    /// # Parameters
    ///
    /// - `options`: Configuration including model, API key, tools, hooks, etc.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be built. This can happen due to:
    /// - Invalid TLS configuration
    /// - System resource exhaustion
    /// - Invalid timeout values
    ///
    /// # Examples
    ///
    /// ```rust
    /// use open_agent::{Client, AgentOptions};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new(AgentOptions::builder()
    ///     .model("gpt-4")
    ///     .base_url("http://localhost:1234/v1")
    ///     .build()?)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(options: AgentOptions) -> Result<Self> {
        // Build HTTP client with configured timeout
        // This client is reused across all requests for connection pooling
        let http_client = model_http_client_builder(&options)
            .build()
            .map_err(|e| Error::config(format!("Failed to build HTTP client: {}", e)))?;

        Ok(Self {
            options,
            history: Vec::new(),  // Empty conversation history
            current_stream: None, // No active stream yet
            http_client,
            interrupted: Arc::new(AtomicBool::new(false)), // Not interrupted initially
            auto_exec_buffer: Vec::new(),                  // Empty buffer for auto mode
            auto_exec_index: 0,                            // Start at beginning of buffer
            manual_receive_buffer: Vec::new(),             // Empty buffer for manual mode
            last_finish_reason: None,                      // No stream has completed yet
            last_reasoning: String::new(),                 // No reasoning captured yet
        })
    }

    async fn start_request(&mut self) -> Result<()> {
        // Clear per-stream observations so a caller reading them between `send()` and the end
        // of the receive loop cannot see the previous turn's values.
        self.last_finish_reason = None;
        self.last_reasoning.clear();

        // Build messages array for API request
        // This includes system prompt + full conversation history
        let mut messages = Vec::new();

        // Add system prompt as first message if configured
        // System prompts are added fresh for each request (not from history)
        if !self.options.system_prompt().is_empty() {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(OpenAIContent::Text(
                    self.options.system_prompt().to_string(),
                )),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Convert conversation history to OpenAI message format
        // This includes user prompts, assistant responses, and tool results
        for msg in &self.history {
            // Separate blocks by type to determine message structure
            let mut text_blocks = Vec::new();
            let mut image_blocks = Vec::new();
            let mut tool_use_blocks = Vec::new();
            let mut tool_result_blocks = Vec::new();

            for block in &msg.content {
                match block {
                    ContentBlock::Text(text) => text_blocks.push(text),
                    ContentBlock::Image(image) => image_blocks.push(image),
                    ContentBlock::ToolUse(tool_use) => tool_use_blocks.push(tool_use),
                    ContentBlock::ToolResult(tool_result) => tool_result_blocks.push(tool_result),
                }
            }

            // Handle different message types based on content blocks
            // Case 1: Message contains tool results (should be separate tool messages)
            if !tool_result_blocks.is_empty() {
                for tool_result in tool_result_blocks {
                    // Serialize the tool result content as JSON string
                    let content =
                        serde_json::to_string(tool_result.content()).unwrap_or_else(|e| {
                            format!("{{\"error\": \"Failed to serialize: {}\"}}", e)
                        });

                    messages.push(OpenAIMessage {
                        role: "tool".to_string(),
                        content: Some(OpenAIContent::Text(content)),
                        tool_calls: None,
                        tool_call_id: Some(tool_result.tool_use_id().to_string()),
                    });
                }
            }
            // Case 2: Message contains tool use blocks (assistant with tool calls)
            else if !tool_use_blocks.is_empty() {
                // Build tool_calls array
                let tool_calls: Vec<OpenAIToolCall> = tool_use_blocks
                    .iter()
                    .map(|tool_use| {
                        // Serialize the input as a JSON string (OpenAI API requirement)
                        let arguments = serde_json::to_string(tool_use.input())
                            .unwrap_or_else(|_| "{}".to_string());

                        OpenAIToolCall {
                            id: tool_use.id().to_string(),
                            call_type: "function".to_string(),
                            function: OpenAIFunction {
                                name: tool_use.name().to_string(),
                                arguments,
                            },
                        }
                    })
                    .collect();

                // Extract any text content (some models include reasoning before tool calls)
                // Note: OpenAI API requires content field even if empty when tool_calls present
                let content = if !text_blocks.is_empty() {
                    let text = text_blocks
                        .iter()
                        .map(|t| t.text.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    Some(OpenAIContent::Text(text))
                } else {
                    // Empty string satisfies OpenAI API schema (content is required)
                    Some(OpenAIContent::Text(String::new()))
                };

                messages.push(OpenAIMessage {
                    role: "assistant".to_string(),
                    content,
                    tool_calls: Some(tool_calls),
                    tool_call_id: None,
                });
            }
            // Case 3: Message contains images (use OpenAIContent::Parts)
            else if !image_blocks.is_empty() {
                // Log debug info about images being serialized
                log::debug!(
                    "Serializing message with {} image(s) for {:?} role",
                    image_blocks.len(),
                    msg.role
                );

                // Build content parts array preserving original order
                let mut content_parts = Vec::new();

                // Re-iterate through content blocks to maintain order
                for block in &msg.content {
                    match block {
                        ContentBlock::Text(text) => {
                            content_parts.push(OpenAIContentPart::text(&text.text));
                        }
                        ContentBlock::Image(image) => {
                            // Log image details (truncate URL for privacy)
                            let url_display = if image.url().len() > 100 {
                                format!("{}... ({} chars)", &image.url()[..100], image.url().len())
                            } else {
                                image.url().to_string()
                            };
                            let detail_str = match image.detail() {
                                crate::types::ImageDetail::Low => "low",
                                crate::types::ImageDetail::High => "high",
                                crate::types::ImageDetail::Auto => "auto",
                            };
                            log::debug!("  - Image: {} (detail: {})", url_display, detail_str);

                            content_parts.push(OpenAIContentPart::from_image(image));
                        }
                        ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) => {}
                    }
                }

                // Defensive check: content_parts should never be empty at this point
                // If it is, it indicates a logic error (e.g., all blocks were filtered out)
                if content_parts.is_empty() {
                    return Err(Error::other(
                        "Internal error: Message with images produced empty content array",
                    ));
                }

                let role_str = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                };

                messages.push(OpenAIMessage {
                    role: role_str.to_string(),
                    content: Some(OpenAIContent::Parts(content_parts)),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            // Case 4: Message contains only text (normal message, backward compatible)
            else {
                let content = text_blocks
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");

                let role_str = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                };

                messages.push(OpenAIMessage {
                    role: role_str.to_string(),
                    content: Some(OpenAIContent::Text(content)),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }

        // Convert tools to OpenAI format if any are registered
        // Each tool is described with name, description, and JSON Schema parameters
        let tools = if !self.options.tools().is_empty() {
            Some(
                self.options
                    .tools()
                    .iter()
                    .map(|t| t.to_openai_format())
                    .collect(),
            )
        } else {
            None
        };

        // Build the OpenAI-compatible request payload
        let request = OpenAIRequest {
            model: self.options.model().to_string(),
            messages,
            stream: true, // Always stream for progressive rendering
            max_tokens: self.options.max_tokens(),
            temperature: self.options.temperature(),
            tools,
        };

        // Store the stream for consumption via receive()
        // The stream is NOT consumed here - that happens in receive()
        self.current_stream = Some(stream_request(&self.http_client, &self.options, &request).await?);

        Ok(())
    }

}
