impl Client {
    async fn receive_one(&mut self) -> Result<Option<ContentBlock>> {
        loop {
            if self.interrupted.load(Ordering::SeqCst) {
                return Ok(None);
            }
            let Some(stream) = &mut self.current_stream else {
                return Ok(None);
            };

            let event = stream.next().await;
            if self.interrupted.load(Ordering::SeqCst) {
                return Ok(None);
            }
            match event {
                Some(Ok(StreamEvent::Block(block))) => return Ok(Some(block)),
                Some(Ok(StreamEvent::Reasoning(reasoning))) => {
                    if self.last_reasoning.is_empty() {
                        self.last_reasoning = reasoning;
                    } else {
                        self.last_reasoning.push_str(&reasoning);
                    }
                }
                Some(Ok(StreamEvent::Finish(reason))) => {
                    self.last_finish_reason = Some(reason);
                }
                Some(Err(e)) => {
                    self.discard_pending_output();
                    return Err(e);
                }
                None => {
                    self.current_stream = None;
                    return Ok(None);
                }
            }
        }
    }

    // Auto mode needs the whole generation before deciding whether to execute tools.
    async fn collect_all_blocks(&mut self) -> Result<Vec<ContentBlock>> {
        let mut blocks = Vec::new();

        while let Some(block) = self.receive_one().await? {
            blocks.push(block);
        }
        if self.interrupted.load(Ordering::SeqCst) {
            self.discard_pending_output();
            Ok(Vec::new())
        } else {
            Ok(blocks)
        }
    }

    async fn execute_tool_internal(
        &self,
        tool_name: &str,
        input: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let tool = self
            .get_tool(tool_name)
            .ok_or_else(|| Error::tool(format!("Tool '{}' not found", tool_name)))?;

        tool.execute(input).await
    }

    fn record_cancelled_tool(&mut self, tool_use_id: &str) {
        self.history
            .push(Message::user_with_blocks(vec![ContentBlock::ToolResult(
                crate::types::ToolResultBlock::new(
                    tool_use_id,
                    serde_json::json!({"error": "Tool execution cancelled"}),
                ),
            )]));
    }

    async fn auto_execute_loop(&mut self) -> Result<Vec<ContentBlock>> {
        use crate::types::ToolResultBlock;

        let mut iteration = 0;
        let max_iterations = self.options.max_tool_iterations();

        loop {
            let blocks = self.collect_all_blocks().await?;

            if blocks.is_empty() {
                return Ok(Vec::new());
            }

            let has_tools = blocks
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse(_)));
            if !has_tools || iteration == max_iterations {
                if has_tools {
                    self.last_finish_reason = Some(FinishReason::MaxToolIterations);
                }
                let text_blocks: Vec<_> = blocks
                    .into_iter()
                    .filter(|block| matches!(block, ContentBlock::Text(_)))
                    .collect();
                if !text_blocks.is_empty() {
                    self.push_assistant(&text_blocks);
                }
                return Ok(text_blocks);
            }
            iteration += 1;
            self.push_assistant(&blocks);
            for block in blocks {
                if let ContentBlock::ToolUse(tool_use) = block {
                    if self.interrupted.load(Ordering::SeqCst) {
                        self.record_cancelled_tool(tool_use.id());
                        continue;
                    }
                    let has_pre_hooks = !self.options.hooks().pre_tool_use.is_empty();
                    let has_post_hooks = !self.options.hooks().post_tool_use.is_empty();
                    let mut history_snapshot = if has_pre_hooks || has_post_hooks {
                        Some(serialize_history_snapshot(&self.history)?)
                    } else {
                        None
                    };

                    use crate::hooks::PreToolUseEvent;
                    let mut tool_input = tool_use.input().clone();
                    let mut should_execute = true;
                    let mut block_reason = None;

                    if has_pre_hooks {
                        let pre_history = if has_post_hooks {
                            history_snapshot
                                .as_ref()
                                .expect("hook history snapshot should exist")
                                .clone()
                        } else {
                            history_snapshot
                                .take()
                                .expect("hook history snapshot should exist")
                        };
                        let pre_event = PreToolUseEvent::new(
                            tool_use.name().to_string(),
                            tool_use.input().clone(),
                            tool_use.id().to_string(),
                            pre_history,
                        );

                        if let Some(decision) =
                            self.options.hooks().execute_pre_tool_use(pre_event).await
                        {
                            if !decision.continue_execution() {
                                should_execute = false;
                                block_reason = decision.reason().map(ToString::to_string);
                            } else if let Some(modified) = decision.modified_input() {
                                tool_input = modified.clone();
                            }
                        }
                    }

                    // Hooks can request cancellation before any tool side effects.
                    if self.interrupted.load(Ordering::SeqCst) {
                        self.record_cancelled_tool(tool_use.id());
                        continue;
                    }
                    let post_input = has_post_hooks.then(|| tool_input.clone());
                    let result = if should_execute {
                        match self
                            .execute_tool_internal(tool_use.name(), tool_input)
                            .await
                        {
                            Ok(res) => res,
                            Err(e) => {
                                serde_json::json!({
                                    "error": e.to_string(),
                                    "tool": tool_use.name(),
                                    "id": tool_use.id()
                                })
                            }
                        }
                    } else {
                        serde_json::json!({
                            "error": "Tool execution blocked by hook",
                            "reason": block_reason.unwrap_or_else(|| "No reason provided".to_string()),
                            "tool": tool_use.name(),
                            "id": tool_use.id()
                        })
                    };

                    let mut final_result = result;
                    if has_post_hooks && !self.interrupted.load(Ordering::SeqCst) {
                        use crate::hooks::PostToolUseEvent;
                        let pending_result =
                            Message::user_with_blocks(vec![ContentBlock::ToolResult(
                                ToolResultBlock::new(tool_use.id(), final_result.clone()),
                            )]);
                        let mut post_history_snapshot = history_snapshot
                            .take()
                            .expect("hook history snapshot should exist");
                        post_history_snapshot.push(serialize_history_message(&pending_result)?);
                        let post_event = PostToolUseEvent::new(
                            tool_use.name().to_string(),
                            post_input.expect("post-hook input is retained"),
                            tool_use.id().to_string(),
                            final_result.clone(),
                            post_history_snapshot,
                        );

                        if let Some(decision) =
                            self.options.hooks().execute_post_tool_use(post_event).await
                        {
                            if let Some(modified) = decision.modified_input() {
                                final_result = modified.clone();
                            }
                        }
                    }

                    let tool_result = ToolResultBlock::new(tool_use.id(), final_result);
                    let tool_result_msg =
                        Message::user_with_blocks(vec![ContentBlock::ToolResult(tool_result)]);
                    self.history.push(tool_result_msg);
                }
            }

            // Every call retains a result, including skipped calls, so a later explicit
            // request can replay valid history. Cancellation must not start a continuation.
            if self.interrupted.load(Ordering::SeqCst) {
                return Ok(Vec::new());
            }
            // Internal sends reset per-stream observations; reasoning spans all tool rounds.
            let carried_reasoning = std::mem::take(&mut self.last_reasoning);
            self.send("").await?;
            self.last_reasoning = carried_reasoning;
        }
    }
}
