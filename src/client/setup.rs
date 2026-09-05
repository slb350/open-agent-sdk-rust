impl Client {
    /// Creates a reusable HTTP client with the configured timeout and redirect policy.
    pub fn new(options: AgentOptions) -> Result<Self> {
        let http_client = model_http_client_builder(&options)
            .build()
            .map_err(|e| Error::config(format!("Failed to build HTTP client: {}", e)))?;

        Ok(Self {
            options,
            history: Vec::new(),
            current_stream: None,
            http_client,
            interrupted: Arc::new(AtomicBool::new(false)),
            auto_output: None,
            manual_receive_buffer: Vec::new(),
            last_finish_reason: None,
            last_reasoning: String::new(),
        })
    }

    async fn start_request(&mut self) -> Result<()> {
        self.last_finish_reason = None;
        self.last_reasoning.clear();
        let messages = request::history_messages(&self.options, &self.history);
        let request = request::build_request(&self.options, messages);
        self.current_stream =
            Some(stream_request(&self.http_client, &self.options, &request).await?);
        Ok(())
    }
}
