impl Client {
    /// Sends a pre-built message to the AI model.
    ///
    /// This method allows sending messages with images or custom content blocks
    /// that cannot be expressed as simple text prompts. Use the `Message` helper
    /// methods like [`user_with_image()`](Message::user_with_image),
    /// [`user_with_image_detail()`](Message::user_with_image_detail), or
    /// [`user_with_base64_image()`](Message::user_with_base64_image) to create
    /// messages with multimodal content.
    ///
    /// Unlike [`send()`](Client::send), this method:
    /// - Accepts pre-built `Message` objects instead of text prompts
    /// - Bypasses `UserPromptSubmit` hooks (since message is already constructed)
    /// - Enables multimodal interactions (text + images)
    ///
    /// After calling this method, use [`receive()`](Client::receive) to get the
    /// response content blocks.
    ///
    /// # Arguments
    ///
    /// * `message` - A pre-built message (typically created with `Message::user_with_image()` or similar helpers)
    ///
    /// # Errors
    ///
    /// Returns `Error` if:
    /// - Network request fails
    /// - Server returns an error
    /// - Response cannot be parsed
    /// - Request is interrupted via [`interrupt()`](Client::interrupt)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use open_agent::{Client, AgentOptions, Message, ImageDetail};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let options = AgentOptions::builder()
    ///     .model("gpt-4-vision-preview")
    ///     .base_url("http://localhost:1234/v1")
    ///     .build()?;
    ///
    /// let mut client = Client::new(options)?;
    ///
    /// // Send a message with an image
    /// let msg = Message::user_with_image(
    ///     "What's in this image?",
    ///     "https://example.com/photo.jpg"
    /// )?;
    /// client.send_message(msg).await?;
    ///
    /// // Receive the response
    /// while let Some(block) = client.receive().await? {
    ///     // Process response blocks
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn send_message(&mut self, message: Message) -> Result<()> {
        // Reset interrupt flag for new query
        // This allows the client to be reused after a previous interruption
        // Uses SeqCst ordering to ensure visibility across all threads
        self.interrupted.store(false, Ordering::SeqCst);

        // Discard any leftover manual-mode blocks from an abandoned stream.
        self.manual_receive_buffer.clear();
        self.current_stream = None;

        // Note: We do NOT run UserPromptSubmit hooks here because:
        // 1. The message is already fully constructed
        // 2. Hooks expect string prompts, not complex Message objects
        // 3. For multimodal messages, there's no single "prompt" to modify

        // Add message to history BEFORE sending request
        // This ensures history consistency even if request fails
        self.history.push(message);

        self.start_request().await
    }

}
