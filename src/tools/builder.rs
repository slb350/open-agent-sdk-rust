/// Builds a [`Tool`] by adding parameters or supplying a complete schema.
///
/// A later [`schema`](Self::schema) call replaces earlier parameters.
/// [`param`](Self::param) resets a non-object schema before adding a property.
/// Prefer one schema style per builder; full JSON Schema and shorthand properties
/// are not merged.
pub struct ToolBuilder {
    /// The tool's unique identifier
    name: String,

    /// Human-readable description of the tool's purpose
    description: String,

    /// The input schema, built up through .param() calls or set via .schema()
    schema: Value,
}

impl ToolBuilder {
    /// Creates a builder with an empty schema. See [`tool()`] for an example.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            schema: serde_json::json!({}),
        }
    }

    /// Replaces the current schema, including parameters added by [`param`](Self::param).
    ///
    /// Accepts the schema formats documented by [`Tool::new`].
    pub fn schema(mut self, schema: Value) -> Self {
        self.schema = schema;
        self
    }

    /// Adds a required parameter using a type name such as `string` or `number`.
    ///
    /// A non-object schema is reset to an empty object first. For optional parameters
    /// or complete JSON Schema, use [`schema`](Self::schema).
    pub fn param(mut self, name: &str, type_str: &str) -> Self {
        if !self.schema.is_object() {
            self.schema = serde_json::json!({});
        }

        let obj = self
            .schema
            .as_object_mut()
            .expect("BUG: schema should be an object after initialization");

        obj.insert(name.to_string(), Value::String(type_str.to_string()));

        self
    }

    /// Consumes the builder and attaches an asynchronous handler.
    ///
    /// The handler owns captured state, is `Send + Sync`, and returns a `Send` future.
    /// Clone shared state inside the closure before moving it into each future:
    ///
    /// ```rust
    /// use open_agent::tool;
    /// use serde_json::json;
    /// use std::sync::{Arc, atomic::{AtomicU32, Ordering}};
    ///
    /// let counter = Arc::new(AtomicU32::new(0));
    /// let increment = tool("increment", "Increment a shared counter")
    ///     .build(move |_| {
    ///         let counter = counter.clone();
    ///         async move {
    ///             Ok(json!({"count": counter.fetch_add(1, Ordering::SeqCst) + 1}))
    ///         }
    ///     });
    /// ```
    pub fn build<F, Fut>(self, handler: F) -> Tool
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        // Delegate to Tool::new which handles schema conversion and handler wrapping
        Tool::new(self.name, self.description, self.schema, handler)
    }
}
