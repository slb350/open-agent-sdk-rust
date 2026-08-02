/// Type alias for tool handler functions.
///
/// ## Handler Anatomy
///
/// A tool handler is a complex type that enables dynamic async execution:
///
/// ```text
/// Arc<                                      // Thread-safe reference counting
///   dyn Fn(Value)                          // Function taking JSON arguments
///     -> Pin<Box<                           // Pinned heap allocation
///       dyn Future<Output = Result<Value>>  // Async computation
///         + Send>>                          // Can cross thread boundaries
///     + Send + Sync>                        // Handler itself is thread-safe
/// ```
///
/// ### Why Arc?
///
/// [`Arc`] (Atomic Reference Counted) allows multiple parts of the system to hold
/// references to the same handler without worrying about ownership. This is essential
/// because tools may be:
/// - Stored in an agent's tool registry
/// - Cloned when creating tool definitions for API calls
/// - Accessed concurrently by multiple agent threads
///
/// The atomic reference counting ensures thread-safe access without locks on the
/// handler reference itself (though the handler may still use internal synchronization).
///
/// ### Why Pin<Box<>>?
///
/// **Pinning** guarantees that the future won't be moved in memory after creation.
/// This is critical because async functions can create self-referential structures
/// (e.g., a future holding a reference to its own data). Moving such a structure
/// would invalidate internal pointers.
///
/// **Boxing** (heap allocation) enables:
/// - Storing futures of different concrete types (different handlers) in one container
/// - Having a predictable, small stack footprint (just a pointer, not the whole future)
/// - Dynamic dispatch - the actual future type is erased but still executable
///
/// ### Why Send + Sync?
///
/// - **Send**: The future can be sent across thread boundaries. Essential for
///   multi-threaded async runtimes (like Tokio) that may move tasks between threads.
///
/// - **Sync**: Multiple threads can safely hold references to the handler. This allows
///   tools to be called concurrently by different parts of the system.
///
/// ## Example Usage
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use std::pin::Pin;
/// use std::future::Future;
/// use serde_json::{json, Value};
/// use open_agent::Result;
///
/// // Define a handler that matches ToolHandler type
/// let handler: Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<Value>> + Send>> + Send + Sync> =
///     Arc::new(|args| {
///         Box::pin(async move {
///             // Handler implementation
///             Ok(json!({"status": "success"}))
///         })
///     });
///
/// // Can be cloned cheaply (only increments Arc counter)
/// let handler_clone = handler.clone();
/// ```
pub type ToolHandler =
    Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<Value>> + Send>> + Send + Sync>;
