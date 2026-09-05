/// An asynchronous JSON-to-JSON handler shared through [`Arc`].
///
/// [`Tool::new`] boxes the returned future and wraps the closure automatically.
/// The closure must be `Send + Sync`; its future must be `Send`.
pub type ToolHandler =
    Arc<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<Value>> + Send>> + Send + Sync>;
