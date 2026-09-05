/// An ordered registry of asynchronous lifecycle hooks.
///
/// The first handler returning `Some(HookDecision)` determines the result and
/// skips later handlers. Return `None` to let subsequent handlers run. Clones share
/// handlers through [`Arc`]; the three vectors also permit direct registration.
///
/// ```rust
/// use open_agent::{Hooks, HookDecision};
///
/// let hooks = Hooks::new()
///     .add_pre_tool_use(|event| async move {
///         if event.tool_name == "delete_file" {
///             return Some(HookDecision::block("Deletion is disabled"));
///         }
///         None
///     })
///     .add_post_tool_use(|event| async move {
///         println!("Completed {}", event.tool_name);
///         None
///     })
///     .add_user_prompt_submit(|event| async move {
///         let prompt = format!("{}\nAnswer in JSON.", event.prompt);
///         Some(HookDecision::modify_prompt(prompt, "Requested output format"))
///     });
/// ```
#[derive(Clone, Default)]
pub struct Hooks {
    /// Collection of PreToolUse hook handlers, executed in registration order
    pub pre_tool_use: Vec<PreToolUseHandler>,

    /// Collection of PostToolUse hook handlers, executed in registration order
    pub post_tool_use: Vec<PostToolUseHandler>,

    /// Collection of UserPromptSubmit hook handlers, executed in registration order
    pub user_prompt_submit: Vec<UserPromptSubmitHandler>,
}

impl Hooks {
    /// Creates an empty hook registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a pre-tool handler. See [`Hooks`] for ordering and an example.
    pub fn add_pre_tool_use<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(PreToolUseEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<HookDecision>> + Send + 'static,
    {
        self.pre_tool_use
            .push(Arc::new(move |event| Box::pin(handler(event))));
        self
    }

    /// Appends a post-tool handler. Return `None` from observers so later hooks run.
    pub fn add_post_tool_use<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(PostToolUseEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<HookDecision>> + Send + 'static,
    {
        self.post_tool_use
            .push(Arc::new(move |event| Box::pin(handler(event))));
        self
    }

    /// Appends a prompt handler. Return a decision to block or replace the prompt.
    pub fn add_user_prompt_submit<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(UserPromptSubmitEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<HookDecision>> + Send + 'static,
    {
        self.user_prompt_submit
            .push(Arc::new(move |event| Box::pin(handler(event))));
        self
    }

    /// Runs handlers in registration order, returning the first `Some` decision.
    ///
    /// Each handler receives the same event snapshot. Returns `None` if no handler
    /// decides, including when the registry is empty.
    pub async fn execute_pre_tool_use(&self, event: PreToolUseEvent) -> Option<HookDecision> {
        let (last, preceding) = self.pre_tool_use.split_last()?;
        for handler in preceding {
            let decision = handler(event.clone()).await;
            if decision.is_some() {
                return decision;
            }
        }
        last(event).await
    }

    /// Runs post-tool handlers with the same first-decision behavior as
    /// [`execute_pre_tool_use`](Self::execute_pre_tool_use).
    pub async fn execute_post_tool_use(&self, event: PostToolUseEvent) -> Option<HookDecision> {
        let (last, preceding) = self.post_tool_use.split_last()?;
        for handler in preceding {
            let decision = handler(event.clone()).await;
            if decision.is_some() {
                return decision;
            }
        }
        last(event).await
    }

    /// Executes all registered UserPromptSubmit hooks in order and returns the first decision.
    ///
    /// Identical in behavior to [`Self::execute_pre_tool_use`] but for UserPromptSubmit events.
    /// See that method for detailed documentation of the execution model.
    pub async fn execute_user_prompt_submit(
        &self,
        event: UserPromptSubmitEvent,
    ) -> Option<HookDecision> {
        let (last, preceding) = self.user_prompt_submit.split_last()?;
        for handler in preceding {
            let decision = handler(event.clone()).await;
            if decision.is_some() {
                return decision;
            }
        }
        last(event).await
    }
}

/// Shows handler counts, omitting the closures.
impl std::fmt::Debug for Hooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hooks")
            .field(
                "pre_tool_use",
                &format!("{} handlers", self.pre_tool_use.len()),
            )
            .field(
                "post_tool_use",
                &format!("{} handlers", self.post_tool_use.len()),
            )
            .field(
                "user_prompt_submit",
                &format!("{} handlers", self.user_prompt_submit.len()),
            )
            .finish()
    }
}

/// String constant for the PreToolUse hook event name.
///
/// This constant can be used for logging, metrics, or when you need a string
/// representation of the hook type. It's primarily used internally but is exposed
/// as part of the public API for consistency.
pub const HOOK_PRE_TOOL_USE: &str = "pre_tool_use";

/// String constant for the PostToolUse hook event name.
///
/// See [`HOOK_PRE_TOOL_USE`] for usage details.
pub const HOOK_POST_TOOL_USE: &str = "post_tool_use";

/// String constant for the UserPromptSubmit hook event name.
///
/// See [`HOOK_PRE_TOOL_USE`] for usage details.
pub const HOOK_USER_PROMPT_SUBMIT: &str = "user_prompt_submit";
