# Open Agent SDK (Rust)

> Build AI agents in Rust, over OpenAI chat completions or Anthropic messages

**What you can build:**

- **Copy editors** that analyze manuscripts and track writing patterns
- **Git commit generators** that write meaningful commit messages
- **Market analyzers** that research competitors and summarize findings
- **Code reviewers**, **data analysts**, **research assistants**, and more

**Why this SDK?**

- **Two protocols** - one API over OpenAI chat completions or Anthropic messages, chosen per endpoint
- **Local or hosted** - run on your own hardware at no API cost and with no data leaving the machine, or point it at a vendor
- **Control** - pick your model (Qwen, Llama, Mistral, Claude, etc.)

[![Crates.io](https://img.shields.io/crates/v/open-agent-sdk.svg?label=open-agent-sdk%200.11.2)](https://crates.io/crates/open-agent-sdk)
[![Documentation](https://docs.rs/open-agent-sdk/badge.svg)](https://docs.rs/open-agent-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Open Agent SDK (Rust) provides a clean, streaming API for working with local and cloud model servers over two wire protocols: OpenAI chat completions and Anthropic messages, selected per endpoint. It supports SSE streaming across transport boundaries, tool call aggregation, lifecycle hooks, and automatic tool execution.

**Streaming is tolerant of real-world servers.** SSE events are buffered across arbitrary HTTP
transport chunk boundaries, and anything still held is flushed when the stream ends — including
when a server closes the connection or sends `data: [DONE]` without ever setting
`finish_reason`, which llama.cpp, vLLM, and several local gateways do. Content is never
silently dropped.

**Every stream reports why it ended.** The stream from `query()` terminates with exactly one
`StreamEvent::Finish` carrying a `FinishReason`, so a response cut off at the token cap
(`Length`) is distinguishable from one the model chose to end (`Stop`) and from a server that
never said (`Unspecified`) — three cases that look identical from the content alone. On
`Client`, the same information is available from `client.finish_reason()`.

## Supported Providers

The protocol is a property of the endpoint, set with `.protocol(..)` and defaulting to
`ApiProtocol::OpenAiChat`.

### `ApiProtocol::OpenAiChat` — `POST {base_url}/chat/completions`, bearer auth

- **LM Studio** - `http://localhost:1234/v1`
- **Ollama** - `http://localhost:11434/v1`
- **llama.cpp server** - OpenAI-compatible mode
- **vLLM** - OpenAI-compatible API
- **Text Generation WebUI** - OpenAI extension
- **Any OpenAI-compatible local endpoint**
- **Cloud vendors** - OpenAI, OpenRouter, Azure OpenAI, z.ai (`https://api.z.ai/api/coding/paas/v4`)
- **Local gateways proxying cloud models** - e.g., Ollama or custom gateways that route to cloud providers

**Note on LM Studio:** LM Studio is particularly well-tested with this SDK and provides reliable OpenAI-compatible API support. If you're looking for a user-friendly local model server with excellent compatibility, LM Studio is highly recommended.

### `ApiProtocol::Anthropic` — `POST {base_url}/messages`, `x-api-key` + `anthropic-version`

- **Anthropic**
- **Moonshot Kimi for Coding** - `https://api.kimi.com/coding/v1`
- **MiniMax** - `https://api.minimax.io/anthropic/v1`

Extended thinking arrives on the existing reasoning channel (`StreamEvent::Reasoning`, opt in
with `.include_reasoning(true)`), and tool calls as ordinary `ContentBlock::ToolUse` blocks.

Third-party Anthropic-compatible endpoints are not uniform, and the SDK invents nothing on
their behalf: `api.kimi.com/coding/v1` requires `max_tokens` and answers a bare
`invalid_request_error` 400 without it, and accepts no `temperature` but 1. Set both
explicitly when an endpoint asks for them.

### Not Supported (Use Official SDKs)

- **Cloud provider SDK-only APIs** - Bedrock, Vertex, etc. (proxied via a compatible gateway is fine)

## Quick Start

### Installation

```toml
[dependencies]
open-agent-sdk = "0.11.2"
tokio = { version = "1", features = ["full"] }
futures = "0.3"
serde_json = "1.0"
```

For development:

```bash
git clone https://github.com/slb350/open-agent-sdk-rust.git
cd open-agent-sdk-rust
cargo build
```

### Custom Request Headers

Caller headers are stored as strings and applied to both `query()` and `Client` requests.
They replace SDK defaults with the same name case-insensitively; names not supplied retain
their protocol defaults.

OpenRouter attribution:

```rust
let options = AgentOptions::builder()
    .model(std::env::var("MODEL")?)
    .base_url("https://openrouter.ai/api/v1")
    .api_key(std::env::var("OPENROUTER_API_KEY")?)
    .header("HTTP-Referer", "https://example.com")
    .header("X-Title", "Example agent")
    .build()?;
```

Azure OpenAI `api-key` authentication uses an empty `api_key` to suppress the SDK bearer
header, then supplies the provider-specific header directly:

```rust
let options = AgentOptions::builder()
    .model(std::env::var("MODEL")?)
    .base_url(std::env::var("AZURE_OPENAI_BASE_URL")?)
    .api_key("")
    .header("api-key", std::env::var("AZURE_OPENAI_API_KEY")?)
    .build()?;
```

Calling `.header()` again with the same name replaces the earlier value. Invalid names or
values are rejected by `.build()` before a request can be sent.

Custom headers are sent only to the configured model origin. Model requests deliberately do
not follow HTTP redirects, including same-origin redirects; any `30x` response surfaces through
the normal API-status error path. Set `base_url` to the exact destination that should receive
the request and its credentials.

### Upgrading from 0.10.x

v0.11.0 is additive. Existing configurations keep their current protocol headers; callers
that use the new `.header(name, value)` method can replace those defaults case-insensitively.
The one explicit opt-in auth behavior is `.api_key("")`, which now omits the SDK auth header
so authentication can come entirely from caller headers.

### Upgrading from 0.9.x

v0.10.0 has one breaking change, and the compiler does **not** catch it: the types are
unchanged, only how many events carry the same text.

**Text and reasoning arrive fragment by fragment.** A response that used to arrive as one
`ContentBlock::Text` at the end of the stream now arrives as one block per delta, in order,
while the stream is still open — which is what the SDK has always claimed streaming meant.
Code that concatenates what it receives needs no change. Code that read the first text block
as the whole answer now reads a prefix of it:

```rust
// Before: happened to work, because there was only ever one text block.
if let Some(StreamEvent::Block(ContentBlock::Text(text))) = stream.next().await.transpose()? {
    println!("{}", text.text);   // now prints the first few tokens only
}

// After: join the fragments, printing them as they land.
let mut answer = String::new();
while let Some(event) = stream.next().await {
    if let StreamEvent::Block(ContentBlock::Text(text)) = event? {
        print!("{}", text.text);
        answer.push_str(&text.text);
    }
}
```

If you already collect blocks and want the old shape back, `coalesce_text_blocks` is the join
the SDK applies internally, now exported:

```rust
use open_agent::coalesce_text_blocks;

let whole = coalesce_text_blocks(&collected);   // adjacent text joined, tool calls untouched
```

Conversation history is unaffected — the fragments are joined before the assistant turn is
written, so the next request replays exactly what 0.9.x replayed. Tool calls are unaffected:
their arguments are only valid JSON once the last fragment lands, so they still emit whole at
the end of the stream, in ascending index order.

### Upgrading from 0.8.x

v0.9.0 has one breaking change, and the compiler catches it.

**`AgentOptions::temperature()` returns `Option<f32>`, and unset now means unset.** It used
to default to 0.7 and was always sent. A growing number of models reject the parameter
outright — Anthropic's range stops at 1.0, and Moonshot's `k3` answers
`only temperature 1 is allowed for this model` with a 400 — so a client-invented default
turns a working request into a hard error. `None` omits the field and the server decides,
exactly as `max_tokens` has behaved since 0.7.0. The builder is unchanged:
`.temperature(0.2)` still sets one. Callers that relied on the old default now pass it
explicitly.

Reaching an Anthropic messages endpoint is additive:

```rust
let options = AgentOptions::builder()
    .model("k3")
    .base_url("https://api.kimi.com/coding/v1")
    .api_key(&key)
    .protocol(ApiProtocol::Anthropic)
    .max_tokens(200_000)   // this endpoint requires it
    .build()?;
```

### Upgrading from 0.7.x

v0.8.0 had one breaking change, and the compiler caught it.

**`query()` yields `StreamEvent` instead of `ContentBlock`.** The stream needed room for
something that is not content: the reason generation stopped. Every stream now ends with
exactly one `StreamEvent::Finish`.

```rust
// Before (0.7.x)
while let Some(block) = stream.next().await {
    match block? {
        ContentBlock::Text(text) => print!("{}", text.text),
        _ => {}
    }
}

// After (0.8.0) — smallest possible edit
while let Some(event) = stream.next().await {
    match event?.into_block() {
        Some(ContentBlock::Text(text)) => print!("{}", text.text),
        _ => {}
    }
}
```

If you parse structured output, the reason you upgraded is the `Finish` event — match it
rather than discarding it:

```rust
use open_agent::{ContentBlock, FinishReason, StreamEvent};

let mut answer = String::new();
while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::Block(ContentBlock::Text(text)) => answer.push_str(&text.text),
        // Truncated at the token cap: the JSON is missing because generation ran out of
        // budget, not because the model refused. Retry with a larger cap.
        StreamEvent::Finish(FinishReason::Length) => return Err("truncated".into()),
        // The model finished and still did not produce JSON. Retrying will not help.
        StreamEvent::Finish(FinishReason::Stop) => {}
        // The server never said. Neither conclusion is available.
        StreamEvent::Finish(FinishReason::Unspecified) => {}
        _ => {}
    }
}
```

`ContentStream` was renamed `EventStream`. `ToolCallAggregator` was renamed `StreamAccumulator`
— if you imported it directly, update the import; the API is otherwise unchanged. `ContentBlock`
is unchanged — no new variant and no wire-shape change, so exhaustive matches over it and its
`serde` representation still work.

**`Client` is unaffected.** `client.receive()` still yields `ContentBlock`; the finish reason
is recorded on the client instead:

```rust
client.send("Reply with JSON.").await?;
while let Some(block) = client.receive().await? { /* ... */ }

if client.finish_reason().is_some_and(FinishReason::is_truncated) {
    // Retry with a larger max_tokens.
}
```

New in 0.8.0, with no action required: reasoning-model side channels
(`reasoning_content` on DeepSeek, `reasoning` on OpenRouter) are now explicitly parsed and
routed away from assistant text rather than dropped as unknown fields, so deliberation prose
can never be spliced into a response you parse as JSON. Opt into seeing it with
`.include_reasoning(true)`, which surfaces it as `StreamEvent::Reasoning` and
`client.reasoning()`.

### Upgrading from 0.6.x

v0.7.0 has two breaking changes. Most projects need no edits at all; the compiler catches
the first, and the second is a behaviour change with no compile error.

**1. `Error::Api` carries the HTTP status.** It changed from a tuple variant to a struct
variant, so any pattern match must be updated:

```rust
// Before (0.6.x)
if let Error::Api(msg) = &err { eprintln!("{msg}"); }

// After (0.7.0)
if let Error::Api { message, status } = &err {
    eprintln!("{message} (status: {status:?})");
}
```

Constructing errors is unchanged — `Error::api(msg)` still works and yields `status: None`.
Use the new `Error::api_status(status, msg)` when you have a status code, because
`is_retryable_error` classifies on the status and treats a statusless API error as permanent.

**2. `max_tokens` is no longer defaulted to 4096.** Leaving `.max_tokens()` unset now omits
the field from the request so the server applies its own limit. This is a silent behaviour
change: if you relied on the implicit cap, set it explicitly.

```rust
let options = AgentOptions::builder()
    .model("qwen2.5-32b-instruct")
    .base_url("http://localhost:1234/v1")
    .max_tokens(4096)  // add this to keep the old behaviour
    .build()?;
```

Leaving it unset is recommended for long-context and reasoning models, which a 4096-token
client-side cap truncates mid-response.

Also fixed in 0.7.0, with no action required: streamed content is no longer discarded when a
server ends its stream without ever sending `finish_reason` (llama.cpp, vLLM, and several
local gateways do this), and `429` is now correctly treated as retryable.

### Simple Query (LM Studio)

```rust
use open_agent::{query, AgentOptions, ContentBlock, FinishReason, StreamEvent};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = AgentOptions::builder()
        .system_prompt("You are a professional copy editor")
        .model("qwen2.5-32b-instruct")
        .base_url("http://localhost:1234/v1")
        .temperature(0.1)
        .build()?;

    let mut stream = query("Analyze this text...", &options).await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Block(ContentBlock::Text(text)) => print!("{}", text.text),
            // Always emitted, exactly once, as the last event.
            StreamEvent::Finish(FinishReason::Length) => eprintln!("\n[truncated]"),
            _ => {}
        }
    }

    Ok(())
}
```

### Multi-Turn Conversation (Ollama)

```rust
use open_agent::{Client, AgentOptions, ContentBlock};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant")
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .build()?;

    let mut client = Client::new(options)?;

    client.send("What's the capital of France?").await?;

    while let Some(block) = client.receive().await? {
        match block {
            ContentBlock::Text(text) => {
                print!("{}", text.text);
            }
            ContentBlock::ToolUse(tool_use) => {
                println!("Tool used: {}", tool_use.name());
                // Execute tool and add result
                // client.add_tool_result(tool_use.id(), result)?;
            }
            _ => {}
        }
    }

    Ok(())
}
```

### Function Calling with Tools

Define tools using the builder pattern for clean, type-safe function calling:

```rust
use open_agent::{tool, Client, AgentOptions, ContentBlock};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define tools
    let add_tool = tool("add", "Add two numbers")
        .param("a", "number")
        .param("b", "number")
        .build(|args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!({"result": a + b}))
        });

    // Enable automatic tool execution (recommended)
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant with access to tools.")
        .model("qwen2.5-32b-instruct")
        .base_url("http://localhost:1234/v1")
        .tool(add_tool)
        .auto_execute_tools(true)      // Tools execute automatically
        .max_tool_iterations(10)       // Safety limit for tool loops
        .build()?;

    let mut client = Client::new(options)?;
    client.send("What's 25 + 17?").await?;

    // Simply iterate - tools execute automatically!
    while let Some(block) = client.receive().await? {
        match block {
            ContentBlock::Text(text) => {
                println!("Response: {}", text.text);
            }
            _ => {}
        }
    }

    Ok(())
}
```

### Advanced: Manual Tool Execution

For custom execution logic or result interception:

```rust
// Disable auto-execution
let options = AgentOptions::builder()
    .system_prompt("You are a helpful assistant with access to tools.")
    .model("qwen2.5-32b-instruct")
    .base_url("http://localhost:1234/v1")
    .tool(add_tool.clone())
    .auto_execute_tools(false)  // Manual mode
    .build()?;

let mut client = Client::new(options)?;
client.send("What's 25 + 17?").await?;

while let Some(block) = client.receive().await? {
    match block {
        ContentBlock::ToolUse(tool_use) => {
            // You execute the tool manually
            let result = add_tool.execute(tool_use.input().clone()).await?;

            // Return result to agent
            client.add_tool_result(tool_use.id(), result)?;

            // Continue conversation
            client.send("").await?;
        }
        ContentBlock::Text(text) => {
            print!("{}", text.text);
        }
        _ => {}
    }
}
```

**Key Features:**

- **Automatic execution** - Tools run automatically with safety limits
- **Type-safe schemas** - Automatic JSON schema generation from parameters
- **Both protocols** - The same tool definitions serve OpenAI function calling and Anthropic tool use
- **Clean builder API** - Fluent API for tool definition
- **Hook integration** - PreToolUse/PostToolUse hooks work in both modes

See `examples/calculator_tools.rs` and `examples/auto_execution_demo.rs` for complete examples.

## Multimodal Vision Support

Send images alongside text to vision-capable models like llava, qwen-vl, or minicpm-v. The SDK formats images for whichever protocol the endpoint speaks.

### Simple Image + Text

```rust
use open_agent::{Client, Message, MessageRole, ContentBlock, TextBlock, ImageBlock, ImageDetail};

// From URL
let msg = Message::user_with_image(
    "What's in this image?",
    "https://example.com/photo.jpg"
)?;
client.send_message(msg).await?;

// From local file path (NEW!)
let msg = Message::new(
    MessageRole::User,
    vec![
        ContentBlock::Text(TextBlock::new("Describe this photo")),
        ContentBlock::Image(ImageBlock::from_file_path("/path/to/photo.jpg")?),
    ],
);
client.send_message(msg).await?;

// From base64 data
let msg = Message::user_with_base64_image(
    "Describe this diagram",
    base64_data,
    "image/png"
)?;
client.send_message(msg).await?;

// Control detail level for token costs
let msg = Message::user_with_image_detail(
    "Analyze the fine details",
    "https://example.com/diagram.png",
    ImageDetail::High  // Low: ~85 tokens, High: variable, Auto: default
)?;
client.send_message(msg).await?;
```

**Supported Image Sources:**

- **`ImageBlock::from_url(url)`** - HTTPS/HTTP URLs or data URIs (e.g., `data:image/png;base64,...`)
- **`ImageBlock::from_file_path(path)`** - Local filesystem (automatically encodes as base64)
  - Supports: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.svg`
  - MIME type inferred from file extension
  - File is read and encoded automatically
- **`ImageBlock::from_base64(data, mime)`** - Manual base64 with explicit MIME type

### Token Cost Management

Control image processing costs using `ImageDetail` levels:

- **`ImageDetail::Low`** - Lower resolution (typically more cost-effective)
- **`ImageDetail::High`** - Higher resolution (typically more detailed analysis)
- **`ImageDetail::Auto`** - Model decides (balanced default)

**⚠️ Token Costs Vary by Model:**

OpenAI's Vision API uses ~85 tokens (Low) and variable tokens based on dimensions (High), but **local models may have completely different token costs**—or no token costs for images at all. The `ImageDetail` setting may even be ignored by some models.

**Always benchmark your specific model** instead of relying on OpenAI's published values for capacity planning.

### Complex Multi-Image Messages

```rust
use open_agent::{Message, MessageRole, ContentBlock, TextBlock, ImageBlock, ImageDetail};

let msg = Message::new(
    MessageRole::User,
    vec![
        ContentBlock::Text(TextBlock::new("Compare these images:")),
        ContentBlock::Image(
            ImageBlock::from_url("https://example.com/before.jpg")?
                .with_detail(ImageDetail::Low)
        ),
        ContentBlock::Image(
            ImageBlock::from_url("https://example.com/after.jpg")?
                .with_detail(ImageDetail::Low)
        ),
    ],
);
```

**Key Features:**

- **`send_message()` API** - Send pre-built messages with images via `client.send_message(msg).await?`
- **Automatic serialization** - Images converted to OpenAI Vision or Anthropic image blocks (`ImageDetail` has no Anthropic equivalent and is dropped there)
- **Multiple sources** - URLs, local file paths, or base64 data
- **Backward compatible** - Text-only messages still work with `send("text")`
- **Data URIs supported** - Base64-encoded images transmitted seamlessly
- **Token cost control** - Choose detail level based on use case

See `examples/vision_example.rs` for comprehensive working examples including local file paths.

## Context Management

Local models have fixed context windows (typically 8k-32k tokens). The SDK provides utilities for manual history management—no silent mutations, you stay in control.

### Token Estimation & Truncation

```rust
use open_agent::{Client, AgentOptions, estimate_tokens, is_approaching_limit, truncate_messages};

let mut client = Client::new(options)?;

// Long conversation...
for i in 0..50 {
    client.send(&format!("Question {}", i)).await?;
    while let Some(block) = client.receive().await? {
        // Process blocks
        let _ = block;
    }
}

// Check token usage
let tokens = estimate_tokens(client.history());
println!("Context size: ~{} tokens", tokens);

// Check if approaching limit (margin = 0.8 means warn at 80% of limit)
if is_approaching_limit(client.history(), 32000, 0.8) {
    println!("Warning: approaching context limit");
}

// Manually truncate when needed
if tokens > 28000 {
    let truncated = truncate_messages(client.history(), 10, true);
    *client.history_mut() = truncated;
}
```

### Recommended Patterns

**1. Stateless Agents** (Best for single-task agents):

```rust
// Process each task independently - no history accumulation
for task in tasks {
    let mut client = Client::new(options.clone());
    client.send(&task).await?;
    // Client dropped, fresh context for next task
}
```

**2. Manual Truncation** (At natural breakpoints):

```rust
use open_agent::truncate_messages;

let mut client = Client::new(options)?;
for task in tasks {
    client.send(&task).await?;
    // Truncate after each major task
    let truncated = truncate_messages(client.history(), 5, false);
    *client.history_mut() = truncated;
}
```

**3. External Memory** (RAG-lite for research agents):

```rust
// Store important facts in database, keep conversation context small
let mut database = HashMap::new();
let mut client = Client::new(options)?;

client.send("Research topic X").await?;
// Save response to database
database.insert("topic_x", extract_facts(&response));

// Clear history, query database when needed
let truncated = truncate_messages(client.history(), 0, false);
*client.history_mut() = truncated;
```

### Why Manual?

The SDK **intentionally** does not auto-compact history because:

- **Domain-specific needs**: Copy editors need different strategies than research agents
- **Token accuracy varies**: Each model family has different tokenizers
- **Risk of breaking context**: Silently removing messages could break tool chains
- **Natural limits exist**: Compaction doesn't bypass model context windows

See `examples/context_management.rs` for complete patterns and usage.

## Lifecycle Hooks

Monitor and control agent behavior at key execution points with zero-cost Rust hooks.

### Quick Example

```rust
use open_agent::{
    AgentOptions, Client, Hooks,
    PreToolUseEvent, PostToolUseEvent,
    HookDecision,
};

// Security gate - block dangerous operations
let hooks = Hooks::new()
    .add_pre_tool_use(|event| async move {
        if event.tool_name == "delete_file" {
            return Some(HookDecision::block("Delete operations require approval"));
        }
        Some(HookDecision::continue_())
    })
    .add_post_tool_use(|event| async move {
        // Audit logger - track all tool executions
        println!("Tool executed: {} -> {:?}", event.tool_name, event.tool_result);
        None
    });

// Register hooks in AgentOptions
let options = AgentOptions::builder()
    .system_prompt("You are a helpful assistant")
    .model("qwen2.5-32b-instruct")
    .base_url("http://localhost:1234/v1")
    .hooks(hooks)
    .build()?;

let mut client = Client::new(options)?;
```

### Hook Types

**PreToolUse** - Fires before tool execution

- **Block operations**: Return `Some(HookDecision::block(reason))`
- **Modify inputs**: Return `Some(HookDecision::modify_input(json!({}), reason))`
- **Allow**: Return `Some(HookDecision::continue_())`

**PostToolUse** - Fires after the tool completes and before the final result is committed

- **Observational** (tool already executed)
- Use for audit logging, metrics, result validation
- Return `None` or `Some(HookDecision::...)`

Every hook event exposes `history` as `Vec<serde_json::Value>`, with one structured
JSON object per internal `Message` (`role` plus typed `content` blocks). Prompt and
pre-tool snapshots contain history up to that lifecycle point; post-tool snapshots
also include the completed tool call and its unmodified result.

**UserPromptSubmit** - Fires before sending prompt to API

- **Block prompts**: Return `Some(HookDecision::block(reason))`
- **Modify prompts**: Return `Some(HookDecision::modify_prompt(text, reason))`
- **Allow**: Return `Some(HookDecision::continue_())`

### Common Patterns

#### Pattern 1: Redirect to Sandbox

```rust
hooks.add_pre_tool_use(|event| async move {
    if event.tool_name == "file_writer" {
        let path = event.tool_input.get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if !path.starts_with("/tmp/") {
            let safe_path = format!("/tmp/sandbox/{}", path.trim_start_matches('/'));
            let mut modified = event.tool_input.clone();
            modified["path"] = json!(safe_path);
            return Some(HookDecision::modify_input(modified, "Redirected to sandbox"));
        }
    }
    Some(HookDecision::continue_())
})
```

#### Pattern 2: Compliance Audit Log

```rust
let audit_log = Arc::new(Mutex::new(Vec::new()));
let log_clone = audit_log.clone();

// Note: add_post_tool_use consumes and returns Hooks (builder pattern) — always rebind
let hooks = hooks.add_post_tool_use(move |event| {
    let log = log_clone.clone();
    async move {
        log.lock().unwrap().push(format!(
            "{} -> {:?}",
            event.tool_name,
            event.tool_result
        ));
        None
    }
});
```

### Hook Execution Flow

- Hooks run **sequentially** in the order registered
- **First non-None decision wins** (short-circuit behavior)
- Hooks run **inline on async runtime** (spawn tasks for heavy work)
- Works with both **Client** and **query()** function

See `examples/hooks_example.rs` and `examples/multi_tool_agent.rs` for comprehensive patterns.

## Interrupt Capability

Cancel long-running operations cleanly without corrupting client state. Perfect for timeouts, user cancellations, or conditional interruptions.

### Interrupt Quick Example

```rust
use open_agent::{Client, AgentOptions};
use tokio::time::{timeout, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = AgentOptions::builder()
        .system_prompt("You are a helpful assistant.")
        .model("qwen2.5-32b-instruct")
        .base_url("http://localhost:1234/v1")
        .build()?;

    let mut client = Client::new(options)?;
    client.send("Write a detailed 1000-word essay...").await?;

    // Timeout after 5 seconds
    match timeout(Duration::from_secs(5), async {
        while let Some(block) = client.receive().await? {
            // Process blocks
            let _ = block;
        }
        Ok::<_, Box<dyn std::error::Error>>(())
    }).await {
        Ok(_) => println!("Completed"),
        Err(_) => {
            client.interrupt();  // Clean cancellation
            println!("Operation timed out!");
        }
    }

    // Client is still usable after interrupt
    client.send("Short question?").await?;
    // Continue using client...

    Ok(())
}
```

### Common Interrupt Patterns

#### 1. Conditional Interruption

```rust
let mut full_text = String::new();
while let Some(block) = client.receive().await? {
    if let ContentBlock::Text(text) = block {
        full_text.push_str(&text.text);
        if full_text.contains("error") {
            client.interrupt();
            break;
        }
    }
}
```

#### 2. Concurrent Cancellation

```rust
use std::sync::atomic::Ordering;

let interrupt_handle = client.interrupt_handle();
let cancel_task = tokio::spawn(async move {
    tokio::time::sleep(Duration::from_secs(2)).await;
    interrupt_handle.store(true, Ordering::SeqCst);
});

while let Some(block) = client.receive().await? {
    // Process blocks until cancellation is observed
    let _ = block;
}

cancel_task.await?;
```

### How It Works

When you call `client.interrupt()`:

1. **Atomic signal** - A thread-safe flag tells the receive loop to stop
2. **Stream cleanup** - `receive()` observes the flag, drops the active stream, and returns `Ok(None)`
3. **Clean history** - Partial responses are discarded instead of committing incomplete assistant messages
4. **Idempotent** - Safe to call multiple times
5. **Cross-task safe** - `interrupt_handle()` lets another task cancel without locking the `Client`

Cancellation is observed between stream events, tool calls, and hook phases. A pending
network read or running tool/hook future must finish first. Auto mode retains completed
tool results and marks skipped calls as cancelled, keeping history valid for a later send.

See `examples/interrupt_demo.rs` for comprehensive patterns.

## Practical Examples

Example agents demonstrating real-world usage:

### Git Commit Agent

**[examples/git_commit_agent.rs](examples/git_commit_agent.rs)**

Analyzes your staged git changes and writes professional commit messages following conventional commit format.

```bash
# Stage your changes
git add .

# Run the agent
cargo run --example git_commit_agent

# Output:
# Found staged changes in 3 file(s)
# Analyzing changes and generating commit message...
#
# Suggested commit message:
# feat(auth): Add OAuth2 integration with refresh tokens
#
# - Implement token refresh mechanism
# - Add secure cookie storage for tokens
# - Update login flow to support OAuth2 providers
```

**Features:**

- Analyzes diff to determine commit type (feat/fix/docs/etc)
- Writes clear, descriptive commit messages
- Follows conventional commit standards

### Log Analyzer Agent

**[examples/log_analyzer_agent.rs](examples/log_analyzer_agent.rs)**

Intelligently analyzes application logs to identify patterns, errors, and provide actionable insights.

```bash
# Analyze a log file
cargo run --example log_analyzer_agent -- /var/log/app.log
```

**Features:**

- Automatic error pattern detection
- Time-based analysis (peak error times)
- Root cause suggestions
- Supports multiple log formats

### Why These Examples?

These agents demonstrate:

- **Practical Value**: Solve real problems developers face daily
- **Tool Integration**: Show how to integrate with system commands (git, file I/O)
- **Structured Output**: Parse and format LLM responses for actionable results
- **Privacy-First**: Keep your code and logs local while getting AI assistance

## Why Not Just Use OpenAI Client?

**Without open-agent-sdk** (raw reqwest):

```rust
use reqwest::Client;

let client = Client::new();
let response = client
    .post("http://localhost:1234/v1/chat/completions")
    .json(&json!({
        "model": "qwen2.5-32b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": true
    }))
    .send()
    .await?;

// Complex parsing of SSE chunks
// Extract delta content
// Handle tool calls manually
// Track conversation state yourself
```

**With open-agent-sdk**:

```rust
use open_agent::{query, AgentOptions};

let options = AgentOptions::builder()
    .system_prompt(system_prompt)
    .model("qwen2.5-32b-instruct")
    .base_url("http://localhost:1234/v1")
    .build()?;

let mut stream = query(user_prompt, &options).await?;
// Clean message types (TextBlock, ToolUseBlock)
// Automatic streaming and tool call handling
// Terminating StreamEvent::Finish tells you why generation stopped
```

**Value**: Familiar patterns + Less boilerplate + Rust performance

## Why Rust?

**Performance**: Zero-cost abstractions mean no runtime overhead. Streaming responses with Tokio delivers throughput comparable to C/C++ while maintaining memory safety.

**Safety**: Compile-time guarantees prevent data races, null pointer dereferences, and buffer overflows. Your agents won't crash from memory issues.

**Concurrency**: Fearless concurrency with `async`/`await` lets you run multiple agents or handle hundreds of concurrent requests without fear of race conditions.

**Production Ready**: Strong type system catches bugs at compile time. Comprehensive error handling with `Result` types. No surprises in production.

**Small Binaries**: Standalone executables under 10MB. Deploy anywhere without runtime dependencies.

## API Reference

### AgentOptions

```rust
AgentOptions::builder()
    .system_prompt(str)                  // System prompt
    .model(str)                          // Model name (required)
    .base_url(str)                       // Endpoint URL; the path comes from .protocol() (required)
    .tool(Tool)                          // Add a single tool for function calling
    .tools(Vec<Tool>)                    // Add multiple tools at once
    .hooks(Hooks)                        // Lifecycle hooks for monitoring/control
    .auto_execute_tools(bool)            // Enable automatic tool execution
    .max_tool_iterations(u32)            // Max automatic Client tool rounds
    .max_tokens(u32)                     // Tokens to generate (unset: omitted, server decides); getter returns Option<u32>
    .max_turns(u32)                      // Legacy stored value; does not limit execution
    .temperature(f32)                    // Sampling temperature (unset: omitted, server decides)
    .protocol(ApiProtocol)               // Wire protocol (default: ApiProtocol::OpenAiChat)
    .timeout(u64)                        // Request timeout in seconds (default: 60)
    .api_key(str)                        // API key (default: "not-needed")
    .header(name, value)                 // Add or replace a model-request header
    .include_reasoning(bool)             // Surface reasoning as StreamEvent::Reasoning (default: false)
    .build()?
```

### query()

Simple single-turn query function.

```rust
pub async fn query(prompt: &str, options: &AgentOptions) -> Result<EventStream>

// where
pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>;
```

Returns a stream yielding `StreamEvent` items. Use `futures::StreamExt` to iterate.

### StreamEvent and FinishReason

```rust
pub enum StreamEvent {
    Block(ContentBlock),   // Assistant text, or a fully assembled tool call
    Reasoning(String),     // Chain of thought; only with .include_reasoning(true)
    Finish(FinishReason),  // Exactly once per stream, always last
}

pub enum FinishReason {
    Stop,               // "stop"           — completed naturally
    Length,             // "length"         — cut off at the token limit
    ToolCalls,          // "tool_calls"     — finished in order to call tools
    ContentFilter,      // "content_filter" — halted by a content filter
    Other(String),      // anything else, preserved verbatim
    MaxToolIterations,  // the SDK's auto-execution loop hit max_tool_iterations
    Unspecified,        // the stream ended and the server never said
}
```

Both are `#[non_exhaustive]`; match with a `_` arm. `StreamEvent` provides `as_block()`,
`into_block()`, `as_text()`, `as_reasoning()`, and `finish_reason()`. `FinishReason` provides
`from_wire()`, `as_str()`, `is_truncated()`, and `Display`.

`Unspecified` is not an error — it is the normal behaviour of llama.cpp, vLLM, and several
local gateways, which stream content and then close without setting `finish_reason`. It is
kept distinct from `Stop` because "the model finished" and "the SDK has no information" call
for different handling.

`MaxToolIterations` is the one variant that does not come from a server: in auto-execution
mode the SDK, not the model, ends the run when it hits `max_tool_iterations`. It is reported
only by `client.finish_reason()` and never appears in a `StreamEvent::Finish`.

### Client

Multi-turn conversation client with tool monitoring.

```rust
let mut client = Client::new(options)?;
client.send(prompt).await?;

while let Some(block) = client.receive().await? {
    // Process ContentBlock items
    let _ = block;
}
```

**Additional Client methods:**

```rust
// Send a pre-built Message (e.g., with images)
client.send_message(msg).await?;

// Access the AgentOptions this client was created with
let opts = client.options();

// Clear conversation history and pending output, retaining configuration
client.clear_history();

// Look up a registered tool by name
if let Some(t) = client.get_tool("my_tool") { /* ... */ }

// Obtain a shareable interrupt handle (Arc<AtomicBool>) for use across tasks
let handle = client.interrupt_handle();

// Why the most recent stream stopped; None until one completes, reset on the next send()
if let Some(reason) = client.finish_reason() { println!("stopped: {reason}"); }

// Reasoning captured from the most recent turn (requires .include_reasoning(true));
// accumulates across every round of an auto-execution tool loop
if let Some(reasoning) = client.reasoning() { println!("thought: {reasoning}"); }
```

### MessageRole

Who sent a message. Used when constructing `Message` values directly.

```rust
use open_agent::MessageRole;

MessageRole::System     // Establishes context and instructions
MessageRole::User       // Input from the human or calling application
MessageRole::Assistant  // Response from the AI model
MessageRole::Tool       // Results from tool/function execution
```

### Message

Pre-built message values (for `client.send_message()`). Convenience constructors:

```rust
use open_agent::{Message, MessageRole, ContentBlock, TextBlock};

// Build a message manually (any role)
Message::new(role: MessageRole, content: Vec<ContentBlock>) -> Self

// Convenience constructors — all return Self (infallible):
Message::user(text: &str) -> Self
Message::assistant(text: &str) -> Self
Message::system(text: &str) -> Self
Message::user_with_blocks(blocks: Vec<ContentBlock>) -> Self

// Vision constructors — return Result<Self>:
Message::user_with_image(text: &str, image_url: &str) -> Result<Self>
Message::user_with_image_detail(text: &str, image_url: &str, detail: ImageDetail) -> Result<Self>
Message::user_with_base64_image(text: &str, base64_data: &str, mime: &str) -> Result<Self>
```

### Message Types

- `ContentBlock::Text(TextBlock)` - Text content from model
- `ContentBlock::Image(ImageBlock)` - Image content (for vision models)
- `ContentBlock::ToolUse(ToolUseBlock)` - Tool calls from model
- `ContentBlock::ToolResult(ToolResultBlock)` - Tool execution results

### Tool System

```rust
use open_agent::tool;

let my_tool = tool("name", "description")
    .param("param_name", "type")
    .build(|args| async move {
        // Tool implementation
        Ok(json!({"result": value}))
    });
```

For full JSON Schema control, use `.schema()` instead of chaining `.param()` calls:

```rust
let my_tool = tool("name", "description")
    .schema(json!({
        "type": "object",
        "properties": { "x": { "type": "number" } },
        "required": ["x"]
    }))
    .build(|args| async move { Ok(json!({})) });
```

### ToolBuilder

The `tool()` function returns a `ToolBuilder` for fluent construction of tool definitions:

```rust
use open_agent::{tool, ToolBuilder, Tool};

let t: Tool = tool("name", "description")
    .param("arg", "string")
    .build(|args| async move { Ok(json!({})) });
```

### Provider Configuration

Helper types and functions for mapping provider names to their default endpoints:

```rust
use open_agent::{Provider, get_base_url, get_model};

// get_base_url(provider: Option<Provider>, fallback: Option<&str>) -> String
let url = get_base_url(Some(Provider::LMStudio), None);   // http://localhost:1234/v1
let url_with_fallback = get_base_url(None, Some("http://localhost:8080/v1"));

// get_model(fallback: Option<&str>, prefer_env: bool) -> Option<String>
let model = get_model(Some("qwen2.5-32b"), false);  // use provided model
let env_model = get_model(None, true);              // prefer OPEN_AGENT_MODEL env var
```

### Wire Types

Low-level serialization types matching each protocol's request and streaming format, exported for callers that need to name what goes over the wire:

```rust
// OpenAI chat completions
use open_agent::{
    OpenAIContent, OpenAIContentPart, OpenAIFunction, OpenAIMessage, OpenAIRequest, OpenAIToolCall,
};

// Anthropic messages
use open_agent::{
    AnthropicBlockStart, AnthropicDelta, AnthropicErrorBody, AnthropicEvent, AnthropicMessage,
    AnthropicMessageDelta, AnthropicRequest, anthropic_finish_reason,
};
```

`AnthropicRequest::from_openai` takes an `OpenAIRequest`, which is why the OpenAI request half is exported alongside the Anthropic types. `anthropic_finish_reason` maps Anthropic stop reasons onto `FinishReason`, and `query()`/`Client` apply it for you.

### Error and Result Types

```rust
use open_agent::{Error, Result};
```

`Error` is the SDK's unified error type; `Result<T>` is an alias for `std::result::Result<T, Error>`.

| Variant | Meaning |
| --- | --- |
| `Http(reqwest::Error)` | Transport failure — connection refused, DNS, TLS, network timeout |
| `Json(serde_json::Error)` | Serialization or deserialization failure |
| `Config(String)` | Invalid configuration caught by `AgentOptions::build()` |
| `Api { status: Option<u16>, message: String }` | Error response from the model server |
| `Stream(String)` | SSE parsing or stream processing failure |
| `Tool(String)` | Tool execution or registration failure |
| `InvalidInput(String)` | User-provided input failed validation |
| `Timeout` | Request exceeded the configured timeout |
| `Other(String)` | Anything else |

`Api` carries the HTTP status as structured data so retry logic never has to parse the
message text:

```rust
use open_agent::Error;

// From an HTTP error response — this is what the client constructs internally
let err = Error::api_status(429, "Rate limit exceeded");
assert_eq!(err.status_code(), Some(429));
assert_eq!(err.to_string(), "API error 429: Rate limit exceeded");

// Without a status
let err = Error::api("Model 'gpt-4' not found");
assert_eq!(err.status_code(), None);
```

`status_code()` returns `None` for every non-`Api` variant, so it is safe to call on any error.

### Newtype Wrappers

Strong-typed wrappers used internally by `AgentOptions` and exported for external use:

```rust
use open_agent::{BaseUrl, ModelName, Temperature};
```

### Retry Module

Exponential-backoff retry utilities, exported as a public module:

```rust
use open_agent::retry::{RetryConfig, retry_with_backoff, retry_with_backoff_conditional, is_retryable_error};

// Configure retry behavior (builder pattern)
let config = RetryConfig::default()          // 3 attempts, exponential backoff
    .max_attempts(5)
    .initial_delay_ms(100)
    .max_delay_ms(5000)
    .backoff_multiplier(2.0);

// Retry any async operation
let result = retry_with_backoff(config.clone(), || async {
    some_fallible_operation().await
}).await?;

// Retry only transient failures; anything else fails on the first attempt
let result = retry_with_backoff_conditional(config, || async {
    some_fallible_operation().await
}).await?;

// Check if an SDK error is worth retrying
let retryable = is_retryable_error(&some_error);
```

`is_retryable_error` treats network errors, timeouts, and stream errors as transient. API
errors are classified on `Error::status_code()`, which reads the status `Error::Api` carries as
structured data; the retryable set is **408, 429, 500, 502, 503, 504, 529**. Everything else —
including API errors raised without a status — is non-retryable, so a `400 Bad Request` fails
immediately rather than burning the full attempt budget.

A mid-stream Anthropic `error` event arrives on a response that already returned 200, so it
carries no status of its own. The SDK maps the two transient kinds onto the statuses they
would have had earlier — `overloaded_error` to 529 and `rate_limit_error` to 429, with
`api_error` to 500 — which is what lets `retry_with_backoff` see them as retryable.

```rust
use open_agent::Error;

let err = Error::api_status(429, "Rate limit exceeded");   // status: Some(429)
assert_eq!(err.status_code(), Some(429));

let err = Error::api("Model 'gpt-4' not found");            // status: None
assert_eq!(err.status_code(), None);
```

### Prelude Import

For convenience, import the most commonly used types at once:

```rust
use open_agent::prelude::*;
```

### Hook Name Constants

String constants for hook event types are exported for use in custom registries:

```rust
use open_agent::{HOOK_PRE_TOOL_USE, HOOK_POST_TOOL_USE, HOOK_USER_PROMPT_SUBMIT};
```

### Context Utilities

```rust
use open_agent::{estimate_tokens, is_approaching_limit, truncate_messages};

// Estimate tokens in message history (character-based approximation)
let tokens = estimate_tokens(client.history());

// Check if approaching a context limit (margin=0.8 means 80% of limit)
let near_limit = is_approaching_limit(client.history(), 32000, 0.8);

// Truncate history, keeping the last N messages (preserve_system=true keeps system prompt)
let truncated = truncate_messages(client.history(), 10, true);
```

## Recommended Models

**Local models** (LM Studio, Ollama, llama.cpp):

- **GPT-OSS-120B** - Best in class for speed and quality
- **Qwen 3 30B** - Excellent instruction following, good for most tasks
- **GPT-OSS-20B** - Solid all-around performance
- **Mistral 7B** - Fast and efficient for simple agents

**Cloud-proxied via local gateway**:

- **kimi-k2:1t-cloud** - Tested and working via Ollama gateway
- **deepseek-v3.1:671b-cloud** - High-quality reasoning model
- **qwen3-coder:480b-cloud** - Code-focused models

## Project Structure

```text
open-agent-sdk-rust/
├── src/
│   ├── client.rs          # Public client module docs/imports and fragment orchestration
│   ├── client/            # Query, send, send_message, setup, streaming, receive, history, state, and tests
│   ├── config.rs          # Provider helpers (Provider, get_base_url, get_model)
│   ├── context.rs         # Token estimation and truncation
│   ├── error.rs           # Error types
│   ├── hooks.rs           # Public lifecycle-hook module orchestration
│   ├── hooks/             # Hook events, decisions, handlers, registry, and tests
│   ├── lib.rs             # Public exports and prelude module
│   ├── retry.rs           # Retry logic with exponential backoff
│   ├── retry/             # Retry unit tests
│   ├── tools.rs           # Public tool module orchestration
│   ├── tools/             # Tool, schema, builder, handler, factory, and tests
│   ├── types.rs           # Public core-type module orchestration
│   ├── types/             # Options, messages, images, OpenAI + Anthropic wire types, ApiProtocol
│   ├── utils.rs           # SSE parsing, stream accumulation, and the shared stream driver
│   └── utils/             # accumulator.rs + anthropic_accumulator.rs (wire decoding),
│                          # buffers.rs (the shared drain), coalesce.rs (text joining for
│                          # history), driver.rs, sse.rs
├── examples/
│   ├── simple_query.rs              # Basic streaming query
│   ├── anthropic_query.rs           # Anthropic messages endpoint via ApiProtocol
│   ├── calculator_tools.rs          # Function calling (manual mode)
│   ├── auto_execution_demo.rs       # Automatic tool execution
│   ├── multi_tool_agent.rs          # Production agent with 5 tools and hooks
│   ├── hooks_example.rs             # Lifecycle hooks patterns
│   ├── context_management.rs        # Context management patterns
│   ├── interrupt_demo.rs            # Interrupt capability patterns
│   ├── git_commit_agent.rs          # Production: Git commit generator
│   ├── log_analyzer_agent.rs        # Production: Log analyzer
│   ├── advanced_patterns.rs         # Retry logic and concurrent requests
│   ├── vision_example.rs            # Multimodal: URLs, local files, base64
│   ├── vision_api_demo.rs           # Vision API walkthrough
│   └── test_tool_serialization.rs   # Tool call serialization verification
├── benches/
│   └── performance.rs               # Criterion benchmarks (token estimation, history ops)
├── tests/                         # Wire protocol, lifecycle, validation, and infrastructure tests
│   └── common/mod.rs                # Shared loopback-server and stream helpers
├── scripts/
│   ├── mutants-ci-scope.py          # Complete event diff policy for CI mutation runs
│   ├── mutants-common.sh            # The one definition of the results directory
│   ├── mutants-run.sh               # Owns the verdict (missed.txt); called by the hook and CI
│   ├── mutants-remote.sh            # rsync + ssh to a build host, falls back loudly
│   └── mutants-staged.sh            # Staged-diff scope, through mutants-remote.sh
├── .githooks/
│   └── pre-commit                   # fmt, clippy, tests, and a --in-diff cargo-mutants sweep
├── .github/
│   ├── dependabot.yml               # Grouped weekly Cargo dependency updates
│   └── workflows/
│       ├── ci.yml                   # GitHub CI (fmt, clippy, MSRV, Linux/macOS stable + beta matrix, security audit, mutation sweep, docs, LLVM Tarpaulin coverage, benchmarks)
│       └── scheduled-audit.yml      # Scheduled dependency audit
├── .markdownlint.json               # Markdown lint rules (disable MD013, allow duplicate sibling headings)
├── Cargo.toml
├── Cargo.lock
├── CHANGELOG.md
└── README.md
```

## Examples

### Production Agents

- **`git_commit_agent.rs`** – Analyzes git diffs and writes professional commit messages
- **`log_analyzer_agent.rs`** – Parses logs, finds patterns, suggests fixes
- **`multi_tool_agent.rs`** – Complete production setup with 5 tools, hooks, and auto-execution

### Core SDK Usage

- `simple_query.rs` – Minimal streaming query (simplest quickstart)
- `anthropic_query.rs` – Same query against an Anthropic messages endpoint via `ApiProtocol`
- `calculator_tools.rs` – Manual tool execution pattern
- `auto_execution_demo.rs` – Automatic tool execution pattern
- `vision_example.rs` – Multimodal image support (URLs, local files, base64)
- `vision_api_demo.rs` – Vision API walkthrough with token cost notes
- `hooks_example.rs` – Lifecycle hooks patterns (security gates, audit logging)
- `context_management.rs` – Manual history management patterns
- `interrupt_demo.rs` – Interrupt capability patterns (timeout, conditional, concurrent)
- `advanced_patterns.rs` – Retry logic and concurrent request handling
- `test_tool_serialization.rs` – Verifies tool call serialization (see `examples/test_tool_serialization.rs`)

## Documentation

- [API Documentation](https://docs.rs/open-agent-sdk)
- [Python SDK](https://github.com/slb350/open-agent-sdk) - Reference implementation
- [Examples](examples/) - Comprehensive usage examples

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_agent_options_builder

# Mutation sweep (must report zero survivors)
./scripts/mutants-remote.sh
```

Tests exercise request bodies on loopback servers, stream events, tool/hook outcomes,
validation boundaries, and client lifecycle transitions. One provider smoke test is
ignored by default and requires an explicitly configured local server. Run `cargo test`
for the current results instead of relying on a manually maintained test count.

Mutation testing checks whether the suite detects changed behavior. Ordinary CI runs a
full sweep when the complete pushed or pull-request diff adds Rust tests or doctests;
manual dispatch and the monthly schedule also run a full sweep. Other revisions run
the fast policy check. To run the same check before each commit:

```bash
git config core.hooksPath .githooks
```

The hook runs `cargo fmt --all -- --check`,
`cargo clippy --all-targets --all-features -- -D warnings`,
`cargo test --all-features --all`, and a `cargo mutants --in-diff` sweep scoped to the staged
Rust changes. Both the hook and CI reach their verdict through `scripts/mutants-run.sh`, which
reads `missed.txt` rather than the exit code — cargo-mutants reports a timeout in preference
to a survivor, so a run with one of each would otherwise look like a timeout.

## Requirements

- Rust 1.85+
- Tokio 1.50+ (async runtime)
- serde, serde_json (serialization)
- reqwest (HTTP client)
- futures, tokio-stream (async streams)
- eventsource-stream (SSE parsing)
- async-trait (async trait support)
- thiserror 2.0 + anyhow 1.0.103+ (error handling)
- log 0.4.29+ (logging)
- base64 0.23 (multimodal image encoding)
- wiremock =0.6.4 (dev-only: 0.6.5 uses syntax unavailable on the Rust 1.85 MSRV)
- cargo-mutants 27.1.0 (dev-only: mutation testing gate)
- rand (retry jitter)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Rust port of [open-agent-sdk](https://github.com/slb350/open-agent-sdk) Python library
- API design inspired by claude-agent-sdk
- Built for local/open-source LLM enthusiasts

## Repository Hosting

[GitHub](https://github.com/slb350/open-agent-sdk-rust) is the canonical repository and CI/release host. Any family Gitea copy is a passive Git mirror and does not run a separate required Actions pipeline.

---

**Status**: v0.11.2 - Model requests deliberately reject redirects so custom headers and other credentials are sent only to the configured origin, plus validated caller-supplied model-request headers, incremental streaming, OpenAI and Anthropic wire protocols, finish reasons, reasoning-channel separation, structured retry classification, context controls, hooks, tools and multimodal image support

Star this repo if you're building AI agents with local models in Rust!
