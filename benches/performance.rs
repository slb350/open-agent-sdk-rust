use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use open_agent::{
    estimate_tokens, is_approaching_limit, truncate_messages, ContentBlock, Message, MessageRole,
    TextBlock, ToolResultBlock, ToolUseBlock,
};
use serde_json::json;

// Helper function to create test messages with varying sizes
fn create_messages(count: usize, text_size: usize) -> Vec<Message> {
    let text = "a".repeat(text_size);
    (0..count)
        .map(|i| {
            if i == 0 {
                Message::system(&text)
            } else if i % 2 == 0 {
                Message::user(&text)
            } else {
                Message::assistant(vec![ContentBlock::Text(TextBlock { text: text.clone() })])
            }
        })
        .collect()
}

// Helper to create messages with tool calls
fn create_messages_with_tools(count: usize) -> Vec<Message> {
    let mut messages = vec![Message::system("You are a helpful assistant")];

    for i in 0..count {
        if i % 3 == 0 {
            messages.push(Message::user("Calculate 2 + 2"));
        } else if i % 3 == 1 {
            // Tool use
            let tool_use = ToolUseBlock {
                id: format!("tool_{}", i),
                name: "calculator".to_string(),
                input: json!({"operation": "add", "a": 2, "b": 2}),
            };
            messages.push(Message::new(
                MessageRole::Assistant,
                vec![ContentBlock::ToolUse(tool_use)],
            ));
        } else {
            // Tool result
            let tool_result = ToolResultBlock {
                tool_use_id: format!("tool_{}", i - 1),
                content: json!({"result": 4}),
            };
            messages.push(Message::new(
                MessageRole::User,
                vec![ContentBlock::ToolResult(tool_result)],
            ));
        }
    }

    messages
}

// Benchmark: estimate_tokens with varying message counts
fn bench_estimate_tokens_by_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate_tokens_by_count");

    for count in [0, 1, 5, 10, 20, 50, 100].iter() {
        let messages = create_messages(*count, 100);
        group.bench_with_input(BenchmarkId::from_parameter(count), &messages, |b, msgs| {
            b.iter(|| estimate_tokens(black_box(msgs)));
        });
    }

    group.finish();
}

// Benchmark: estimate_tokens with varying message sizes
fn bench_estimate_tokens_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate_tokens_by_size");

    for size in [10, 100, 1000, 10000].iter() {
        let messages = create_messages(10, *size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &messages, |b, msgs| {
            b.iter(|| estimate_tokens(black_box(msgs)));
        });
    }

    group.finish();
}

// Benchmark: estimate_tokens with tool calls
fn bench_estimate_tokens_with_tools(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate_tokens_with_tools");

    for count in [3, 9, 30, 90].iter() {
        let messages = create_messages_with_tools(*count);
        group.bench_with_input(BenchmarkId::from_parameter(count), &messages, |b, msgs| {
            b.iter(|| estimate_tokens(black_box(msgs)));
        });
    }

    group.finish();
}

// Benchmark: truncate_messages with varying inputs
fn bench_truncate_messages(c: &mut Criterion) {
    let mut group = c.benchmark_group("truncate_messages");

    let test_cases = vec![
        ("small_keep_5", create_messages(20, 100), 5, true),
        ("medium_keep_10", create_messages(50, 100), 10, true),
        ("large_keep_20", create_messages(100, 100), 20, true),
        ("no_preserve_system", create_messages(50, 100), 10, false),
    ];

    for (name, messages, keep, preserve) in test_cases {
        group.bench_with_input(
            BenchmarkId::new(name, format!("{}/{}", messages.len(), keep)),
            &(messages, keep, preserve),
            |b, (msgs, k, p)| {
                b.iter(|| truncate_messages(black_box(msgs), black_box(*k), black_box(*p)));
            },
        );
    }

    group.finish();
}

// Benchmark: truncate_messages with tool calls
fn bench_truncate_messages_with_tools(c: &mut Criterion) {
    let mut group = c.benchmark_group("truncate_messages_with_tools");

    for count in [30, 90].iter() {
        let messages = create_messages_with_tools(*count);
        group.bench_with_input(BenchmarkId::from_parameter(count), &messages, |b, msgs| {
            b.iter(|| truncate_messages(black_box(msgs), black_box(10), black_box(true)));
        });
    }

    group.finish();
}

// Benchmark: is_approaching_limit
fn bench_is_approaching_limit(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_approaching_limit");

    for count in [10, 50, 100].iter() {
        let messages = create_messages(*count, 500);
        group.bench_with_input(BenchmarkId::from_parameter(count), &messages, |b, msgs| {
            b.iter(|| is_approaching_limit(black_box(msgs), black_box(32000), black_box(0.9)));
        });
    }

    group.finish();
}

// Benchmark: realistic workflow - check and truncate if needed
fn bench_realistic_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_workflow");

    let messages = create_messages(50, 200);

    group.bench_function("check_and_truncate", |b| {
        b.iter(|| {
            let msgs = black_box(&messages);
            let tokens = estimate_tokens(msgs);
            if tokens > black_box(10000) {
                truncate_messages(msgs, black_box(10), black_box(true))
            } else {
                msgs.to_vec()
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_estimate_tokens_by_count,
    bench_estimate_tokens_by_size,
    bench_estimate_tokens_with_tools,
    bench_truncate_messages,
    bench_truncate_messages_with_tools,
    bench_is_approaching_limit,
    bench_realistic_workflow,
);
criterion_main!(benches);
