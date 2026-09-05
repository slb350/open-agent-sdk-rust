//! Image logging must truncate payloads without damaging UTF-8 or warning on image-only input.

mod common;

use log::{Level, LevelFilter, Log, Metadata, Record};
use open_agent::{Client, ContentBlock, ImageBlock, ImageDetail, Message, MessageRole};
use std::sync::Mutex;

struct CapturedLogs(Mutex<Vec<(Level, String)>>);

impl Log for CapturedLogs {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &Record<'_>) {
        if record.target().starts_with("open_agent") {
            self.0
                .lock()
                .unwrap()
                .push((record.level(), record.args().to_string()));
        }
    }

    fn flush(&self) {}
}

static LOGS: CapturedLogs = CapturedLogs(Mutex::new(Vec::new()));

#[tokio::test]
async fn image_logs_truncate_utf8_urls_and_include_detail_without_image_only_warnings() {
    log::set_logger(&LOGS).unwrap();
    log::set_max_level(LevelFilter::Debug);
    let server = common::sse_server(common::DONE).await;
    let mut client = Client::new(common::options_for(&server)).unwrap();
    let ascii_url = format!("https://example.com/{}", "a".repeat(180));
    let unicode_url = format!("https://example.com/{}🌸{}", "a".repeat(79), "b".repeat(40));
    let base64 = "A".repeat(200);
    let short_url = "https://example.com/short.jpg";
    let images = vec![
        ImageBlock::from_url(&ascii_url)
            .unwrap()
            .with_detail(ImageDetail::Low),
        ImageBlock::from_url(&unicode_url)
            .unwrap()
            .with_detail(ImageDetail::High),
        ImageBlock::from_base64(&base64, "image/png").unwrap(),
        ImageBlock::from_url(short_url).unwrap(),
    ];
    client
        .send_message(Message::new(
            MessageRole::User,
            images.into_iter().map(ContentBlock::Image).collect(),
        ))
        .await
        .unwrap();

    let logs = LOGS.0.lock().unwrap();
    for detail in ["low", "high", "auto"] {
        assert!(
            logs.iter()
                .any(|(_, line)| line.contains(&format!("detail: {detail}")))
        );
    }
    assert!(logs.iter().any(|(_, line)| line.contains(short_url)));
    let unicode_prefix = format!("https://example.com/{}...", "a".repeat(79));
    assert!(logs.iter().any(|(_, line)| line.contains(&unicode_prefix)));
    for private_value in [&ascii_url, &unicode_url, &base64] {
        assert!(logs.iter().all(|(_, line)| !line.contains(private_value)));
    }
    assert!(!logs.iter().any(|(level, _)| *level == Level::Warn));
}
