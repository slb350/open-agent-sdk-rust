//! Vision API demonstration with multimodal image support
//!
//! This example demonstrates how to construct messages with images for
//! vision-capable models. It shows the convenience methods for working
//! with images alongside text prompts.
//!
//! # Features Demonstrated
//!
//! - Creating messages with images via URL
//! - Using base64-encoded images
//! - Controlling image detail levels for token cost management
//! - Manual content block construction for complex multi-image messages
//!
//! # Usage
//!
//! This example demonstrates the API without making actual network calls.
//! To use with a real vision model:
//!
//! 1. Start a local vision-capable model server (e.g., LM Studio with llava)
//! 2. Replace placeholder URLs with actual image URLs
//! 3. Uncomment the client creation and API call sections
//!
//! Run: `cargo run --example vision_api_demo`

use open_agent::{ContentBlock, ImageBlock, ImageDetail, Message, MessageRole, TextBlock};

fn main() {
    println!("🖼️  Vision API Demo\n");
    println!("This demo shows how to construct messages with images.\n");

    // ========================================================================
    // Example 1: Simple image with text (most common pattern)
    // ========================================================================
    println!("📸 Example 1: Simple image + text message");
    println!("──────────────────────────────────────────");

    let msg1 = Message::user_with_image(
        "What's in this image? Describe it in detail.",
        "https://example.com/photo.jpg",
    )
    .expect("Failed to create message with image");

    println!("Created message with {} content blocks", msg1.content.len());
    println!("  - Block 0: Text");
    println!("  - Block 1: Image (detail: Auto)\n");

    // ========================================================================
    // Example 2: Image with detail level control
    // ========================================================================
    println!("🔍 Example 2: Image with HIGH detail level");
    println!("────────────────────────────────────────────");
    println!("Use ImageDetail::High for accurate analysis (higher token cost)\n");

    let msg2 = Message::user_with_image_detail(
        "Analyze the fine details in this diagram.",
        "https://example.com/diagram.png",
        ImageDetail::High,
    )
    .expect("Failed to create message with image detail");

    println!("Created message with {} content blocks", msg2.content.len());
    println!("  - Block 0: Text");
    println!("  - Block 1: Image (detail: High)\n");

    // ========================================================================
    // Example 3: Base64-encoded image
    // ========================================================================
    println!("📊 Example 3: Base64-encoded image");
    println!("────────────────────────────────────");

    // Simple 1x1 red pixel as a minimal example
    let base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

    let msg3 =
        Message::user_with_base64_image("What color is this pixel?", base64_data, "image/png")
            .expect("Failed to create message with base64 image");

    println!("Created message with {} content blocks", msg3.content.len());
    if let ContentBlock::Image(img) = &msg3.content[1] {
        println!("  - Block 0: Text");
        println!("  - Block 1: Image (data URI)");
        println!("    URL starts with: {}", &img.url()[..30]);
    }
    println!();

    // ========================================================================
    // Example 4: Token cost comparison
    // ========================================================================
    println!("💰 Example 4: Detail level token costs");
    println!("────────────────────────────────────────");

    let _low = ImageBlock::from_url("https://example.com/img.jpg")
        .expect("Valid URL")
        .with_detail(ImageDetail::Low);

    let _high = ImageBlock::from_url("https://example.com/img.jpg")
        .expect("Valid URL")
        .with_detail(ImageDetail::High);

    let _auto = ImageBlock::from_url("https://example.com/img.jpg")
        .expect("Valid URL")
        .with_detail(ImageDetail::Auto);

    println!("ImageDetail::Low  - Fixed ~85 tokens (512x512 max)");
    println!("ImageDetail::High - Variable tokens (based on dimensions)");
    println!("ImageDetail::Auto - Model decides (default)\n");

    // ========================================================================
    // Example 5: Manual content block construction
    // ========================================================================
    println!("🔧 Example 5: Complex multi-image message");
    println!("──────────────────────────────────────────");
    println!("For advanced use cases, manually construct content blocks.\n");

    let msg5 = Message::new(
        MessageRole::User,
        vec![
            ContentBlock::Text(TextBlock::new("Compare these three aspects:")),
            ContentBlock::Text(TextBlock::new("1. Composition:")),
            ContentBlock::Image(
                ImageBlock::from_url("https://example.com/photo1.jpg")
                    .expect("Valid URL")
                    .with_detail(ImageDetail::Low),
            ),
            ContentBlock::Text(TextBlock::new("2. Color palette:")),
            ContentBlock::Image(
                ImageBlock::from_url("https://example.com/photo2.jpg")
                    .expect("Valid URL")
                    .with_detail(ImageDetail::Low),
            ),
            ContentBlock::Text(TextBlock::new("3. Lighting:")),
            ContentBlock::Image(
                ImageBlock::from_url("https://example.com/photo3.jpg")
                    .expect("Valid URL")
                    .with_detail(ImageDetail::Low),
            ),
        ],
    );

    println!(
        "Created complex message with {} content blocks:",
        msg5.content.len()
    );
    for (i, block) in msg5.content.iter().enumerate() {
        match block {
            ContentBlock::Text(t) => println!(
                "  - Block {}: Text ({})",
                i,
                &t.text[..30.min(t.text.len())]
            ),
            ContentBlock::Image(_) => println!("  - Block {}: Image", i),
            ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) => {}
        }
    }
    println!();

    // ========================================================================
    // Usage with Client (commented out - requires live server)
    // ========================================================================
    println!("📡 Example 6: Actual usage pattern (pseudocode)");
    println!("────────────────────────────────────────────────────");
    println!(
        r#"
// Configure client for vision model
let options = AgentOptions::builder()
    .system_prompt("You are a vision assistant.")
    .model("llava-v1.6-34b")
    .base_url("http://localhost:1234/v1")
    .build()?;

let mut client = Client::new(options)?;

// Create message with image
let msg = Message::user_with_image(
    "Describe this image",
    "https://example.com/photo.jpg"
);

// Add to history and send
client.history_mut().push(msg);
// Note: Current API requires using send() with text
// Future enhancement: client.send_message(msg)?
"#
    );

    println!("\n✅ Vision API demo complete!");
    println!("\nKey takeaways:");
    println!("• Message::user_with_image() - simplest image + text");
    println!("• Message::user_with_image_detail() - control token costs");
    println!("• Message::user_with_base64_image() - in-memory images");
    println!("• ImageDetail::Low (~85 tokens) vs High (variable) vs Auto");
    println!("• Manually construct ContentBlock vectors for complex cases");
}
