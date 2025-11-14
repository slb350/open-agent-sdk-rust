//! Example demonstrating multimodal image support with the Vision API
//!
//! This example shows how to use the `send_message()` API with image helper methods
//! to send messages containing images to a vision-capable model.
//!
//! Run this example against a local vision-capable model:
//! ```bash
//! cargo run --example vision_example
//! ```

use open_agent::{AgentOptions, Client, ContentBlock, ImageDetail, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure client for a vision-capable model
    // For LM Studio: Use models like "llava" or "bakllava"
    // For Ollama: ollama run llava or ollama run bakllava
    let options = AgentOptions::builder()
        .model("llava") // Change to your vision model
        .base_url("http://localhost:11434/v1") // Ollama endpoint
        .build()?;

    let mut client = Client::new(options)?;

    println!("=== Example 1: Simple Image Query ===\n");

    // Use the user_with_image helper to create a message with an image
    let msg = Message::user_with_image(
        "What's in this image? Describe it in detail.",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    )?;

    // Send the message using the new send_message() API
    client.send_message(msg).await?;

    // Receive and print the response
    println!("Response:");
    while let Some(block) = client.receive().await? {
        match block {
            ContentBlock::Text(text) => {
                print!("{}", text.text);
            }
            _ => {}
        }
    }
    println!("\n");

    println!("=== Example 2: Image with High Detail ===\n");

    // For detailed analysis, use user_with_image_detail to specify detail level
    let msg = Message::user_with_image_detail(
        "Analyze this diagram in detail, focusing on the architecture.",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Three_layer_neural_network-en.svg/1200px-Three_layer_neural_network-en.svg.png",
        ImageDetail::High,
    )?;

    client.send_message(msg).await?;

    println!("Response:");
    while let Some(block) = client.receive().await? {
        match block {
            ContentBlock::Text(text) => {
                print!("{}", text.text);
            }
            _ => {}
        }
    }
    println!("\n");

    println!("=== Example 3: Base64 Image ===\n");

    // For in-memory images, use user_with_base64_image
    // This is a 1x1 red pixel PNG for demonstration
    let base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

    let msg = Message::user_with_base64_image(
        "What color is this pixel?",
        base64_data,
        "image/png",
    )?;

    client.send_message(msg).await?;

    println!("Response:");
    while let Some(block) = client.receive().await? {
        match block {
            ContentBlock::Text(text) => {
                print!("{}", text.text);
            }
            _ => {}
        }
    }
    println!("\n");

    println!("=== Example 4: Multiple Images (Custom Blocks) ===\n");

    // For complex scenarios, manually construct a Message with multiple images
    use open_agent::{ImageBlock, MessageRole, TextBlock};

    let image1 = ImageBlock::from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg")?;
    let image2 = ImageBlock::from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/400px-Cat_November_2010-1a.jpg")?;

    let msg = Message::new(
        MessageRole::User,
        vec![
            ContentBlock::Text(TextBlock::new("Compare these two images. What are the similarities and differences?")),
            ContentBlock::Image(image1),
            ContentBlock::Image(image2),
        ],
    );

    client.send_message(msg).await?;

    println!("Response:");
    while let Some(block) = client.receive().await? {
        match block {
            ContentBlock::Text(text) => {
                print!("{}", text.text);
            }
            _ => {}
        }
    }
    println!("\n");

    Ok(())
}
