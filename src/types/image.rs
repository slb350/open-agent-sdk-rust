/// Image detail level for vision API calls.
///
/// Controls the resolution and token cost of image processing.
///
/// # Token Costs Vary by Model ⚠️
///
/// **OpenAI Vision API** (reference values):
/// - `Low`: ~85 tokens (512x512 max resolution)
/// - `High`: Variable tokens based on image dimensions
/// - `Auto`: Model decides (balanced default)
///
/// **Local models** (llama.cpp, Ollama, vLLM):
/// - May have **completely different** token calculations
/// - Some models don't charge tokens for images at all
/// - The `ImageDetail` setting may be ignored entirely
///
/// **Anthropic messages protocol**: the API has no equivalent field, so the setting is
/// dropped in translation rather than mapped onto something approximate.
///
/// **Recommendation:** Always benchmark your specific model to understand
/// actual token consumption. Do not rely on OpenAI's values for capacity planning
/// with local models.
///
/// # Examples
///
/// ```
/// use open_agent::ImageDetail;
///
/// let detail = ImageDetail::High;
/// assert_eq!(detail.to_string(), "high");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ImageDetail {
    /// Low resolution (512x512), fixed 85 tokens
    Low,
    /// High resolution, variable tokens based on dimensions
    High,
    /// Automatic selection (default)
    #[default]
    Auto,
}

impl std::fmt::Display for ImageDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageDetail::Low => write!(f, "low"),
            ImageDetail::High => write!(f, "high"),
            ImageDetail::Auto => write!(f, "auto"),
        }
    }
}

/// Image content block for vision-capable models.
///
/// Supports both URL-based images and base64-encoded images.
///
/// # Examples
///
/// ```
/// use open_agent::{ImageBlock, ImageDetail};
///
/// // From URL
/// let image = ImageBlock::from_url("https://example.com/image.jpg")?;
///
/// // From base64 (use properly formatted base64)
/// let image = ImageBlock::from_base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", "image/png")?;
///
/// // With detail level
/// let image = ImageBlock::from_url("https://example.com/image.jpg")?
///     .with_detail(ImageDetail::High);
/// # Ok::<(), open_agent::Error>(())
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageBlock {
    url: String,
    #[serde(default)]
    detail: ImageDetail,
}

impl ImageBlock {
    /// Creates a new image block from a URL.
    ///
    /// # Arguments
    ///
    /// * `url` - The image URL (must be HTTP, HTTPS, or data URI)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if:
    /// - URL is empty
    /// - URL contains control characters (newline, tab, null, etc.)
    /// - URL scheme is not `http://`, `https://`, or `data:`
    /// - Data URI is malformed (missing MIME type or base64 encoding)
    /// - Data URI base64 portion has invalid characters, length, or padding
    ///
    /// # Warnings
    ///
    /// - Logs a warning to stderr if URL exceeds 2000 characters
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::ImageBlock;
    ///
    /// let image = ImageBlock::from_url("https://example.com/cat.jpg")?;
    /// assert_eq!(image.url(), "https://example.com/cat.jpg");
    /// # Ok::<(), open_agent::Error>(())
    /// ```
    pub fn from_url(url: impl Into<String>) -> crate::Result<Self> {
        let url = url.into();

        // Validate URL is not empty
        if url.is_empty() {
            return Err(crate::Error::invalid_input("Image URL cannot be empty"));
        }

        // Check for control characters in URL
        if url.contains(char::is_control) {
            return Err(crate::Error::invalid_input(
                "Image URL contains invalid control characters",
            ));
        }

        // Warn about very long URLs (>2000 chars)
        if url.len() > 2000 {
            eprintln!(
                "WARNING: Very long image URL ({} chars). \
                 Some APIs may have URL length limits.",
                url.len()
            );
        }

        // Validate URL scheme
        if url.starts_with("http://") || url.starts_with("https://") {
            // Valid HTTP/HTTPS URL
            Ok(Self {
                url,
                detail: ImageDetail::default(),
            })
        } else if let Some(mime_part) = url.strip_prefix("data:") {
            // Validate data URI format: data:MIME;base64,DATA
            if !url.contains(";base64,") {
                return Err(crate::Error::invalid_input(
                    "Data URI must be in format: data:image/TYPE;base64,DATA",
                ));
            }

            // Extract MIME type from data:MIME;base64,DATA
            let mime_type = if let Some(semicolon_pos) = mime_part.find(';') {
                &mime_part[..semicolon_pos]
            } else {
                return Err(crate::Error::invalid_input(
                    "Malformed data URI: missing MIME type",
                ));
            };

            if mime_type.is_empty() || !mime_type.starts_with("image/") {
                return Err(crate::Error::invalid_input(
                    "Data URI MIME type must start with 'image/'",
                ));
            }

            // Extract and validate base64 data portion
            if let Some(base64_start_pos) = url.find(";base64,") {
                let base64_data = &url[base64_start_pos + 8..]; // Skip ";base64,"

                // Validate base64 data using same rules as from_base64()
                // Check data is not empty
                if base64_data.is_empty() {
                    return Err(crate::Error::invalid_input(
                        "Data URI base64 data cannot be empty",
                    ));
                }

                // Check character set
                if !base64_data
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=')
                {
                    return Err(crate::Error::invalid_input(
                        "Data URI base64 data contains invalid characters. Valid characters: A-Z, a-z, 0-9, +, /, =",
                    ));
                }

                // Check length (must be multiple of 4)
                if base64_data.len() % 4 != 0 {
                    return Err(crate::Error::invalid_input(
                        "Data URI base64 data has invalid length (must be multiple of 4)",
                    ));
                }

                // Validate padding
                let equals_count = base64_data.chars().filter(|c| *c == '=').count();
                if equals_count > 2 {
                    return Err(crate::Error::invalid_input(
                        "Data URI base64 data has invalid padding (max 2 '=' characters allowed)",
                    ));
                }
                // Padding must be at the end
                if equals_count > 0 {
                    let trimmed = base64_data.trim_end_matches('=');
                    if trimmed.len() + equals_count != base64_data.len() {
                        return Err(crate::Error::invalid_input(
                            "Data URI base64 padding characters must be at the end",
                        ));
                    }
                }
            }

            Ok(Self {
                url,
                detail: ImageDetail::default(),
            })
        } else {
            Err(crate::Error::invalid_input(
                "Image URL must start with http://, https://, or data:",
            ))
        }
    }

    /// Creates a new image block from base64-encoded data.
    ///
    /// # Arguments
    ///
    /// * `base64_data` - The base64-encoded image data
    /// * `mime_type` - The MIME type (e.g., "image/jpeg", "image/png")
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if:
    /// - Base64 data is empty
    /// - Base64 contains invalid characters (only A-Z, a-z, 0-9, +, /, = allowed)
    /// - Base64 length is not a multiple of 4
    /// - Base64 has invalid padding (more than 2 '=' characters or not at end)
    /// - MIME type is empty
    /// - MIME type does not start with "image/"
    /// - MIME type contains injection characters (;, \\n, \\r, ,)
    ///
    /// # Warnings
    ///
    /// - Logs a warning to stderr if base64 data exceeds 10MB (~7.5MB decoded)
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::ImageBlock;
    ///
    /// let base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    /// let image = ImageBlock::from_base64(base64, "image/png")?;
    /// assert!(image.url().starts_with("data:image/png;base64,"));
    /// # Ok::<(), open_agent::Error>(())
    /// ```
    pub fn from_base64(
        base64_data: impl AsRef<str>,
        mime_type: impl AsRef<str>,
    ) -> crate::Result<Self> {
        let data = base64_data.as_ref();
        let mime = mime_type.as_ref();

        // Validate base64 data is not empty
        if data.is_empty() {
            return Err(crate::Error::invalid_input(
                "Base64 image data cannot be empty",
            ));
        }

        // Validate base64 character set (alphanumeric + +/=)
        // This catches common errors like spaces, special characters, etc.
        if !data
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=')
        {
            return Err(crate::Error::invalid_input(
                "Base64 data contains invalid characters. Valid characters: A-Z, a-z, 0-9, +, /, =",
            ));
        }

        // Validate base64 padding (length must be multiple of 4)
        if data.len() % 4 != 0 {
            return Err(crate::Error::invalid_input(
                "Base64 data has invalid length (must be multiple of 4)",
            ));
        }

        // Validate padding characters only appear at the end (max 2)
        let equals_count = data.chars().filter(|c| *c == '=').count();
        if equals_count > 2 {
            return Err(crate::Error::invalid_input(
                "Base64 data has invalid padding (max 2 '=' characters allowed)",
            ));
        }
        if equals_count > 0 {
            // Padding must be at the end
            let trimmed = data.trim_end_matches('=');
            if trimmed.len() + equals_count != data.len() {
                return Err(crate::Error::invalid_input(
                    "Base64 padding characters must be at the end",
                ));
            }
        }

        // Validate MIME type is not empty
        if mime.is_empty() {
            return Err(crate::Error::invalid_input("MIME type cannot be empty"));
        }

        // Validate MIME type starts with "image/"
        if !mime.starts_with("image/") {
            return Err(crate::Error::invalid_input(
                "MIME type must start with 'image/' (e.g., 'image/png', 'image/jpeg')",
            ));
        }

        // Check for MIME type injection characters
        if mime.contains([';', ',', '\n', '\r']) {
            return Err(crate::Error::invalid_input(
                "MIME type contains invalid characters (;, \\n, \\r not allowed)",
            ));
        }

        // Warn about extremely large base64 data (>10MB)
        if data.len() > 10_000_000 {
            eprintln!(
                "WARNING: Very large base64 image data ({} chars, ~{:.1}MB). \
                 This may exceed API limits or cause performance issues.",
                data.len(),
                (data.len() as f64 * 0.75) / 1_000_000.0
            );
        }

        let url = format!("data:{};base64,{}", mime, data);
        Ok(Self {
            url,
            detail: ImageDetail::default(),
        })
    }

    /// Creates a new image block from a local file path.
    ///
    /// This is a convenience method that reads the file from disk, encodes it as
    /// base64, and creates an ImageBlock with a data URI. The MIME type is inferred
    /// from the file extension.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the image file on the local filesystem
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidInput` if:
    /// - File cannot be read
    /// - File extension is missing or unsupported
    /// - File is too large (>10MB warning)
    ///
    /// # Supported Formats
    ///
    /// - `.jpg`, `.jpeg` → `image/jpeg`
    /// - `.png` → `image/png`
    /// - `.gif` → `image/gif`
    /// - `.webp` → `image/webp`
    /// - `.bmp` → `image/bmp`
    /// - `.svg` → `image/svg+xml`
    ///
    /// # Example
    ///
    /// ```no_run
    /// use open_agent::ImageBlock;
    ///
    /// let image = ImageBlock::from_file_path("/path/to/photo.jpg")?;
    /// # Ok::<(), open_agent::Error>(())
    /// ```
    ///
    /// # Security Note
    ///
    /// This method reads files from the local filesystem. Ensure the path comes from
    /// a trusted source to prevent unauthorized file access.
    pub fn from_file_path(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        use base64::{Engine as _, engine::general_purpose};

        let path = path.as_ref();

        // Read file bytes
        let bytes = std::fs::read(path).map_err(|e| {
            crate::Error::invalid_input(format!(
                "Failed to read image file '{}': {}",
                path.display(),
                e
            ))
        })?;

        // Determine MIME type from file extension
        let mime_type = match path.extension().and_then(|e| e.to_str()) {
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("png") => "image/png",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            Some("bmp") => "image/bmp",
            Some("svg") => "image/svg+xml",
            Some(ext) => {
                return Err(crate::Error::invalid_input(format!(
                    "Unsupported image file extension: .{}. Supported: jpg, jpeg, png, gif, webp, bmp, svg",
                    ext
                )));
            }
            None => {
                return Err(crate::Error::invalid_input(
                    "Image file path must have a file extension (e.g., .jpg, .png)",
                ));
            }
        };

        // Encode to base64
        let base64_data = general_purpose::STANDARD.encode(&bytes);

        // Use existing from_base64 method for validation
        Self::from_base64(&base64_data, mime_type)
    }

    /// Sets the image detail level.
    ///
    /// # Example
    ///
    /// ```
    /// use open_agent::{ImageBlock, ImageDetail};
    ///
    /// let image = ImageBlock::from_url("https://example.com/image.jpg")?
    ///     .with_detail(ImageDetail::High);
    /// # Ok::<(), open_agent::Error>(())
    /// ```
    pub fn with_detail(mut self, detail: ImageDetail) -> Self {
        self.detail = detail;
        self
    }

    /// Returns the image URL (or data URI for base64 images).
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Returns the image detail level.
    pub fn detail(&self) -> ImageDetail {
        self.detail
    }
}
