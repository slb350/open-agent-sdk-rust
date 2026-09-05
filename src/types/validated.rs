/// A model name validated at construction time.
///
/// # Validation Rules
///
/// - Must not be empty
/// - Must not be only whitespace
///
/// # Example
///
/// ```
/// use open_agent::ModelName;
///
/// // Valid model name
/// let model = ModelName::new("qwen2.5-32b-instruct").unwrap();
/// assert_eq!(model.as_str(), "qwen2.5-32b-instruct");
///
/// // Invalid: empty string
/// assert!(ModelName::new("").is_err());
///
/// // Invalid: whitespace only
/// assert!(ModelName::new("   ").is_err());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelName(String);

impl ModelName {
    /// Creates a new `ModelName` after validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the model name is empty or contains only whitespace.
    pub fn new(name: impl Into<String>) -> crate::Result<Self> {
        let name = name.into();
        let trimmed = name.trim();

        if trimmed.is_empty() {
            return Err(Error::invalid_input(
                "Model name cannot be empty or whitespace",
            ));
        }

        Ok(ModelName(name))
    }

    /// Returns the model name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the `ModelName` and returns the inner `String`.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl std::fmt::Display for ModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A base URL validated at construction time.
///
/// # Validation Rules
///
/// - Must not be empty
/// - After trimming, must start with `http://` or `https://`
///
/// Validation preserves the original string, including whitespace. The options
/// builder checks the scheme on the untrimmed input for compatibility.
///
/// # Example
///
/// ```
/// use open_agent::BaseUrl;
///
/// // Valid base URLs
/// let url = BaseUrl::new("http://localhost:1234/v1").unwrap();
/// assert_eq!(url.as_str(), "http://localhost:1234/v1");
///
/// let url = BaseUrl::new("https://api.openai.com/v1").unwrap();
/// assert_eq!(url.as_str(), "https://api.openai.com/v1");
///
/// // Invalid: no http/https prefix
/// assert!(BaseUrl::new("localhost:1234").is_err());
///
/// // Invalid: empty string
/// assert!(BaseUrl::new("").is_err());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BaseUrl(String);

impl BaseUrl {
    /// Creates a new `BaseUrl` after validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the URL is empty or doesn't start with http:// or https://.
    pub fn new(url: impl Into<String>) -> crate::Result<Self> {
        let url = url.into();
        let trimmed = url.trim();

        if trimmed.is_empty() {
            return Err(Error::invalid_input("base_url cannot be empty"));
        }

        if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
            return Err(Error::invalid_input(
                "base_url must start with http:// or https://",
            ));
        }

        Ok(BaseUrl(url))
    }

    /// Returns the base URL as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the `BaseUrl` and returns the inner `String`.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl std::fmt::Display for BaseUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A temperature validated at construction time.
///
/// # Validation Rules
///
/// - Must be between 0.0 and 2.0 (inclusive)
///
/// # Example
///
/// ```
/// use open_agent::Temperature;
///
/// // Valid temperatures
/// let temp = Temperature::new(0.7).unwrap();
/// assert_eq!(temp.value(), 0.7);
///
/// let temp = Temperature::new(0.0).unwrap();
/// assert_eq!(temp.value(), 0.0);
///
/// let temp = Temperature::new(2.0).unwrap();
/// assert_eq!(temp.value(), 2.0);
///
/// // Invalid: below range
/// assert!(Temperature::new(-0.1).is_err());
///
/// // Invalid: above range
/// assert!(Temperature::new(2.1).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Temperature(f32);

impl Temperature {
    /// Creates a new `Temperature` after validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the temperature is not between 0.0 and 2.0 (inclusive).
    pub fn new(temp: f32) -> crate::Result<Self> {
        if !(0.0..=2.0).contains(&temp) {
            return Err(Error::invalid_input(
                "temperature must be between 0.0 and 2.0",
            ));
        }

        Ok(Temperature(temp))
    }

    /// Returns the temperature value.
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl std::fmt::Display for Temperature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
