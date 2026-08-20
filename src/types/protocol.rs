//! Which HTTP API an endpoint speaks.
//!
//! The SDK started as an OpenAI-compatible client and the shape of that API was baked into
//! the request path, the auth header, the request body and the streaming vocabulary. Several
//! providers that are otherwise ordinary OpenAI-compatible vendors publish their
//! subscription tiers only behind an Anthropic-shaped `/messages` endpoint, so "which
//! protocol" became a per-endpoint fact rather than a property of the SDK.
//!
//! This is deliberately not a provider enum. It names the wire format, so a new vendor
//! speaking an existing protocol needs no change here.

/// The wire protocol an endpoint exposes.
///
/// Selects the request path, the authentication header, the request body shape and the
/// streaming event vocabulary. Defaults to [`ApiProtocol::OpenAiChat`], which is what every
/// endpoint the SDK supported before 0.9.0 speaks.
///
/// # Examples
///
/// ```rust
/// use open_agent::ApiProtocol;
///
/// assert_eq!(ApiProtocol::default(), ApiProtocol::OpenAiChat);
/// assert_eq!(ApiProtocol::Anthropic.path(), "/messages");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ApiProtocol {
    /// OpenAI chat completions: `POST {base_url}/chat/completions`, bearer auth.
    #[default]
    OpenAiChat,

    /// Anthropic messages: `POST {base_url}/messages`, `x-api-key` auth.
    Anthropic,
}

impl ApiProtocol {
    /// The path appended to `base_url` to reach the completion endpoint.
    ///
    /// Includes the leading slash, because `base_url` is documented as carrying no trailing
    /// one.
    pub fn path(&self) -> &'static str {
        match self {
            Self::OpenAiChat => "/chat/completions",
            Self::Anthropic => "/messages",
        }
    }

    /// The lowercase name used in configuration files and diagnostics.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OpenAiChat => "openai",
            Self::Anthropic => "anthropic",
        }
    }

    /// Parses a protocol name, ASCII-case-insensitively.
    ///
    /// Returns `None` for anything unrecognised, so a caller reading a config file can name
    /// the offending value in its own error rather than silently falling back to a default
    /// the user did not ask for.
    ///
    /// `openai` is accepted alongside `openai-chat` and `openai_chat`: configuration files
    /// in the wild write it every way, and rejecting the short form would be a trap with no
    /// upside.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use open_agent::ApiProtocol;
    ///
    /// assert_eq!(ApiProtocol::from_wire("Anthropic"), Some(ApiProtocol::Anthropic));
    /// assert_eq!(ApiProtocol::from_wire("openai_chat"), Some(ApiProtocol::OpenAiChat));
    /// assert_eq!(ApiProtocol::from_wire("cohere"), None);
    /// ```
    pub fn from_wire(raw: &str) -> Option<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "openai" | "openai-chat" | "openai_chat" => Some(Self::OpenAiChat),
            "anthropic" => Some(Self::Anthropic),
            _ => None,
        }
    }
}

impl std::fmt::Display for ApiProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_openai_chat() {
        assert_eq!(ApiProtocol::default(), ApiProtocol::OpenAiChat);
    }

    #[test]
    fn each_protocol_has_its_own_path() {
        assert_eq!(ApiProtocol::OpenAiChat.path(), "/chat/completions");
        assert_eq!(ApiProtocol::Anthropic.path(), "/messages");
    }

    #[test]
    fn each_protocol_has_its_own_name() {
        assert_eq!(ApiProtocol::OpenAiChat.as_str(), "openai");
        assert_eq!(ApiProtocol::Anthropic.as_str(), "anthropic");
    }

    #[test]
    fn from_wire_is_case_insensitive() {
        assert_eq!(
            ApiProtocol::from_wire("ANTHROPIC"),
            Some(ApiProtocol::Anthropic)
        );
        assert_eq!(
            ApiProtocol::from_wire("OpenAI"),
            Some(ApiProtocol::OpenAiChat)
        );
    }

    #[test]
    fn from_wire_accepts_every_spelling_of_openai() {
        for spelling in ["openai", "openai-chat", "openai_chat"] {
            assert_eq!(
                ApiProtocol::from_wire(spelling),
                Some(ApiProtocol::OpenAiChat),
                "{spelling} should parse"
            );
        }
    }

    #[test]
    fn from_wire_rejects_an_unknown_protocol() {
        assert_eq!(ApiProtocol::from_wire("cohere"), None);
        assert_eq!(ApiProtocol::from_wire(""), None);
    }

    #[test]
    fn display_matches_as_str() {
        assert_eq!(ApiProtocol::Anthropic.to_string(), "anthropic");
        assert_eq!(ApiProtocol::OpenAiChat.to_string(), "openai");
    }

    #[test]
    fn from_wire_round_trips_as_str() {
        for protocol in [ApiProtocol::OpenAiChat, ApiProtocol::Anthropic] {
            assert_eq!(ApiProtocol::from_wire(protocol.as_str()), Some(protocol));
        }
    }
}
