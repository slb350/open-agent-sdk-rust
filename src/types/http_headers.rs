use super::AgentOptionsBuilder;
use crate::{Error, Result};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use std::collections::BTreeMap;

impl AgentOptionsBuilder {
    /// Adds a caller-supplied HTTP header to every model request.
    ///
    /// A later call with the same name replaces the earlier value, even when the casing differs.
    /// Caller headers take precedence over the defaults selected by [`Self::protocol`].
    /// Names and values are validated by [`Self::build`], which identifies the offending name
    /// in its configuration error.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        let name = name.into();
        self.headers
            .retain(|existing, _| !existing.eq_ignore_ascii_case(&name));
        self.headers.insert(name, value.into());
        self
    }
}

pub(super) fn validate(headers: &BTreeMap<String, String>) -> Result<()> {
    // Deferring malformed metadata to `send()` turns a configuration defect into a
    // request-time failure and makes the two public request paths fail at different points.
    insert_all(&mut HeaderMap::new(), headers)
}

pub(crate) fn insert_all(target: &mut HeaderMap, headers: &BTreeMap<String, String>) -> Result<()> {
    for (name, value) in headers {
        let parsed_name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| Error::config(format!("Invalid HTTP header '{name}': invalid name")))?;
        let parsed_value = HeaderValue::from_bytes(value.as_bytes())
            .map_err(|_| Error::config(format!("Invalid HTTP header '{name}': invalid value")))?;
        target.insert(parsed_name, parsed_value);
    }
    Ok(())
}
