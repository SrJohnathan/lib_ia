use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error as StdError;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    NullPointer(&'static str),
    NulByte(String),
    Utf8(String),
    Api(String),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NullPointer(where_) => write!(f, "null pointer returned from {}", where_),
            Error::NulByte(value) => write!(f, "string contains interior NUL byte: {}", value),
            Error::Utf8(value) => write!(f, "invalid UTF-8 from C API: {}", value),
            Error::Api(value) => write!(f, "{}", value),
        }
    }
}

impl StdError for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GenerationType {
    Content(String),
    Reasoning(String),
    ToolPreview(ToolPreview),
    CallTool(ToolCall),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prop {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub call_id: String,
    pub command: String,
    pub args: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolPreview {
    pub path: String,
    pub action: String,
    pub content_chunk: String,
    pub reset: bool,
    pub done: bool,
    pub truncated: bool,
}
