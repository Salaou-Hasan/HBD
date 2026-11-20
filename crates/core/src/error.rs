use thiserror::Error;

/// Top-level engine error.
///
/// Crates should prefer their own typed errors internally and convert to this
/// type at the public boundary.
#[derive(Debug, Error)]
pub enum HbdError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("platform error: {0}")]
    Platform(String),

    #[error("graphics error: {0}")]
    Gfx(String),

    #[error("asset error: {0}")]
    Asset(String),

    #[error("physics error: {0}")]
    Physics(String),

    #[error("script error: {0}")]
    Script(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, HbdError>;
