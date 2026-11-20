//! Asset database, cooking, and streaming.
//!
//! Runtime loads cooked, chunked assets from `.pak` archives. Source formats
//! (glTF, PNG, FBX) are imported by the offline cooker, never by the shipping
//! runtime.

use hbd_core::Handle;

/// Stable, content-independent asset identifier stored in `.meta` files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetId([u8; 16]);

impl AssetId {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetKind {
    Mesh,
    Texture,
    Material,
    Scene,
    Audio,
    Script,
    Shader,
    Other,
}

/// Runtime handle to a loaded asset.
pub type AssetHandle<T> = Handle<T>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetState {
    Unloaded,
    Queued,
    Loading,
    Ready,
    Failed,
}

/// Per-frame streaming budget. The scheduler must never exceed these without
/// explicitly entering a blocking load path.
#[derive(Debug, Clone, Copy)]
pub struct StreamingBudget {
    pub io_bytes_per_frame: u64,
    pub upload_bytes_per_frame: u64,
    pub max_io_ms: f32,
}

impl Default for StreamingBudget {
    fn default() -> Self {
        Self {
            io_bytes_per_frame: 8 * 1024 * 1024,
            upload_bytes_per_frame: 4 * 1024 * 1024,
            max_io_ms: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_budget_is_bounded() {
        let budget = StreamingBudget::default();
        assert!(budget.io_bytes_per_frame > 0);
        assert!(budget.upload_bytes_per_frame > 0);
        assert!(budget.max_io_ms > 0.0);
    }
}
