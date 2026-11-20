//! Renderer architecture.
//!
//! M1 owns a shadow → HDR geometry → tonemap pipeline with an instanced mesh
//! pass, procedural sky, analytic IBL, cascaded shadows, SSAO, bloom, and AgX
//! tonemapping. The TAA pass is implemented but bypassed at runtime; see
//! `docs/RENDERER.md` and `docs/adr/0003-taa-disabled.md`.
//!
//! Path tracing is research-only and disabled by default in `RendererConfig`.

use ash::vk;
use hbd_gfx::GfxError;
use thiserror::Error;

mod bloom;
mod camera;
mod mesh;
mod render_targets;
mod renderer;
pub mod shader;
mod shadow;
mod ssao;
mod taa;
mod tonemap;

pub use camera::{Camera, DirectionalLight};

/// Selectable buffer visualization for renderer debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DebugView {
    /// AgX-tonemapped final image.
    #[default]
    Final,
    /// Legacy filmic curve, for comparison with AgX.
    Filmic,
    /// Scene-referred HDR radiance (Reinhard compressed for display).
    Hdr,
    /// Bloom contribution only.
    Bloom,
    /// Screen-space ambient occlusion only.
    AmbientOcclusion,
    /// Scene depth.
    Depth,
    /// First directional shadow cascade.
    ShadowCascade,
    /// TAA history buffer.
    TemporalHistory,
}
pub use renderer::Renderer;
pub use shader::ShaderError;

#[derive(Debug, Error)]
pub enum RenderError {
    #[error(transparent)]
    Gfx(#[from] GfxError),

    #[error(transparent)]
    Shader(#[from] ShaderError),

    #[error("failed to create the graphics pipeline: {0:?}")]
    Pipeline(vk::Result),

    #[error("the graphics pipeline creation returned no pipeline")]
    MissingPipeline,
}

/// Which lighting path the renderer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderPath {
    /// Clustered Forward+ only. Useful for debugging and low-end paths.
    ForwardPlus,
    /// Deferred G-buffer only. Useful for comparison and debugging.
    Deferred,
    /// Deferred opaque G-buffer + Forward+ for transparent/forward-only materials.
    #[default]
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AntiAliasing {
    Off,
    Msaa(u32),
    /// Temporal AA. Implemented, but bypassed by the M1 runtime because it
    /// produced artifacts; see `docs/adr/0003-taa-disabled.md`.
    #[default]
    Taa,
    /// Research track; enabled only when it wins the visual test suite.
    Custom,
}

#[derive(Debug, Clone)]
pub struct RendererConfig {
    pub path: RenderPath,
    pub anti_aliasing: AntiAliasing,
    pub hdr: bool,
    /// Research-only global-illumination experiment. Never enabled by default
    /// and not part of the M1 renderer.
    pub path_tracing: bool,
    pub render_scale: f32,
    pub vram_budget_bytes: u64,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            path: RenderPath::Hybrid,
            anti_aliasing: AntiAliasing::Taa,
            hdr: true,
            path_tracing: false,
            render_scale: 1.0,
            vram_budget_bytes: 8 * 1024 * 1024 * 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_accepted_stack() {
        let config = RendererConfig::default();
        assert_eq!(config.path, RenderPath::Hybrid);
        assert_eq!(config.anti_aliasing, AntiAliasing::Taa);
        assert!(config.hdr);
        assert!(!config.path_tracing, "path tracing must be off by default");
        assert_eq!(config.vram_budget_bytes, 8 * 1024 * 1024 * 1024);
    }
}
