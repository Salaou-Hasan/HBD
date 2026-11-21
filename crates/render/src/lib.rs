//! Renderer architecture.
//!
//! M1 owns a shadow → HDR geometry → tonemap pipeline with an instanced mesh
//! pass, procedural sky, analytic IBL, cascaded shadows, SSAO, bloom, and AgX
//! tonemapping. HBTR temporal resolve is the default anti-aliasing path; see
//! `docs/ANTI-ALIASING.md` and `docs/adr/0005-hbtr.md`.
//!
//! Path tracing is research-only and disabled by default in `RendererConfig`.

use ash::vk;
use hbd_gfx::GfxError;
use thiserror::Error;

mod bloom;
mod camera;
mod hbtr;
mod mesh;
mod profiler;
mod render_targets;
mod renderer;
pub mod shader;
mod shadow;
mod ssao;
mod tonemap;

pub use camera::{Camera, DirectionalLight};
pub use mesh::Instance;

/// Internal render resolution. The window/swapchain size never changes: all
/// offscreen passes run at this resolution and the tonemap pass upscales with
/// a linear sampler into the presentation target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderResolution {
    /// Render at the swapchain's own resolution (lossless, default).
    #[default]
    Native,
    /// 1920x1080.
    P1080,
    /// 2560x1440.
    P1440,
    /// DCI 2K: 2048x1080.
    P2K,
    /// UHD 4K: 3840x2160.
    P4K,
    /// Explicit dimensions. Zero dimensions fall back to `Native`.
    Custom(u32, u32),
}

impl RenderResolution {
    /// Resolves the preset against the current swapchain extent.
    pub fn extent(self, native: vk::Extent2D) -> vk::Extent2D {
        let (width, height) = match self {
            Self::Native => (native.width, native.height),
            Self::P1080 => (1920, 1080),
            Self::P1440 => (2560, 1440),
            Self::P2K => (2048, 1080),
            Self::P4K => (3840, 2160),
            Self::Custom(width, height) => (width, height),
        };
        if width == 0 || height == 0 {
            native
        } else {
            vk::Extent2D { width, height }
        }
    }

    /// Short, log-friendly name.
    pub fn label(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::P1080 => "1080p",
            Self::P1440 => "1440p",
            Self::P2K => "2k",
            Self::P4K => "4k",
            Self::Custom(_, _) => "custom",
        }
    }
}

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

    #[error("OBJ load failed: {0}")]
    Obj(String),

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
    /// HBD Temporal Resolve (HBTR) temporal anti-aliasing path.
    #[default]
    TemporalResolve,
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
    /// Internal render resolution; independent of the window size.
    pub resolution: RenderResolution,
    /// Legacy scalar kept for source compatibility; `resolution` is authoritative.
    pub render_scale: f32,
    pub vram_budget_bytes: u64,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            path: RenderPath::Hybrid,
            anti_aliasing: AntiAliasing::TemporalResolve,
            hdr: true,
            path_tracing: false,
            resolution: RenderResolution::Native,
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
        assert_eq!(config.anti_aliasing, AntiAliasing::TemporalResolve);
        assert!(config.hdr, "HDR must be on by default");
        assert!(!config.path_tracing, "path tracing must be off by default");
        assert_eq!(config.vram_budget_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(config.resolution, RenderResolution::Native);
    }

    #[test]
    fn render_resolution_presets_are_independent_of_the_window() {
        let window = vk::Extent2D {
            width: 1280,
            height: 720,
        };
        assert_eq!(RenderResolution::Native.extent(window), window);
        assert_eq!(
            RenderResolution::P1080.extent(window),
            vk::Extent2D {
                width: 1920,
                height: 1080
            }
        );
        assert_eq!(
            RenderResolution::P1440.extent(window),
            vk::Extent2D {
                width: 2560,
                height: 1440
            }
        );
        assert_eq!(
            RenderResolution::P2K.extent(window),
            vk::Extent2D {
                width: 2048,
                height: 1080
            }
        );
        assert_eq!(
            RenderResolution::P4K.extent(window),
            vk::Extent2D {
                width: 3840,
                height: 2160
            }
        );
        assert_eq!(
            RenderResolution::Custom(800, 600).extent(window),
            vk::Extent2D {
                width: 800,
                height: 600
            }
        );
        assert_eq!(RenderResolution::Custom(0, 0).extent(window), window);
    }
}
