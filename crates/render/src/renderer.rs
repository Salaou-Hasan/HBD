use std::path::{Path, PathBuf};

use ash::vk;
use hbd_core::math::Vec2;
use hbd_gfx::VulkanContext;
use tracing::info;

use crate::bloom::BloomPass;
use crate::camera::{Camera, DirectionalLight};
use crate::mesh::MeshPass;
use crate::render_targets::RenderTargets;
use crate::shadow::ShadowPass;
use crate::ssao::SsaoPass;
use crate::taa::TaaPass;
use crate::tonemap::TonemapPass;
use crate::{DebugView, RenderError};

/// TAA is disabled because it produces visible artifacts. Bloom and tonemap
/// read the raw HDR target instead of the TAA resolve. Set to `true` only after
/// the temporal resolve itself has been redesigned.
const TAA_ENABLED: bool = false;

/// Owns the presentation context and the full
/// shadow → HDR geometry → TAA → SSAO → bloom → tonemap pipeline.
///
/// Field order matters: render resources are destroyed before `context` drops
/// the device and allocator. Descriptor consumers are declared before the
/// resources they reference.
pub struct Renderer {
    tonemap: TonemapPass,
    taa: TaaPass,
    ssao: SsaoPass,
    bloom: BloomPass,
    mesh: MeshPass,
    shadow: ShadowPass,
    targets: RenderTargets,
    context: VulkanContext,
    frame_index: u64,
    /// Temporary early-visualization override; not part of the asset pipeline.
    obj_path: Option<PathBuf>,
}

impl Renderer {
    pub fn new(context: VulkanContext) -> Result<Self, RenderError> {
        Self::create(context, None)
    }

    /// Early-validation hook: render a single OBJ instead of the cube grid.
    /// See `docs/EARLY-OBJ-TEST.md`. This is not the asset pipeline.
    pub fn new_with_obj(
        context: VulkanContext,
        obj_path: impl AsRef<Path>,
    ) -> Result<Self, RenderError> {
        Self::create(context, Some(obj_path.as_ref().to_path_buf()))
    }

    fn create(context: VulkanContext, obj_path: Option<PathBuf>) -> Result<Self, RenderError> {
        let extent = context.swapchain_extent();
        let image_count = context.swapchain_image_views().len();
        let light = DirectionalLight::default();

        let targets = RenderTargets::new(&context, extent, image_count)?;
        let shadow = ShadowPass::new(&context, &light)?;
        let mesh = MeshPass::new(
            &context,
            targets.render_pass(),
            shadow.views(),
            shadow.sampler(),
            obj_path.as_deref(),
        )?;
        let taa = TaaPass::new(&context, targets.hdr_views(), targets.depth_views(), extent)?;
        let ssao = SsaoPass::new(&context, targets.depth_views(), extent)?;
        let presented_views = if TAA_ENABLED {
            taa.resolved_views()
        } else {
            targets.hdr_views()
        };
        let bloom = BloomPass::new(&context, presented_views, extent)?;
        let tonemap = TonemapPass::new(
            &context,
            presented_views,
            bloom.b_views(),
            ssao.views(),
            targets.depth_views(),
            shadow.views(),
            taa.history_views(),
        )?;

        info!(
            format = ?targets.hdr_format(),
            width = extent.width,
            height = extent.height,
            targets = image_count,
            "HDR geometry targets ready"
        );

        Ok(Self {
            tonemap,
            taa,
            ssao,
            bloom,
            mesh,
            shadow,
            targets,
            context,
            frame_index: 0,
            obj_path,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RenderError> {
        self.context.resize(width, height)?;

        let Self {
            tonemap,
            taa,
            ssao,
            bloom,
            mesh,
            shadow,
            targets,
            context,
            frame_index: _,
            obj_path,
        } = self;
        let extent = context.swapchain_extent();
        let image_count = context.swapchain_image_views().len();
        let light = DirectionalLight::default();

        let new_targets = RenderTargets::new(context, extent, image_count)?;
        let new_shadow = ShadowPass::new(context, &light)?;
        let new_mesh = MeshPass::new(
            context,
            new_targets.render_pass(),
            new_shadow.views(),
            new_shadow.sampler(),
            obj_path.as_deref(),
        )?;
        let new_taa = TaaPass::new(
            context,
            new_targets.hdr_views(),
            new_targets.depth_views(),
            extent,
        )?;
        let new_ssao = SsaoPass::new(context, new_targets.depth_views(), extent)?;
        let presented_views = if TAA_ENABLED {
            new_taa.resolved_views()
        } else {
            new_targets.hdr_views()
        };
        let new_bloom = BloomPass::new(context, presented_views, extent)?;
        let new_tonemap = TonemapPass::new(
            context,
            presented_views,
            new_bloom.b_views(),
            new_ssao.views(),
            new_targets.depth_views(),
            new_shadow.views(),
            new_taa.history_views(),
        )?;

        // Drop descriptor consumers before the resources they reference.
        *tonemap = new_tonemap;
        *taa = new_taa;
        *ssao = new_ssao;
        *bloom = new_bloom;
        *mesh = new_mesh;
        *shadow = new_shadow;
        *targets = new_targets;

        Ok(())
    }

    pub fn render(&mut self, camera: &Camera, clear_color: [f32; 4]) -> Result<(), RenderError> {
        // Destructure is split because `frame_index`/`taa` need mutable access
        // while the other resources are only borrowed.
        let Self {
            tonemap,
            taa,
            ssao,
            bloom,
            mesh,
            shadow,
            targets,
            context,
            frame_index,
            obj_path: _,
        } = self;
        let extent = context.swapchain_extent();
        let aspect = extent.width as f32 / extent.height.max(1) as f32;
        let light = DirectionalLight::default();

        let jitter = if TAA_ENABLED {
            Vec2::new(
                (halton(*frame_index + 1, 2) - 0.5) * 2.0 / extent.width as f32,
                (halton(*frame_index + 1, 3) - 0.5) * 2.0 / extent.height as f32,
            )
        } else {
            Vec2::ZERO
        };
        let jittered_view_projection = camera.view_projection_with_jitter(aspect, jitter);
        let unjittered_view_projection = camera.view_projection(aspect);
        let geometry_clear_values = targets.clear_values();
        let mut taa_error: Option<RenderError> = None;

        mesh.update(camera, &light, aspect, jitter)?;
        ssao.update_all(camera, aspect)?;

        context
            .render(
                clear_color,
                |device, command_buffer, image_index, extent| {
                    // Write only the acquired image's TAA uniform, and do it
                    // after its in-flight fence has been waited on.
                    if TAA_ENABLED
                        && let Err(error) = taa.begin_frame(image_index, jittered_view_projection)
                    {
                        taa_error = Some(error);
                        return;
                    }

                    shadow.record(command_buffer);

                    let render_pass_info = vk::RenderPassBeginInfo::default()
                        .render_pass(targets.render_pass())
                        .framebuffer(targets.framebuffer(image_index))
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent,
                        })
                        .clear_values(&geometry_clear_values);

                    unsafe {
                        device.cmd_begin_render_pass(
                            command_buffer,
                            &render_pass_info,
                            vk::SubpassContents::INLINE,
                        );
                    }
                    mesh.record(command_buffer, extent);
                    unsafe {
                        device.cmd_end_render_pass(command_buffer);
                    }

                    if TAA_ENABLED {
                        taa.record(command_buffer, image_index);
                        taa.end_frame_for_index(unjittered_view_projection, image_index);
                    }
                    ssao.record(command_buffer, image_index);
                    bloom.record(command_buffer, image_index);
                },
                |_device, command_buffer, image_index, extent| {
                    tonemap.record(command_buffer, extent, image_index);
                },
            )
            .map_err(RenderError::Gfx)?;

        if let Some(error) = taa_error {
            return Err(error);
        }

        *frame_index += 1;
        Ok(())
    }

    pub fn set_debug_view(&self, view: DebugView) -> Result<(), RenderError> {
        let mode = match view {
            DebugView::Final => 0,
            DebugView::Filmic => 1,
            DebugView::Hdr => 2,
            DebugView::Bloom => 3,
            DebugView::AmbientOcclusion => 4,
            DebugView::Depth => 5,
            DebugView::ShadowCascade => 6,
            DebugView::TemporalHistory => 7,
        };
        self.tonemap.set_mode(mode)
    }

    pub fn swapchain_extent(&self) -> vk::Extent2D {
        self.context.swapchain_extent()
    }

    pub fn swapchain_format(&self) -> vk::Format {
        self.context.swapchain_format()
    }
}

fn halton(index: u64, base: u64) -> f32 {
    let mut result = 0.0;
    let mut fraction = 1.0 / base as f32;
    let mut remaining = index;
    while remaining > 0 {
        result += (remaining % base) as f32 * fraction;
        remaining /= base;
        fraction /= base as f32;
    }
    result
}
