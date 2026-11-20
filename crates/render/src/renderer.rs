use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ash::vk;
use hbd_core::math::{Mat4, Vec2};
use hbd_gfx::VulkanContext;
use tracing::info;

use crate::bloom::BloomPass;
use crate::camera::{Camera, DirectionalLight};
use crate::mesh::MeshPass;
use crate::profiler::Profiler;
use crate::render_targets::RenderTargets;
use crate::shadow::ShadowPass;
use crate::ssao::SsaoPass;
use crate::taa::TaaPass;
use crate::tonemap::TonemapPass;
use crate::{DebugView, RenderError};

/// HBD Temporal Resolve (HBTR) is the default AA path. See `docs/ANTI-ALIASING.md`.
const HBTR_ENABLED: bool = true;

/// Timestamp marker indices. They must match the pass order in `PASS_NAMES`.
const MARK_SHADOW: usize = 1;
const MARK_GEOMETRY: usize = 2;
const MARK_TAA: usize = 3;
const MARK_SSAO: usize = 4;
const MARK_BLOOM: usize = 5;
const MARK_TONEMAP: usize = 6;
const MARK_FRAME_END: usize = 7;

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
    profiler: RefCell<Option<Profiler>>,
    context: VulkanContext,
    frame_index: u64,
    /// Visible, explicit offscreen-pass cache. Only used when the view is
    /// bit-identical to the pose that produced a given image's cached passes.
    last_view_projection: Option<Mat4>,
    pass_valid: Vec<bool>,
    pass_cache_enabled: bool,
    cached_frames: u64,
    full_frames: u64,
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
        let presented_views = if HBTR_ENABLED {
            taa.resolved_views()
        } else {
            targets.hdr_views()
        };
        let bloom = BloomPass::new(&context, presented_views, extent)?;
        let tonemap = TonemapPass::new(
            &context,
            presented_views,
            bloom.b_views(),
            ssao.denoised_views(),
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
            profiler: RefCell::new(None),
            context,
            frame_index: 0,
            last_view_projection: None,
            pass_valid: vec![false; image_count],
            pass_cache_enabled: true,
            cached_frames: 0,
            full_frames: 0,
            obj_path,
        })
    }

    /// Explicit A/B switch for the offscreen-pass cache. Always logged.
    pub fn set_pass_cache_enabled(&mut self, enabled: bool) {
        if self.pass_cache_enabled != enabled {
            tracing::info!(enabled, "offscreen pass cache toggled; cache invalidated");
            self.pass_cache_enabled = enabled;
            self.last_view_projection = None;
            self.pass_valid.fill(false);
        } else {
            tracing::info!(enabled, "offscreen pass cache already in requested state");
        }
    }

    /// Explicit instance-count override for GPU-throughput testing.
    pub fn set_instance_count(&mut self, count: u32) -> Result<(), RenderError> {
        self.mesh.set_instance_count(count)?;
        tracing::info!(
            count,
            base = "indirect draw",
            "instance count override applied"
        );
        Ok(())
    }

    /// Explicitly enables E2E profiling. Every subsequent pass records visible
    /// timestamps; a periodic summary is logged and an optional CSV is written
    /// by `finish_profiling`.
    pub fn enable_profiling(
        &self,
        report_path: Option<PathBuf>,
        report_interval: u32,
    ) -> Result<(), RenderError> {
        let profiler = Profiler::new(&self.context, report_path, report_interval)?;
        *self.profiler.borrow_mut() = Some(profiler);
        Ok(())
    }

    /// Writes the profile report and logs the totals. Called on shutdown.
    pub fn finish_profiling(&self) {
        if let Some(profiler) = self.profiler.borrow().as_ref() {
            profiler.finish();
        } else {
            tracing::info!("profiling was never enabled; nothing to finalize");
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RenderError> {
        let before = self.context.swapchain_extent();
        self.context.resize(width, height)?;
        let after = self.context.swapchain_extent();
        if before.width == after.width && before.height == after.height {
            tracing::info!(
                width,
                height,
                "resize ignored: swapchain extent unchanged (no pass rebuild)"
            );
            return Ok(());
        }
        tracing::info!(
            ?before,
            ?after,
            "resize accepted: rebuilding all render passes"
        );

        let Self {
            tonemap,
            taa,
            ssao,
            bloom,
            mesh,
            shadow,
            targets,
            profiler: _,
            context,
            frame_index: _,
            last_view_projection,
            pass_valid,
            pass_cache_enabled: _,
            cached_frames: _,
            full_frames: _,
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
        let presented_views = if HBTR_ENABLED {
            new_taa.resolved_views()
        } else {
            new_targets.hdr_views()
        };
        let new_bloom = BloomPass::new(context, presented_views, extent)?;
        let new_tonemap = TonemapPass::new(
            context,
            presented_views,
            new_bloom.b_views(),
            new_ssao.denoised_views(),
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
        *last_view_projection = None;
        *pass_valid = vec![false; image_count];
        tracing::info!(
            images = image_count,
            "offscreen pass cache invalidated on resize"
        );

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
            profiler,
            context,
            frame_index,
            last_view_projection,
            pass_valid,
            pass_cache_enabled,
            cached_frames,
            full_frames,
            obj_path: _,
        } = self;
        let extent = context.swapchain_extent();
        let aspect = extent.width as f32 / extent.height.max(1) as f32;
        let light = DirectionalLight::default();
        let frame_number = *frame_index;
        let slot = (frame_number % 2) as usize;
        let view_projection = camera.view_projection(aspect);
        let camera_unchanged =
            *pass_cache_enabled && *last_view_projection == Some(view_projection);

        if camera_unchanged {
            *cached_frames += 1;
        } else {
            *full_frames += 1;
            *last_view_projection = Some(view_projection);
            pass_valid.fill(false);
            tracing::info!(
                frame = frame_number,
                "camera changed or cache disabled: all offscreen passes invalidated"
            );
        }
        if frame_number > 0 && frame_number.is_multiple_of(240) {
            tracing::info!(
                frame = frame_number,
                full = *full_frames,
                cached = *cached_frames,
                "offscreen pass cache accounting"
            );
        }

        let cpu_start = if profiler.borrow().is_some() {
            Some(Instant::now())
        } else {
            None
        };
        if let Some(active) = profiler.borrow_mut().as_mut() {
            active.resolve_slot(slot, frame_number);
        }

        let jitter = if HBTR_ENABLED {
            Vec2::new(
                (halton(*frame_index + 1, 2) - 0.5) * 2.0 / extent.width as f32,
                (halton(*frame_index + 1, 3) - 0.5) * 2.0 / extent.height as f32,
            )
        } else {
            Vec2::ZERO
        };
        let jittered_view_projection = camera.view_projection_with_jitter(aspect, jitter);
        let unjittered_view_projection = view_projection;
        let geometry_clear_values = targets.clear_values();
        let mut taa_error: Option<RenderError> = None;

        if camera_unchanged {
            tracing::info!(
                frame = frame_number,
                "uniform uploads skipped: camera pose unchanged"
            );
        } else {
            mesh.update(camera, &light, aspect, jitter)?;
            ssao.update_all(camera, aspect)?;
        }

        context
            .render(
                clear_color,
                |device, command_buffer, image_index, extent| {
                    if let Some(active) = profiler.borrow_mut().as_mut() {
                        active.begin_frame(command_buffer, slot, frame_number);
                    }

                    let reuse = camera_unchanged
                        && pass_valid
                            .get(image_index as usize)
                            .copied()
                            .unwrap_or(false);
                    if reuse {
                        tracing::info!(
                            frame = frame_number,
                            image_index,
                            "offscreen cache HIT: shadow/geometry/SSAO/bloom skipped, tonemap only"
                        );
                    } else {
                        if let Some(slot_valid) = pass_valid.get_mut(image_index as usize) {
                            *slot_valid = true;
                        }
                        // Write only the acquired image's TAA uniform, and do it
                        // after its in-flight fence has been waited on.
                        if HBTR_ENABLED
                            && let Err(error) =
                                taa.begin_frame(image_index, jittered_view_projection)
                        {
                            taa_error = Some(error);
                            return;
                        }
                        shadow.record(command_buffer);
                    }
                    if let Some(active) = profiler.borrow_mut().as_mut() {
                        active.mark(command_buffer, slot, MARK_SHADOW, frame_number);
                    }

                    if !reuse {
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
                    }
                    if let Some(active) = profiler.borrow_mut().as_mut() {
                        active.mark(command_buffer, slot, MARK_GEOMETRY, frame_number);
                    }

                    if HBTR_ENABLED && !reuse {
                        taa.record(command_buffer, image_index);
                        taa.end_frame_for_index(unjittered_view_projection, image_index);
                    }
                    if let Some(active) = profiler.borrow_mut().as_mut() {
                        active.mark(command_buffer, slot, MARK_TAA, frame_number);
                    }

                    if !reuse {
                        ssao.record(command_buffer, image_index);
                    }
                    if let Some(active) = profiler.borrow_mut().as_mut() {
                        active.mark(command_buffer, slot, MARK_SSAO, frame_number);
                    }

                    if !reuse {
                        bloom.record(command_buffer, image_index);
                    }
                    if let Some(active) = profiler.borrow_mut().as_mut() {
                        active.mark(command_buffer, slot, MARK_BLOOM, frame_number);
                    }
                },
                |_device, command_buffer, image_index, extent| {
                    tonemap.record(command_buffer, extent, image_index);
                    if let Some(active) = profiler.borrow_mut().as_mut() {
                        active.mark(command_buffer, slot, MARK_TONEMAP, frame_number);
                        active.mark(command_buffer, slot, MARK_FRAME_END, frame_number);
                    }
                },
            )
            .map_err(RenderError::Gfx)?;

        if let Some(start) = cpu_start {
            let mut guard = profiler.borrow_mut();
            if let Some(active) = guard.as_mut() {
                active.record_cpu_frame(start.elapsed().as_secs_f32() * 1000.0);
            }
        }

        if let Some(error) = taa_error {
            return Err(error);
        }

        *frame_index += 1;
        Ok(())
    }

    pub fn set_fps(&mut self, fps: f32) {
        self.tonemap.set_fps(fps);
    }

    pub fn set_debug_view(&mut self, view: DebugView) -> Result<(), RenderError> {
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
