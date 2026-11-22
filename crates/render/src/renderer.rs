use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ash::vk;
use hbd_core::math::{Mat4, Vec2};
use hbd_gfx::VulkanContext;
use tracing::info;

use crate::bloom::BloomPass;
use crate::camera::{Camera, DirectionalLight};
use crate::hbtr::HbtrPass;
use crate::mesh::{Instance, MeshPass};
use crate::profiler::{FRAME_SLOTS, Profiler};
use crate::render_targets::RenderTargets;
use crate::shadow::ShadowPass;
use crate::ssao::SsaoPass;
use crate::tonemap::TonemapPass;
use crate::{DebugView, HbtrQuality, RenderError, RenderResolution, RendererConfig};

/// HBD Temporal Resolve (HBTR) is the default AA path. See `docs/ANTI-ALIASING.md`.
const HBTR_ENABLED: bool = true;

/// Timestamp marker indices. They must match the pass order in `PASS_NAMES`.
const MARK_SHADOW: usize = 1;
const MARK_GEOMETRY: usize = 2;
const MARK_HBTR: usize = 3;
const MARK_SSAO: usize = 4;
const MARK_BLOOM: usize = 5;
const MARK_TONEMAP: usize = 6;
const MARK_FRAME_END: usize = 7;

/// Explicit, user-driven renderer settings.
///
/// Passes are destroyed and rebuilt on resize. This struct is the single
/// source of truth for every explicit override, and `resize` re-applies it to
/// the fresh passes. Adding a new toggle here plus `apply_settings` prevents
/// it from being silently reset when the window changes size.
#[derive(Debug, Clone)]
struct RendererSettings {
    cull_enabled: bool,
    instance_count: Option<u32>,
    /// Internal render resolution, independent of the swapchain window size.
    resolution: RenderResolution,
    /// Temporal resolver work tier.
    hbtr_quality: HbtrQuality,
    light: DirectionalLight,
    hdr_enabled: bool,
    debug_view: DebugView,
    object_animation: bool,
    pass_cache_enabled: bool,
    shadow_live: bool,
    /// Last live entity poses, replayed into the rebuilt shadow pass.
    shadow_instances: Vec<Instance>,
}

impl RendererSettings {
    fn new(
        instance_count: Option<u32>,
        hdr_enabled: bool,
        resolution: RenderResolution,
        hbtr_quality: HbtrQuality,
    ) -> Self {
        Self {
            cull_enabled: true,
            instance_count,
            resolution,
            hbtr_quality,
            light: DirectionalLight::default(),
            hdr_enabled,
            debug_view: DebugView::default(),
            object_animation: false,
            pass_cache_enabled: true,
            shadow_live: false,
            shadow_instances: Vec::new(),
        }
    }
}

/// Tonemap debug mode for a `DebugView`. Kept in one place so `set_debug_view`
/// and the post-resize restore cannot disagree.
fn debug_view_mode(view: DebugView) -> u32 {
    match view {
        DebugView::Final => 0,
        DebugView::Filmic => 1,
        DebugView::Hdr => 2,
        DebugView::Bloom => 3,
        DebugView::AmbientOcclusion => 4,
        DebugView::Depth => 5,
        DebugView::ShadowCascade => 6,
        DebugView::TemporalHistory => 7,
    }
}

/// Owns the presentation context and the full
/// shadow → HDR geometry → HBTR → SSAO → bloom → tonemap pipeline.
///
/// Field order matters: render resources are destroyed before `context` drops
/// the device and allocator. Descriptor consumers are declared before the
/// resources they reference.
pub struct Renderer {
    tonemap: TonemapPass,
    hbtr: HbtrPass,
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
    cached_frames: u64,
    full_frames: u64,
    /// Temporary early-visualization override; not part of the asset pipeline.
    obj_path: Option<PathBuf>,
    /// All explicit overrides, restored after every pass rebuild.
    settings: RendererSettings,
    /// Swapchain generation observed when the passes were last built. Any
    /// recreation (including implicit suboptimal presents) invalidates them.
    swapchain_generation: u64,
    /// Last presented image; used only by the explicit cull-stats readback.
    last_image_index: u32,
}

impl Renderer {
    pub fn new(context: VulkanContext) -> Result<Self, RenderError> {
        let config = RendererConfig::default();
        Self::create(
            context,
            None,
            None,
            config.hdr,
            config.resolution,
            config.hbtr_quality,
        )
    }

    /// Early-validation hook: render a single OBJ instead of the cube grid.
    /// See `docs/EARLY-OBJ-TEST.md`. This is not the asset pipeline.
    pub fn new_with_obj(
        context: VulkanContext,
        obj_path: impl AsRef<Path>,
    ) -> Result<Self, RenderError> {
        let config = RendererConfig::default();
        Self::create(
            context,
            Some(obj_path.as_ref().to_path_buf()),
            None,
            config.hdr,
            config.resolution,
            config.hbtr_quality,
        )
    }

    /// GPU-throughput stress mode: generate `count` cubes on the GPU.
    pub fn new_with_instance_count(
        context: VulkanContext,
        count: u32,
    ) -> Result<Self, RenderError> {
        let config = RendererConfig::default();
        Self::create(
            context,
            None,
            Some(count),
            config.hdr,
            config.resolution,
            config.hbtr_quality,
        )
    }

    /// Constructs the renderer from the public config. `hdr: false` presents
    /// clamped scene-referred color directly, keeping saturated palette colors.
    pub fn new_with_config(
        context: VulkanContext,
        config: RendererConfig,
    ) -> Result<Self, RenderError> {
        Self::create(
            context,
            None,
            None,
            config.hdr,
            config.resolution,
            config.hbtr_quality,
        )
    }

    fn create(
        context: VulkanContext,
        obj_path: Option<PathBuf>,
        instance_count: Option<u32>,
        hdr_enabled: bool,
        resolution: RenderResolution,
        hbtr_quality: HbtrQuality,
    ) -> Result<Self, RenderError> {
        let swapchain_extent = context.swapchain_extent();
        let extent = resolution.extent(swapchain_extent);
        let image_count = context.swapchain_image_views().len();
        let light = DirectionalLight::default();
        let settings = RendererSettings::new(instance_count, hdr_enabled, resolution, hbtr_quality);
        info!(
            resolution = resolution.label(),
            width = extent.width,
            height = extent.height,
            window_width = swapchain_extent.width,
            window_height = swapchain_extent.height,
            "internal render resolution selected"
        );

        let targets = RenderTargets::new(&context, extent, image_count)?;
        let mut shadow = ShadowPass::new(&context, &light)?;
        let mesh = MeshPass::new(
            &context,
            targets.render_pass(),
            shadow.views(),
            shadow.sampler(),
            obj_path.as_deref(),
            instance_count,
            image_count as u32,
            targets.depth_views(),
            extent,
        )?;
        let (source, source_size, per_image, per_image_size) = mesh.shadow_instance_buffers();
        shadow.attach_mesh(source, source_size, &per_image, per_image_size);
        let hbtr = HbtrPass::new(
            &context,
            targets.hdr_views(),
            targets.depth_views(),
            targets.velocity_views(),
            extent,
            swapchain_extent,
            settings.hbtr_quality,
        )?;
        let ssao = SsaoPass::new(&context, targets.depth_views(), extent)?;
        let presented_views = if HBTR_ENABLED {
            hbtr.resolved_views()
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
            hbtr.history_views(),
        )?;
        let mut tonemap = tonemap;
        tonemap.set_hdr_enabled(hdr_enabled)?;
        info!(
            hdr = hdr_enabled,
            "presentation path selected (clamped LDR when off, AgX when on)"
        );

        info!(
            format = ?targets.hdr_format(),
            width = extent.width,
            height = extent.height,
            targets = image_count,
            "HDR geometry targets ready"
        );
        let swapchain_generation = context.swapchain_generation();

        Ok(Self {
            tonemap,
            hbtr,
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
            cached_frames: 0,
            full_frames: 0,
            obj_path,
            settings,
            swapchain_generation,
            last_image_index: 0,
        })
    }

    /// Explicit A/B switch for GPU frustum culling. Toggling invalidates the
    /// pass cache so the next frame re-culls with the new setting.
    pub fn set_culling_enabled(&mut self, enabled: bool) {
        self.settings.cull_enabled = enabled;
        self.mesh.set_culling_enabled(enabled);
        self.last_view_projection = None;
        self.pass_valid.fill(false);
        info!(enabled, "GPU frustum culling explicitly toggled");
    }

    /// Explicit demo animation switch for motion-vector validation. The pose
    /// pass writes the compacted buffers, so it requires culling and it
    /// invalidates the static-camera pass cache.
    pub fn set_object_animation(&mut self, enabled: bool) {
        self.settings.object_animation = enabled && self.settings.cull_enabled;
        if enabled && !self.settings.cull_enabled {
            info!("object animation requires culling; staying static");
        }
        if enabled {
            self.last_view_projection = None;
            self.pass_valid.fill(false);
        }
        info!(
            enabled = self.settings.object_animation,
            "object animation toggled"
        );
    }

    /// Explicit, stalled readback of the last culled count. Returns
    /// (visible instances, total instances, triangles per instance).
    pub fn cull_counts(&self) -> Result<(u32, u32, u32), RenderError> {
        let visible = self
            .mesh
            .read_visible_count(&self.context, self.last_image_index)?;
        Ok((
            visible,
            self.mesh.total_count(),
            self.mesh.triangles_per_instance(),
        ))
    }

    /// Explicit, stalled readback that proves the animation pass wrote poses.
    /// Returns (instances with current != previous model, visible, max delta).
    pub fn visible_motion(&self) -> Result<(u32, u32, f32), RenderError> {
        let visible = self
            .mesh
            .read_visible_count(&self.context, self.last_image_index)?;
        let (moved, max_delta) =
            self.mesh
                .read_visible_motion(&self.context, self.last_image_index, visible)?;
        Ok((moved, visible, max_delta))
    }

    /// Explicit presentation-path switch. Mirrors `RendererConfig::hdr`.
    pub fn set_hdr_enabled(&mut self, enabled: bool) -> Result<(), RenderError> {
        self.tonemap.set_hdr_enabled(enabled)?;
        self.settings.hdr_enabled = enabled;
        info!(hdr = enabled, "presentation path explicitly changed");
        Ok(())
    }

    /// Explicit directional-light override, used by the ECS scene bridge.
    /// Shadow cascades are still built for the default light; this updates the
    /// geometry/shading uniform from the next frame onward.
    pub fn set_light(&mut self, light: DirectionalLight) {
        info!(direction = ?light.direction, intensity = light.intensity, "directional light override applied");
        self.settings.light = light;
    }

    /// Replaces the instance data the shadow pass draws, hot-reloading the
    /// shadow scene. Empty input keeps the data but returns the shadow to the
    /// mesh's live culled set, so the default startup scene still renders.
    /// Invalidate the pass cache: shadows must redraw when entities change.
    pub fn update_instances(&mut self, instances: &[Instance]) -> Result<(), RenderError> {
        self.shadow.update_instances(&self.context, instances)?;
        let live = !instances.is_empty();
        if live {
            self.last_view_projection = None;
            self.pass_valid.fill(false);
            tracing::info!(count = instances.len(), "shadow instances hot-reloaded");
            self.settings.shadow_instances = instances.to_vec();
        } else {
            self.settings.shadow_instances.clear();
        }
        self.settings.shadow_live = live;
        Ok(())
    }

    /// Rewrites the visible scene's instance data through the engine's live
    /// path: the update is recorded with `vkCmdUpdateBuffer` into the current
    /// frame, targeting the acquired image's host-visible live source buffer,
    /// so it is ordered before the cull pass and the no-cull draw read it.
    /// Empty input clears any queued update. Always logged; invalidates the
    /// pass cache so edited poses re-render.
    pub fn update_scene_instances(&mut self, instances: &[Instance]) -> Result<(), RenderError> {
        self.mesh.set_pending_live_instances(instances)?;
        self.last_view_projection = None;
        self.pass_valid.fill(false);
        tracing::info!(
            count = instances.len(),
            "scene instances update invalidated offscreen pass cache"
        );
        Ok(())
    }

    /// Explicit A/B switch for the offscreen-pass cache. Always logged.
    pub fn set_pass_cache_enabled(&mut self, enabled: bool) {
        if self.settings.pass_cache_enabled != enabled {
            tracing::info!(enabled, "offscreen pass cache toggled; cache invalidated");
            self.settings.pass_cache_enabled = enabled;
            self.last_view_projection = None;
            self.pass_valid.fill(false);
        } else {
            tracing::info!(enabled, "offscreen pass cache already in requested state");
        }
    }

    /// Changes the temporal resolver work tier without rebuilding passes.
    /// All tiers share the same shader and visual grammar; higher tiers only
    /// enable extra rectification/detail work.
    pub fn set_hbtr_quality(&mut self, quality: HbtrQuality) {
        if self.settings.hbtr_quality == quality {
            info!(tier = quality.level(), "HBTR work tier unchanged");
            return;
        }
        self.settings.hbtr_quality = quality;
        self.hbtr.set_quality(quality);
    }

    /// Changes the internal render resolution and rebuilds every pass. The
    /// window and swapchain are untouched: the tonemap pass upscales the
    /// resolved image into the presentation target with a linear filter.
    pub fn set_render_resolution(
        &mut self,
        resolution: RenderResolution,
    ) -> Result<(), RenderError> {
        if self.settings.resolution == resolution {
            info!(
                resolution = resolution.label(),
                "render resolution unchanged"
            );
            return Ok(());
        }
        self.settings.resolution = resolution;
        self.last_view_projection = None;
        self.pass_valid.fill(false);
        info!(
            resolution = resolution.label(),
            "internal render resolution changing"
        );
        self.rebuild_passes()
    }

    /// Extent of the internal render targets (may differ from the window).
    pub fn render_extent(&self) -> vk::Extent2D {
        self.settings
            .resolution
            .extent(self.context.swapchain_extent())
    }

    /// Explicit instance-count override for GPU-throughput testing. Stored in
    /// the settings so a resize rebuilds the same instance scene.
    pub fn set_instance_count(&mut self, count: u32) -> Result<(), RenderError> {
        self.settings.instance_count = Some(count);
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
        let generation_before = self.context.swapchain_generation();
        self.context.resize(width, height)?;
        let after = self.context.swapchain_extent();
        let generation_after = self.context.swapchain_generation();
        if generation_before == generation_after
            && before.width == after.width
            && before.height == after.height
        {
            tracing::info!(
                width,
                height,
                "resize ignored: swapchain unchanged (no pass rebuild)"
            );
            return Ok(());
        }
        tracing::info!(
            ?before,
            ?after,
            generation_before,
            generation_after,
            "swapchain changed: rebuilding all render passes"
        );
        self.swapchain_generation = generation_after;
        self.rebuild_passes()
    }

    /// Rebuilds every pass against the current swapchain and restores all
    /// explicit settings. Called from `resize` and whenever the swapchain was
    /// recreated outside a resize event.
    /// Rebuilds every pass against the current swapchain and settings. The
    /// editor calls this after a shader file changes under the `hot-reload`
    /// feature.
    pub fn rebuild_passes(&mut self) -> Result<(), RenderError> {
        let Self {
            tonemap,
            hbtr,
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
            cached_frames: _,
            full_frames: _,
            obj_path,
            settings,
            swapchain_generation: _,
            last_image_index: _,
        } = self;
        let swapchain_extent = context.swapchain_extent();
        let extent = settings.resolution.extent(swapchain_extent);
        let image_count = context.swapchain_image_views().len();
        let light = settings.light;
        info!(
            resolution = settings.resolution.label(),
            width = extent.width,
            height = extent.height,
            window_width = swapchain_extent.width,
            window_height = swapchain_extent.height,
            "rebuilding passes at internal render resolution"
        );

        let new_targets = RenderTargets::new(context, extent, image_count)?;
        let mut new_shadow = ShadowPass::new(context, &light)?;
        let mut new_mesh = MeshPass::new(
            context,
            new_targets.render_pass(),
            new_shadow.views(),
            new_shadow.sampler(),
            obj_path.as_deref(),
            settings.instance_count,
            image_count as u32,
            new_targets.depth_views(),
            extent,
        )?;
        let (source, source_size, per_image, per_image_size) = new_mesh.shadow_instance_buffers();
        new_shadow.attach_mesh(source, source_size, &per_image, per_image_size);

        // Single restore point for explicit overrides. Fresh passes start with
        // defaults, so without this a maximize (or any resize) silently drops
        // --no-cull, the instance count, the debug view, and live entities.
        new_mesh.set_culling_enabled(settings.cull_enabled);
        if let Some(count) = settings.instance_count {
            new_mesh.set_instance_count(count)?;
        }
        if settings.shadow_live && !settings.shadow_instances.is_empty() {
            new_shadow.update_instances(context, &settings.shadow_instances)?;
        }
        info!(
            cull = settings.cull_enabled,
            instance_count = ?settings.instance_count,
            debug_view = ?settings.debug_view,
            shadow_live = settings.shadow_live,
            hbtr_tier = settings.hbtr_quality.level(),
            "explicit renderer settings restored after resize"
        );
        let new_hbtr = HbtrPass::new(
            context,
            new_targets.hdr_views(),
            new_targets.depth_views(),
            new_targets.velocity_views(),
            extent,
            swapchain_extent,
            settings.hbtr_quality,
        )?;
        let new_ssao = SsaoPass::new(context, new_targets.depth_views(), extent)?;
        let presented_views = if HBTR_ENABLED {
            new_hbtr.resolved_views()
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
            new_hbtr.history_views(),
        )?;
        let mut new_tonemap = new_tonemap;
        new_tonemap.set_hdr_enabled(settings.hdr_enabled)?;
        new_tonemap.set_mode(debug_view_mode(settings.debug_view))?;

        // Drop descriptor consumers before the resources they reference.
        *tonemap = new_tonemap;
        *hbtr = new_hbtr;
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

    /// Renders a frame with no editor/UI overlay.
    pub fn render(&mut self, camera: &Camera, clear_color: [f32; 4]) -> Result<(), RenderError> {
        self.render_with_overlay(camera, clear_color, |_, _, _, _| {})
    }

    /// Renders a frame and invokes `overlay` inside the presentation pass,
    /// after the tonemap draw. The editor uses this to submit ImGui draw data
    /// into the same swapchain render pass without a second window or wgpu.
    pub fn render_with_overlay<F>(
        &mut self,
        camera: &Camera,
        clear_color: [f32; 4],
        overlay: F,
    ) -> Result<(), RenderError>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer, u32, vk::Extent2D),
    {
        // `VulkanContext::render` can recreate a suboptimal/out-of-date
        // swapchain on its own. Its image views and per-image resources are
        // then stale, so rebuild before drawing anything into them.
        let generation = self.context.swapchain_generation();
        if self.swapchain_generation != generation {
            tracing::info!(
                from = self.swapchain_generation,
                to = generation,
                "swapchain recreated outside resize: rebuilding all render passes"
            );
            self.swapchain_generation = generation;
            self.rebuild_passes()?;
        }

        // Destructure is split because `frame_index`/`hbtr` need mutable access
        // while the other resources are only borrowed.
        let Self {
            tonemap,
            hbtr,
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
            cached_frames,
            full_frames,
            obj_path: _,
            settings,
            swapchain_generation: _,
            last_image_index,
        } = self;
        let swapchain_extent = context.swapchain_extent();
        let render_extent = settings.resolution.extent(swapchain_extent);
        let aspect = render_extent.width as f32 / render_extent.height.max(1) as f32;
        let light = settings.light;
        let frame_number = *frame_index;
        let slot = (frame_number % FRAME_SLOTS as u64) as usize;
        let view_projection = camera.view_projection(aspect);
        let camera_unchanged = settings.pass_cache_enabled
            && !settings.object_animation
            && *last_view_projection == Some(view_projection);

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
                (halton(*frame_index + 1, 2) - 0.5) * 2.0 / render_extent.width as f32,
                (halton(*frame_index + 1, 3) - 0.5) * 2.0 / render_extent.height as f32,
            )
        } else {
            Vec2::ZERO
        };
        let jittered_view_projection = camera.view_projection_with_jitter(aspect, jitter);
        let geometry_clear_values = targets.clear_values();
        let mut hbtr_error: Option<RenderError> = None;

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
                |device, command_buffer, image_index, _present_extent| {
                    *last_image_index = image_index;
                    if let Some(active) = profiler.borrow_mut().as_mut() {
                        active.begin_frame(command_buffer, slot, frame_number);
                    }
                    // Live scene edits land before the cull pass and the
                    // no-cull draw in this same command buffer.
                    mesh.apply_pending_live(command_buffer, image_index);

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
                        // Write only the acquired image's HBTR uniform, and do it
                        // after its in-flight fence has been waited on.
                        if HBTR_ENABLED
                            && let Err(error) =
                                hbtr.begin_frame(image_index, jittered_view_projection)
                        {
                            hbtr_error = Some(error);
                            return;
                        }
                        // Per-image cull: compacted instances and the indirect
                        // count are private to this swapchain image.
                        mesh.dispatch_cull(
                            command_buffer,
                            image_index,
                            view_projection,
                            frame_number,
                            settings.object_animation,
                        );
                        let cull_enabled = mesh.culling_enabled();
                        shadow.record(
                            command_buffer,
                            image_index,
                            cull_enabled,
                            settings.shadow_live,
                            mesh.shadow_indirect(image_index, cull_enabled),
                            mesh.shadow_indirect(image_index, false),
                        );
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
                                extent: render_extent,
                            })
                            .clear_values(&geometry_clear_values);

                        unsafe {
                            device.cmd_begin_render_pass(
                                command_buffer,
                                &render_pass_info,
                                vk::SubpassContents::INLINE,
                            );
                        }
                        mesh.record(command_buffer, render_extent, image_index);
                        unsafe {
                            device.cmd_end_render_pass(command_buffer);
                        }
                    }
                    if let Some(active) = profiler.borrow_mut().as_mut() {
                        active.mark(command_buffer, slot, MARK_GEOMETRY, frame_number);
                    }

                    if HBTR_ENABLED && !reuse {
                        hbtr.record(command_buffer, image_index);
                        // Store the jittered pose: history samples were written
                        // with this exact projection, so reprojection must use it.
                        hbtr.end_frame_for_index(jittered_view_projection, image_index);
                    }
                    if let Some(active) = profiler.borrow_mut().as_mut() {
                        active.mark(command_buffer, slot, MARK_HBTR, frame_number);
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
                |device, command_buffer, image_index, extent| {
                    tonemap.record(command_buffer, extent, image_index);
                    if let Some(active) = profiler.borrow_mut().as_mut() {
                        active.mark(command_buffer, slot, MARK_TONEMAP, frame_number);
                    }
                    overlay(device, command_buffer, image_index, extent);
                    if let Some(active) = profiler.borrow_mut().as_mut() {
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

        if let Some(error) = hbtr_error {
            return Err(error);
        }

        *frame_index += 1;
        Ok(())
    }

    pub fn set_fps(&mut self, fps: f32) {
        self.tonemap.set_fps(fps);
    }

    pub fn set_debug_view(&mut self, view: DebugView) -> Result<(), RenderError> {
        self.settings.debug_view = view;
        self.tonemap.set_mode(debug_view_mode(view))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_settings_are_defaulted_and_survive_cloning() {
        let mut settings = RendererSettings::new(
            Some(4000),
            false,
            RenderResolution::P1080,
            HbtrQuality::Medium,
        );
        assert!(settings.cull_enabled);
        assert!(settings.pass_cache_enabled);
        assert!(!settings.hdr_enabled);
        assert_eq!(settings.instance_count, Some(4000));
        assert_eq!(settings.debug_view, DebugView::Final);
        assert_eq!(settings.resolution, RenderResolution::P1080);

        settings.cull_enabled = false;
        settings.instance_count = Some(8000);
        settings.debug_view = DebugView::ShadowCascade;
        settings.shadow_live = true;
        settings.shadow_instances.push(Instance {
            model: [[0.0; 4]; 4],
            previous_model: [[0.0; 4]; 4],
            base_color: [0.0; 4],
            material: [0.0; 4],
        });

        // Resize reads these values after rebuilding every pass.
        let restored = settings.clone();
        assert!(!restored.cull_enabled);
        assert_eq!(restored.instance_count, Some(8000));
        assert_eq!(debug_view_mode(restored.debug_view), 6);
        assert_eq!(debug_view_mode(DebugView::Final), 0);
        assert_eq!(restored.resolution, RenderResolution::P1080);
        assert!(restored.shadow_live);
        assert_eq!(restored.shadow_instances.len(), 1);
    }
}
