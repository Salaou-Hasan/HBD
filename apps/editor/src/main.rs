//! HBD editor executable.
//!
//! Depends only on the `hbd-engine` facade. ImGui is drawn into the engine's
//! own Vulkan swapchain pass through `Renderer::render_with_overlay`; there is
//! no second windowing or graphics stack.

use std::sync::Arc;
use std::time::SystemTime;

use dear_imgui_ash::{AshRenderer, AshRendererConfig, Options, TextureRetirementBatch};
use dear_imgui_rs::{ConfigFlags, Context};
use dear_imgui_winit::{HiDpiMode, WinitPlatform};
use hbd_engine::core::math::Vec3;
use hbd_engine::gfx::{self, VulkanContext};
use hbd_engine::platform::{self, PlatformHandler, WindowConfig};
use hbd_engine::render::{
    Camera, DebugView, DirectionalLight, HbtrQuality, RenderResolution, Renderer, RendererConfig,
};
use tracing::{error, info, warn};
use winit::event::WindowEvent;
use winit::window::Window;

const WATCHED_SHADERS: &[&str] = &[
    "crates/render/src/mesh.wgsl",
    "crates/render/src/sky.wgsl",
    "crates/render/src/mesh_generate.wgsl",
    "crates/render/src/cull.wgsl",
    "crates/render/src/cluster.wgsl",
    "crates/render/src/hzb.wgsl",
    "crates/render/src/animate.wgsl",
    "crates/render/src/hbtr_resolve.wgsl",
    "crates/render/src/shadow.wgsl",
    "crates/render/src/ssao.wgsl",
    "crates/render/src/ssao_blur.wgsl",
    "crates/render/src/bloom.wgsl",
    "crates/render/src/tonemap.wgsl",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Selection {
    Camera,
    Light,
    Scene,
}

struct Editor {
    renderer: Option<Renderer>,
    imgui: Context,
    platform: WinitPlatform,
    ash_renderer: Option<AshRenderer>,
    pending_retirement: Option<TextureRetirementBatch>,

    selection: Selection,
    camera_position: [f32; 3],
    light_direction: [f32; 3],
    light_intensity: f32,
    resolution: RenderResolution,
    quality: HbtrQuality,
    shader_mtimes: Vec<Option<SystemTime>>,
    frame: u64,
    last_counts: (u32, u32, u32),
    error_message: Option<String>,
}

impl PlatformHandler for Editor {
    fn on_window_created(&mut self, window: &Arc<Window>) {
        let size = window.inner_size();
        let context = match VulkanContext::new_with_present_policy(
            &**window,
            size.width,
            size.height,
            gfx::PresentPolicy::Vsync,
        ) {
            Ok(context) => context,
            Err(error) => {
                error!(%error, "failed to create editor Vulkan context");
                return;
            }
        };
        let handles = (
            context.instance().clone(),
            context.device().clone(),
            context.physical_device(),
            context.graphics_queue(),
            context.command_pool(),
            context.render_pass(),
        );

        match Renderer::new_with_config(context, RendererConfig::default()) {
            Ok(renderer) => {
                info!(extent = ?renderer.swapchain_extent(), "editor renderer ready");
                self.renderer = Some(renderer);
            }
            Err(error) => {
                error!(%error, "failed to create editor renderer");
                return;
            }
        }

        let (instance, device, physical_device, queue, command_pool, render_pass) = handles;
        let config = AshRendererConfig::with_render_pass(device, queue, command_pool, render_pass)
            .with_options(Options {
                in_flight_frames: 2,
                framebuffer_srgb: true,
                ..Options::default()
            });
        // SAFETY: all handles came from the single VulkanContext above and stay
        // live for as long as the renderer. The queue supports graphics and
        // transfer; the render pass matches the swapchain pass we draw into.
        let ash_renderer = unsafe {
            AshRenderer::with_default_allocator(&instance, physical_device, config, &mut self.imgui)
        };
        match ash_renderer {
            Ok(renderer) => self.ash_renderer = Some(renderer),
            Err(error) => error!(%error, "failed to create ImGui Vulkan renderer"),
        }

        if let Err(error) =
            self.platform
                .attach_window(Arc::clone(window), HiDpiMode::Default, &mut self.imgui)
        {
            error!(%error, "failed to attach ImGui to editor window");
        }
        self.shader_mtimes = WATCHED_SHADERS.iter().map(|_| None).collect();
        self.reload_shaders_if_changed();
    }

    fn on_window_event(&mut self, _window: &Arc<Window>, event: &WindowEvent) {
        if let Some(renderer) = self.renderer.as_mut()
            && let WindowEvent::Resized(size) = event
            && let Err(error) = renderer.resize(size.width, size.height)
        {
            error!(%error, "editor resize failed");
        }

        if let Err(error) = self
            .platform
            .handle_window_event(&mut self.imgui, _window.id(), event)
        {
            warn!(%error, "ImGui event handling error");
        }

        if let WindowEvent::KeyboardInput { event: key, .. } = event
            && key.state == winit::event::ElementState::Pressed
            && let winit::keyboard::PhysicalKey::Code(code) = key.physical_key
            && code == winit::keyboard::KeyCode::Escape
        {
            info!("editor close requested via Escape");
        }
    }

    fn on_redraw(&mut self, _window: &Arc<Window>) {
        self.frame += 1;

        // Texture retirement from ImGui must complete before its resources are
        // reused. This is explicit and only happens on frames with retirements.
        RETIREMENT.with(|slot| {
            if let Some(batch) = slot.borrow_mut().take() {
                self.pending_retirement = Some(batch);
            }
        });
        if let Some(ash) = self.ash_renderer.as_mut()
            && let Some(batch) = self.pending_retirement.take()
            && let Err(error) = ash.wait_for_texture_retirements(batch)
        {
            warn!(%error, "ImGui texture retirement failed");
        }

        self.reload_shaders_if_changed();
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_light(DirectionalLight {
                direction: Vec3::from_array(self.light_direction),
                color: Vec3::new(1.0, 0.96, 0.9),
                intensity: self.light_intensity,
            });
            if self.frame.is_multiple_of(120)
                && let Ok((visible, total, tris)) = renderer.cull_counts()
            {
                self.last_counts = (visible, total, tris);
            }
        }

        if let Err(error) = self.platform.prepare_frame(&mut self.imgui, _window) {
            warn!(%error, "ImGui frame preparation failed");
            return;
        }

        self.build_ui();

        let camera = Camera::new(Vec3::from_array(self.camera_position), Vec3::ZERO);
        let clear_color = [0.015, 0.025, 0.055, 1.0];

        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let Some(ash) = self.ash_renderer.as_mut() else {
            return;
        };
        let imgui = &mut self.imgui;
        let result =
            renderer.render_with_overlay(&camera, clear_color, |device, cmd, _index, extent| {
                let _ = (device, extent);
                let consumer = match ash.renderer_consumer() {
                    Ok(consumer) => consumer,
                    Err(error) => {
                        error!(%error, "ImGui renderer consumer unavailable");
                        return;
                    }
                };
                let pending = imgui.render(consumer);
                let (frame, retirement) = match ash.prepare_frame(pending) {
                    Ok(result) => result,
                    Err(error) => {
                        error!(%error, "ImGui frame preparation failed");
                        return;
                    }
                };
                if let Some(batch) = retirement {
                    info!("ImGui texture retirement deferred to next frame");
                    self_deferred_retirement(batch);
                }
                // SAFETY: `cmd` is recording inside the engine's swapchain render
                // pass, on the same device and queue used to create this renderer.
                if let Err(error) = unsafe { ash.cmd_draw(cmd, frame) } {
                    error!(%error, "ImGui draw failed");
                }
            });
        if let Err(error) = result {
            error!(%error, "editor frame presentation failed");
        }
    }
}

// The overlay closure cannot capture `self`; retirement is recorded through a
// thread-local slot and picked up at the start of the next frame.
thread_local! {
    static RETIREMENT: std::cell::RefCell<Option<TextureRetirementBatch>> = const {
        std::cell::RefCell::new(None)
    };
}

fn self_deferred_retirement(batch: TextureRetirementBatch) {
    RETIREMENT.with(|slot| *slot.borrow_mut() = Some(batch));
}

impl Editor {
    fn build_ui(&mut self) {
        let ui = self.imgui.frame();
        let mut reload = false;
        let mut debug_view = None;
        let mut set_resolution = None;
        let mut set_quality = None;

        ui.window("Hierarchy")
            .position([20.0, 20.0], dear_imgui_rs::Condition::Always)
            .size([220.0, 260.0], dear_imgui_rs::Condition::Always)
            .build(|| {
                if ui
                    .selectable_config("Camera")
                    .selected(self.selection == Selection::Camera)
                    .build()
                {
                    self.selection = Selection::Camera;
                }
                if ui
                    .selectable_config("Directional Light")
                    .selected(self.selection == Selection::Light)
                    .build()
                {
                    self.selection = Selection::Light;
                }
                if ui
                    .selectable_config("Cube Grid (static)")
                    .selected(self.selection == Selection::Scene)
                    .build()
                {
                    self.selection = Selection::Scene;
                }
            });

        ui.window("Inspector")
            .position([20.0, 300.0], dear_imgui_rs::Condition::Always)
            .size([260.0, 240.0], dear_imgui_rs::Condition::Always)
            .build(|| match self.selection {
                Selection::Camera => {
                    ui.text("Camera");
                    let _ = ui.input_float3("Position", &mut self.camera_position);
                }
                Selection::Light => {
                    ui.text("Directional Light");
                    let _ = ui.input_float3("Direction", &mut self.light_direction);
                    ui.input_float("Intensity", &mut self.light_intensity);
                }
                Selection::Scene => {
                    ui.text("Static test scene: 82 cube instances");
                    ui.text("Materials: bindless array (neutral placeholder)");
                }
            });

        ui.window("Stats")
            .position([700.0, 20.0], dear_imgui_rs::Condition::Always)
            .size([310.0, 220.0], dear_imgui_rs::Condition::Always)
            .build(|| {
                ui.text(format!("Frame {}", self.frame));
                ui.text(format!(
                    "Resolution {:?} / {:?}",
                    self.resolution.label(),
                    self.quality
                ));
                ui.text(format!(
                    "Visible {}/{}  ({} tris/instance)",
                    self.last_counts.0, self.last_counts.1, self.last_counts.2
                ));
                if ui.button("Final") {
                    debug_view = Some(DebugView::Final);
                }
                ui.same_line();
                if ui.button("HDR") {
                    debug_view = Some(DebugView::Hdr);
                }
                ui.same_line();
                if ui.button("AO") {
                    debug_view = Some(DebugView::AmbientOcclusion);
                }
                ui.same_line();
                if ui.button("History") {
                    debug_view = Some(DebugView::TemporalHistory);
                }
            });

        ui.window("Shaders")
            .position([700.0, 260.0], dear_imgui_rs::Condition::Always)
            .size([310.0, 440.0], dear_imgui_rs::Condition::Always)
            .build(|| {
                ui.text("Hot reload: crates/render/src/");
                for (path, changed) in WATCHED_SHADERS.iter().zip(self.shader_mtimes.iter()) {
                    let status = if changed.is_some() { "ok" } else { "missing" };
                    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
                    ui.text(format!("{}: {}", name, status));
                }
                if ui.button("Reload shaders now") {
                    reload = true;
                }
                for (label, resolution) in [
                    ("Native", RenderResolution::Native),
                    ("1080p", RenderResolution::P1080),
                    ("Auto 50%", RenderResolution::Auto(50)),
                    ("Auto 85%", RenderResolution::Auto(85)),
                ] {
                    if ui.button(label) {
                        set_resolution = Some(resolution);
                    }
                    ui.same_line();
                }
                ui.new_line();
                for (label, quality) in [
                    ("Low", HbtrQuality::Low),
                    ("Medium", HbtrQuality::Medium),
                    ("High", HbtrQuality::High),
                    ("Epic", HbtrQuality::Epic),
                ] {
                    if ui.button(label) {
                        set_quality = Some(quality);
                    }
                    ui.same_line();
                }
                ui.new_line();
            });

        if reload
            && let Some(renderer) = self.renderer.as_mut()
            && let Err(error) = renderer.rebuild_passes()
        {
            self.error_message = Some(error.to_string());
        }
        if let Some(view) = debug_view
            && let Some(renderer) = self.renderer.as_mut()
            && let Err(error) = renderer.set_debug_view(view)
        {
            self.error_message = Some(error.to_string());
        }
        if let Some(resolution) = set_resolution
            && let Some(renderer) = self.renderer.as_mut()
            && let Err(error) = renderer.set_render_resolution(resolution)
        {
            self.error_message = Some(error.to_string());
        } else if let Some(resolution) = set_resolution {
            self.resolution = resolution;
        }
        if let Some(quality) = set_quality
            && let Some(renderer) = self.renderer.as_mut()
        {
            renderer.set_hbtr_quality(quality);
            self.quality = quality;
        }

        if let Some(error) = &self.error_message {
            ui.window("Error")
                .size([360.0, 120.0], dear_imgui_rs::Condition::Always)
                .build(|| {
                    ui.text_colored([1.0, 0.3, 0.3, 1.0], error);
                });
        }
    }

    fn reload_shaders_if_changed(&mut self) {
        for (index, path) in WATCHED_SHADERS.iter().enumerate() {
            let modified = std::fs::metadata(path)
                .ok()
                .and_then(|metadata| metadata.modified().ok());
            if modified != self.shader_mtimes[index] {
                self.shader_mtimes[index] = modified;
                if self.frame == 0 {
                    continue;
                }
                info!(path, "shader changed; rebuilding engine passes");
                if let Some(renderer) = self.renderer.as_mut()
                    && let Err(error) = renderer.rebuild_passes()
                {
                    self.error_message = Some(format!("{path}: {error}"));
                }
            }
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("hbd_editor=info,hbd_render=info")
            }),
        )
        .init();

    let mut imgui = Context::create();
    let flags = imgui.io().config_flags() | ConfigFlags::DOCKING_ENABLE;
    imgui.io_mut().set_config_flags(flags);
    let platform = match WinitPlatform::new(&mut imgui) {
        Ok(platform) => platform,
        Err(error) => {
            error!(%error, "failed to create ImGui winit platform");
            return;
        }
    };

    let config = WindowConfig {
        title: "HBD Editor".to_owned(),
        width: 1280,
        height: 720,
        resizable: true,
        continuous_redraw: true,
        max_frames: None,
    };
    let editor = Editor {
        renderer: None,
        imgui,
        platform,
        ash_renderer: None,
        pending_retirement: None,
        selection: Selection::Scene,
        camera_position: [9.0, 4.5, 0.0],
        light_direction: [0.35, 0.85, 0.4],
        light_intensity: 1.0,
        resolution: RenderResolution::Native,
        quality: HbtrQuality::Medium,
        shader_mtimes: vec![None; WATCHED_SHADERS.len()],
        frame: 0,
        last_counts: (0, 0, 0),
        error_message: None,
    };

    if let Err(error) = platform::run(config, editor) {
        error!(%error, "editor event loop failed");
    }
}
