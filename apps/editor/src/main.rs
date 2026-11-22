//! HBD editor executable.
//!
//! Depends only on the `hbd-engine` facade. ImGui is drawn into the engine's
//! own Vulkan swapchain pass through `Renderer::render_with_overlay`; there is
//! no second windowing or graphics stack.

mod gizmo;
mod scene;
mod ui;

use std::sync::Arc;
use std::time::SystemTime;

use dear_imgui_ash::{AshRenderer, AshRendererConfig, Options, TextureRetirementBatch};
use dear_imgui_rs::{ConfigFlags, Context};
use dear_imgui_winit::{HiDpiMode, WinitPlatform};
use hbd_engine::core::math::Vec3;
use hbd_engine::gfx::{self, VulkanContext};
use hbd_engine::platform::{self, PlatformHandler, WindowConfig};
use hbd_engine::render::{
    Camera, DirectionalLight, HbtrQuality, RenderResolution, Renderer, RendererConfig,
};
use scene::EditorScene;
use tracing::{error, info, warn};
use winit::event::WindowEvent;
use winit::window::Window;

struct Editor {
    renderer: Option<Renderer>,
    imgui: Context,
    platform: WinitPlatform,
    ash_renderer: Option<AshRenderer>,
    pending_retirement: Option<TextureRetirementBatch>,

    scene: EditorScene,
    camera: Camera,
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
        self.shader_mtimes = ui::WATCHED_SHADERS.iter().map(|_| None).collect();
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
                intensity: self.light_intensity,
                ..DirectionalLight::default()
            });
            if self.frame.is_multiple_of(120)
                && let Ok((visible, total, tris)) = renderer.cull_counts()
            {
                self.last_counts = (visible, total, tris);
                info!(visible, total, tris, "editor cull counts");
            }
        }

        if let Err(error) = self.platform.prepare_frame(&mut self.imgui, _window) {
            warn!(%error, "ImGui frame preparation failed");
            return;
        }

        ui::build_ui(self);

        let camera = self.camera;
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
    fn reload_shaders_if_changed(&mut self) {
        for (index, path) in ui::WATCHED_SHADERS.iter().enumerate() {
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

    // Renderer construction allocates large descriptor arrays on the stack.
    // The sandbox uses the same 16 MB worker thread; the default 1 MB main
    // thread stack overflows in debug builds.
    let runner = std::thread::Builder::new()
        .name("hbd-editor".to_owned())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
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
                scene: EditorScene::new(),
                camera: Camera::new(Vec3::new(9.0, 4.5, 0.0), Vec3::ZERO),
                light_intensity: 1.0,
                resolution: RenderResolution::Native,
                quality: HbtrQuality::Medium,
                shader_mtimes: vec![None; ui::WATCHED_SHADERS.len()],
                frame: 0,
                last_counts: (0, 0, 0),
                error_message: None,
            };

            if let Err(error) = platform::run(config, editor) {
                error!(%error, "editor event loop failed");
            }
        });
    match runner {
        Ok(handle) => {
            if let Err(error) = handle.join() {
                error!("editor worker thread panicked: {error:?}");
            }
        }
        Err(error) => error!(%error, "failed to spawn editor worker thread"),
    }
}
