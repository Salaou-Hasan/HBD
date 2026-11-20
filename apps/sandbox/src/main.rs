use std::path::PathBuf;
use std::sync::Arc;

use hbd_engine::core::Time;
use hbd_engine::core::math::Vec3;
use hbd_engine::gfx;
use hbd_engine::gfx::VulkanContext;
use hbd_engine::platform::{self, PlatformHandler, WindowConfig};
use hbd_engine::render::{Camera, DebugView, Renderer};
use tracing::{error, info, warn};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

struct Sandbox {
    time: Time,
    renderer: Option<Renderer>,
    static_camera: bool,
    obj_path: Option<PathBuf>,
    profile: bool,
    profile_path: Option<PathBuf>,
    profile_interval: u32,
    no_cache: bool,
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        if let Some(renderer) = self.renderer.as_ref() {
            info!("sandbox shutting down: finalizing profiler explicitly");
            renderer.finish_profiling();
        }
    }
}

impl PlatformHandler for Sandbox {
    fn on_window_created(&mut self, window: &Arc<Window>) {
        let size = window.inner_size();
        info!(width = size.width, height = size.height, "window created");

        if let Err(error) = log_vulkan() {
            warn!("Vulkan probe failed: {error}");
        }

        let context = match VulkanContext::new(&**window, size.width, size.height) {
            Ok(context) => context,
            Err(error) => {
                error!("failed to initialize the Vulkan context: {error}");
                return;
            }
        };

        let renderer = match self.obj_path.as_deref() {
            Some(path) => Renderer::new_with_obj(context, path),
            None => Renderer::new(context),
        };
        match renderer {
            Ok(mut renderer) => {
                info!(
                    extent = ?renderer.swapchain_extent(),
                    format = ?renderer.swapchain_format(),
                    "renderer ready"
                );
                if self.profile {
                    if let Err(error) =
                        renderer.enable_profiling(self.profile_path.clone(), self.profile_interval)
                    {
                        error!(%error, "failed to enable E2E profiler");
                    } else {
                        info!("E2E profiler requested via --profile");
                    }
                } else {
                    info!("E2E profiler not requested; no GPU timestamps recorded");
                }
                renderer.set_pass_cache_enabled(!self.no_cache);
                self.renderer = Some(renderer);
            }
            Err(error) => {
                error!("failed to initialize the renderer: {error}");
            }
        }
    }

    fn on_window_event(&mut self, _window: &Arc<Window>, event: &WindowEvent) {
        match event {
            WindowEvent::Resized(size) => {
                info!(width = size.width, height = size.height, "window resized");

                let resize_result = self
                    .renderer
                    .as_mut()
                    .map(|renderer| renderer.resize(size.width, size.height));

                if let Some(Err(error)) = resize_result {
                    error!("failed to resize the swapchain: {error}");
                    self.renderer = None;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    let code = match event.physical_key {
                        PhysicalKey::Code(code) => code,
                        PhysicalKey::Unidentified(_) => return,
                    };
                    let view = match code {
                        KeyCode::Digit0 => Some(DebugView::Final),
                        KeyCode::Digit1 => Some(DebugView::Filmic),
                        KeyCode::Digit2 => Some(DebugView::Hdr),
                        KeyCode::Digit3 => Some(DebugView::Bloom),
                        KeyCode::Digit4 => Some(DebugView::AmbientOcclusion),
                        KeyCode::Digit5 => Some(DebugView::Depth),
                        KeyCode::Digit6 => Some(DebugView::ShadowCascade),
                        KeyCode::Digit7 => Some(DebugView::TemporalHistory),
                        _ => None,
                    };
                    if let Some(view) = view
                        && let Some(renderer) = self.renderer.as_ref()
                    {
                        if let Err(error) = renderer.set_debug_view(view) {
                            error!("failed to set debug view: {error}");
                        } else {
                            info!(?view, "debug view");
                        }
                    }
                }
            }
            WindowEvent::CloseRequested => info!("close requested"),
            _ => {}
        }
    }

    fn on_redraw(&mut self, _window: &Arc<Window>) {
        self.time.tick();

        let seconds = self.time.elapsed_secs() as f32;
        let clear_color = [
            0.015 + 0.005 * (seconds * 0.70).sin(),
            0.025 + 0.008 * (seconds * 1.10).sin(),
            0.055 + 0.015 * (seconds * 1.70).sin(),
            1.0,
        ];
        let angle = if self.static_camera {
            0.0
        } else {
            seconds * 0.25
        };
        let camera = Camera::new(
            Vec3::new(angle.cos() * 9.0, 4.5, angle.sin() * 9.0),
            Vec3::new(0.0, -0.2, 0.0),
        );

        let render_result = self
            .renderer
            .as_mut()
            .map(|renderer| renderer.render(&camera, clear_color));

        if let Some(Err(error)) = render_result {
            error!("frame presentation failed: {error}");
            self.renderer = None;
        }

        if self.time.frame().is_multiple_of(120) {
            let delta = self.time.delta_secs();
            if delta > 0.0 {
                info!(frame = self.time.frame(), fps = 1.0 / delta, "frame");
            }
        }
    }
}

fn log_vulkan() -> Result<(), gfx::GfxError> {
    let info = gfx::probe()?;

    info!(
        loader = %gfx::version_string(info.loader_version),
        "Vulkan loader ready"
    );

    for device in info.devices {
        info!(
            name = %device.name,
            api = %gfx::version_string(device.api_version),
            kind = ?device.device_type,
            "Vulkan device"
        );
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|arg| arg == "--probe") {
        log_vulkan()?;
        return Ok(());
    }

    let static_camera = args.iter().any(|arg| arg == "--static-camera");
    let obj_path = args
        .windows(2)
        .find(|pair| pair[0] == "--obj")
        .map(|pair| PathBuf::from(&pair[1]));
    let profile = args.iter().any(|arg| arg == "--profile");
    let profile_path = args
        .windows(2)
        .find(|pair| pair[0] == "--profile-report")
        .map(|pair| PathBuf::from(&pair[1]));
    let profile_interval = args
        .windows(2)
        .find(|pair| pair[0] == "--profile-interval")
        .and_then(|pair| pair[1].parse::<u32>().ok())
        .unwrap_or(120);
    let no_cache = args.iter().any(|arg| arg == "--no-cache");
    info!(
        profile,
        ?profile_path,
        profile_interval,
        no_cache,
        "sandbox profiling configuration"
    );

    let max_frames: Option<u64> = args
        .windows(2)
        .find(|pair| pair[0] == "--frames")
        .and_then(|pair| pair[1].parse().ok());

    let config = WindowConfig {
        title: "HBD - M1".to_owned(),
        max_frames,
        ..Default::default()
    };

    platform::run(
        config,
        Sandbox {
            time: Time::new(),
            renderer: None,
            static_camera,
            obj_path,
            profile,
            profile_path,
            profile_interval,
            no_cache,
        },
    )?;
    Ok(())
}
