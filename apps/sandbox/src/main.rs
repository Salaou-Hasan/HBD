use std::path::PathBuf;
use std::sync::Arc;

use hbd_engine::core::Time;
use hbd_engine::core::math::Vec3;
use hbd_engine::ecs::EcsScene;
use hbd_engine::gfx;
use hbd_engine::gfx::VulkanContext;
use hbd_engine::platform::{self, PlatformHandler, WindowConfig};
use hbd_engine::render::{Camera, DebugView, Renderer, RendererConfig};
use hbd_engine::scene::{Camera as EcsCamera, Entity, Light as EcsLight};
use tracing::{error, info, warn};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

struct Sandbox {
    time: Time,
    renderer: Option<Renderer>,
    ecs_scene: EcsScene,
    camera_entity: Entity,
    static_camera: bool,
    obj_path: Option<PathBuf>,
    profile: bool,
    profile_path: Option<PathBuf>,
    profile_interval: u32,
    no_cache: bool,
    no_cull: bool,
    cull_stats: bool,
    instances: Option<u32>,
    fps_smoothed: f32,
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

        let renderer = if let Some(path) = self.obj_path.as_deref() {
            Renderer::new_with_obj(context, path)
        } else if let Some(count) = self.instances {
            Renderer::new_with_instance_count(context, count)
        } else {
            // Consume the public config: hdr=false presents saturated LDR color.
            Renderer::new_with_config(context, RendererConfig::default())
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
                if self.no_cull {
                    renderer.set_culling_enabled(false);
                    info!("GPU frustum culling disabled via --no-cull");
                } else {
                    info!("GPU frustum culling enabled (default)");
                }
                if let Some(count) = self.instances
                    && let Err(error) = renderer.set_instance_count(count)
                {
                    error!(%error, count, "failed to set instance count");
                }
                let frame = self.ecs_scene.run();
                if let Some(light) = frame.light {
                    renderer.set_light(light);
                } else {
                    info!("no ECS light entity; keeping the renderer's default directional light");
                }
                info!(
                    camera = frame.camera.is_some(),
                    objects = frame.objects.len(),
                    "ECS scene extracted at renderer creation"
                );
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
                        && let Some(renderer) = self.renderer.as_mut()
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

        let delta = self.time.delta_secs();
        if delta > 0.0 {
            let instant_fps = 1.0 / delta;
            self.fps_smoothed = if self.fps_smoothed == 0.0 {
                instant_fps
            } else {
                self.fps_smoothed * 0.9 + instant_fps * 0.1
            };
        }

        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_fps(self.fps_smoothed);
        }

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
        let camera_position = Vec3::new(angle.cos() * 9.0, 4.5, angle.sin() * 9.0);
        let camera_target = Vec3::new(0.0, -0.2, 0.0);
        if let Some(mut camera) = self
            .ecs_scene
            .world()
            .get_mut::<EcsCamera>(self.camera_entity)
        {
            camera.position = camera_position;
            camera.target = camera_target;
        } else {
            error!(entity = ?self.camera_entity, "ECS camera entity missing");
        }

        // Explicit ECS schedule run: the camera and light consumed by the
        // renderer this frame come from entities, not from local temporaries.
        let frame = self.ecs_scene.run();
        let camera = frame.camera.unwrap_or_else(|| {
            warn!("ECS scene produced no camera; using the renderer default");
            Camera::default()
        });

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
            if self.cull_stats
                && let Some(renderer) = self.renderer.as_ref()
            {
                match renderer.cull_counts() {
                    Ok((visible, total)) => info!(
                        frame = self.time.frame(),
                        visible, total, "GPU cull stats (explicit readback)"
                    ),
                    Err(error) => error!(%error, "GPU cull stats readback failed"),
                }
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
    let no_cull = args.iter().any(|arg| arg == "--no-cull");
    let cull_stats = args.iter().any(|arg| arg == "--cull-stats");
    let instances = args
        .windows(2)
        .find(|pair| pair[0] == "--instances")
        .and_then(|pair| pair[1].parse::<u32>().ok());
    info!(
        profile,
        ?profile_path,
        profile_interval,
        no_cache,
        no_cull,
        cull_stats,
        instances,
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

    let mut ecs_scene = EcsScene::new();
    let camera_entity = ecs_scene
        .world()
        .spawn((EcsCamera {
            position: Vec3::new(9.0, 4.5, 0.0),
            target: Vec3::new(0.0, -0.2, 0.0),
            ..Default::default()
        },))
        .id();
    ecs_scene.world().spawn((EcsLight::default(),));
    info!(?camera_entity, "ECS camera and light entities spawned");

    platform::run(
        config,
        Sandbox {
            time: Time::new(),
            renderer: None,
            ecs_scene,
            camera_entity,
            static_camera,
            obj_path,
            profile,
            profile_path,
            profile_interval,
            no_cache,
            no_cull,
            cull_stats,
            instances,
            fps_smoothed: 0.0,
        },
    )?;
    Ok(())
}
