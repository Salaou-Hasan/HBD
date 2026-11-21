use std::path::PathBuf;
use std::sync::Arc;

use hbd_engine::core::Time;
use hbd_engine::core::math::Vec3;
use hbd_engine::ecs::EcsScene;
use hbd_engine::gfx;
use hbd_engine::gfx::VulkanContext;
use hbd_engine::platform::{self, PlatformHandler, WindowConfig};
use hbd_engine::render::{
    Camera, DebugView, HbtrQuality, RenderResolution, Renderer, RendererConfig,
};
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
    animate: bool,
    uncapped: bool,
    debug_view: Option<DebugView>,
    render_resolution: RenderResolution,
    hbtr_quality: HbtrQuality,
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

        let present_policy = if self.uncapped {
            gfx::PresentPolicy::Uncapped
        } else {
            gfx::PresentPolicy::Vsync
        };
        let context = match VulkanContext::new_with_present_policy(
            &**window,
            size.width,
            size.height,
            present_policy,
        ) {
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
                if self.animate {
                    renderer.set_object_animation(true);
                    info!("object animation requested via --animate");
                }
                if let Some(view) = self.debug_view
                    && let Err(error) = renderer.set_debug_view(view)
                {
                    error!(%error, ?view, "failed to set startup debug view");
                }
                renderer.set_hbtr_quality(self.hbtr_quality);
                info!(tier = self.hbtr_quality.level(), "HBTR work tier applied");
                if let Err(error) = renderer.set_render_resolution(self.render_resolution) {
                    error!(%error, "failed to set render resolution");
                } else {
                    info!(
                        resolution = self.render_resolution.label(),
                        internal = ?renderer.render_extent(),
                        window = ?renderer.swapchain_extent(),
                        "render resolution applied"
                    );
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
                    let resolution = match code {
                        KeyCode::F1 => Some(RenderResolution::Native),
                        KeyCode::F2 => Some(RenderResolution::P1080),
                        KeyCode::F3 => Some(RenderResolution::P1440),
                        KeyCode::F4 => Some(RenderResolution::P2K),
                        KeyCode::F5 => Some(RenderResolution::P4K),
                        _ => None,
                    };
                    if let Some(resolution) = resolution {
                        let result = self
                            .renderer
                            .as_mut()
                            .map(|renderer| renderer.set_render_resolution(resolution));
                        if let Some(Err(error)) = result {
                            error!(%error, "failed to change render resolution");
                        } else {
                            info!(resolution = resolution.label(), "render resolution hotkey");
                            self.render_resolution = resolution;
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
        // Push live entities into the shadow pass: the shadow draws
        // `frame.objects` in real time instead of the startup snapshot. An
        // empty scene returns the shadow to the mesh's culled instance set.
        if let Some(renderer) = self.renderer.as_mut()
            && let Err(error) = renderer.update_instances(&frame.instances())
        {
            error!(%error, "failed to hot-reload shadow instances from ECS objects");
        }
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
                    Ok((visible, total, tris_per_instance)) => {
                        let visible_triangles = u64::from(visible) * u64::from(tris_per_instance);
                        info!(
                            frame = self.time.frame(),
                            visible,
                            total,
                            visible_triangles,
                            target_10m_met = visible_triangles >= 10_000_000,
                            "GPU cull stats (explicit readback)"
                        );
                    }
                    Err(error) => error!(%error, "GPU cull stats readback failed"),
                }
            }
            if self.animate
                && let Some(renderer) = self.renderer.as_ref()
            {
                match renderer.visible_motion() {
                    Ok((moved, visible, max_delta)) => info!(
                        frame = self.time.frame(),
                        moved, visible, max_delta, "motion-vector poses (explicit readback)"
                    ),
                    Err(error) => error!(%error, "motion-vector pose readback failed"),
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
    let hbtr_quality = match args
        .windows(2)
        .find(|pair| pair[0] == "--quality")
        .map(|pair| pair[1].to_ascii_lowercase())
        .as_deref()
    {
        Some("low") => HbtrQuality::Low,
        Some("high") => HbtrQuality::High,
        Some("epic") => HbtrQuality::Epic,
        _ => HbtrQuality::Medium,
    };
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
    let animate = args.iter().any(|arg| arg == "--animate");
    // The sandbox is a performance validation app: run uncapped by default and
    // expose --vsync for the display-locked A/B comparison.
    let uncapped = !args.iter().any(|arg| arg == "--vsync");
    let debug_view =
        args.windows(2)
            .find(|pair| pair[0] == "--debug")
            .and_then(|pair| match pair[1].to_ascii_lowercase().as_str() {
                "final" => Some(DebugView::Final),
                "filmic" => Some(DebugView::Filmic),
                "hdr" => Some(DebugView::Hdr),
                "bloom" => Some(DebugView::Bloom),
                "ao" => Some(DebugView::AmbientOcclusion),
                "depth" => Some(DebugView::Depth),
                "shadow" => Some(DebugView::ShadowCascade),
                "taa" => Some(DebugView::TemporalHistory),
                _ => None,
            });
    // `--upscale` is the 50% shortcut; `--upscale-N` selects N percent of the
    // live window size (1..=100). Both are auto modes that follow resizes.
    let upscale_percent = args.iter().find_map(|arg| {
        arg.strip_prefix("--upscale-")
            .and_then(|value| value.parse::<u32>().ok())
            .map(|value| value.clamp(1, 100))
    });
    let upscale = args.iter().any(|arg| arg == "--upscale") || upscale_percent.is_some();
    let render_resolution = args
        .windows(2)
        .find(|pair| pair[0] == "--res")
        .and_then(|pair| {
            let value = pair[1].to_ascii_lowercase();
            if let Some((width, height)) = value.split_once('x')
                && let (Ok(width), Ok(height)) = (width.parse::<u32>(), height.parse::<u32>())
            {
                return Some(RenderResolution::Custom(width, height));
            }
            match value.as_str() {
                "native" => Some(RenderResolution::Native),
                "1080p" => Some(RenderResolution::P1080),
                "1440p" => Some(RenderResolution::P1440),
                "2k" => Some(RenderResolution::P2K),
                "4k" => Some(RenderResolution::P4K),
                _ => None,
            }
        })
        .unwrap_or_default();
    // --upscale keeps the internal render size at half the live window size,
    // including after maximize/fullscreen, with no manual resolution choice.
    let render_resolution = if upscale {
        RenderResolution::Auto(upscale_percent.unwrap_or(50))
    } else {
        render_resolution
    };
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
        animate,
        uncapped,
        resolution = render_resolution.label(),
        upscale,
        auto_percent = upscale_percent,
        hbtr_tier = hbtr_quality.level(),
        ?debug_view,
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

    // The GPU setup builds several large descriptor arrays; give it a bigger
    // stack than the 1 MB default so naga and Vulkan setup do not overflow.
    info!("starting renderer on a 16 MB worker stack");
    let runner = std::thread::Builder::new()
        .name("hbd-main".to_owned())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
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
                    animate,
                    uncapped,
                    debug_view,
                    render_resolution,
                    hbtr_quality,
                    instances,
                    fps_smoothed: 0.0,
                },
            )
        })
        .map_err(|error| format!("failed to spawn main thread: {error}"))?;
    runner
        .join()
        .map_err(|_| "hbd-main thread panicked".to_owned())??;
    Ok(())
}
