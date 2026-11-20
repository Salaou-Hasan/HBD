//! Windowing and OS integration.
//!
//! The platform layer owns the native event loop and the window. It exposes a
//! deliberately small handler trait so the runtime and renderer do not depend
//! on `winit` types beyond the boundary.

use std::sync::Arc;

use thiserror::Error;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::{Window, WindowId};

#[derive(Debug, Error)]
pub enum PlatformError {
    #[error("failed to create the event loop: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
}

#[derive(Debug, Clone)]
pub struct WindowConfig {
    pub title: String,
    /// Client-area width in physical pixels.
    pub width: u32,
    /// Client-area height in physical pixels.
    pub height: u32,
    pub resizable: bool,
    pub continuous_redraw: bool,
    /// Development/testing aid: exit after this many redraws.
    pub max_frames: Option<u64>,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "HBD".to_owned(),
            width: 1280,
            height: 720,
            resizable: true,
            continuous_redraw: true,
            max_frames: None,
        }
    }
}

/// Receives window lifecycle and redraw events.
///
/// The handler runs on the event-loop thread. GPU work is submitted from here
/// for now; a dedicated render thread will take over in a later milestone.
pub trait PlatformHandler {
    fn on_window_created(&mut self, window: &Arc<Window>);
    fn on_window_event(&mut self, window: &Arc<Window>, event: &WindowEvent);
    fn on_redraw(&mut self, window: &Arc<Window>);
}

pub fn run<H>(config: WindowConfig, handler: H) -> Result<(), PlatformError>
where
    H: PlatformHandler + 'static,
{
    // The sandbox runs the loop on a large-stack worker thread so the GPU
    // setup's descriptor arrays cannot overflow the 1 MB default stack.
    let event_loop = EventLoop::builder().with_any_thread(true).build()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = PlatformApp {
        config,
        handler,
        window: None,
        frames: 0,
    };

    event_loop.run_app(&mut app)?;
    Ok(())
}

struct PlatformApp<H> {
    config: WindowConfig,
    handler: H,
    window: Option<Arc<Window>>,
    frames: u64,
}

impl<H: PlatformHandler> ApplicationHandler for PlatformApp<H> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title(self.config.title.clone())
            .with_inner_size(PhysicalSize::new(self.config.width, self.config.height))
            .with_resizable(self.config.resizable);

        match event_loop.create_window(attributes) {
            Ok(window) => {
                let window = Arc::new(window);
                self.window = Some(Arc::clone(&window));
                self.handler.on_window_created(&window);
                window.request_redraw();
            }
            Err(error) => {
                tracing::error!("failed to create window: {error}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.clone() else {
            return;
        };

        self.handler.on_window_event(&window, &event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                self.frames += 1;
                self.handler.on_redraw(&window);

                if self
                    .config
                    .max_frames
                    .is_some_and(|max_frames| self.frames >= max_frames)
                {
                    event_loop.exit();
                } else if self.config.continuous_redraw {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}
