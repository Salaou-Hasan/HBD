//! Public facade for the HBD engine.
//!
//! Downstream crates (games, tools, the future editor) depend on `hbd-engine`,
//! never on the internal crates directly. That keeps the public surface stable
//! while internals move.

pub use hbd_assets as assets;
pub use hbd_audio as audio;
pub use hbd_core as core;
pub use hbd_gfx as gfx;
pub use hbd_jobs as jobs;
pub use hbd_physics as physics;
pub use hbd_platform as platform;
pub use hbd_render as render;
pub use hbd_runtime as runtime;
pub use hbd_scene as scene;
pub use hbd_script as script;

pub mod ecs;

pub mod prelude {
    pub use hbd_core::math::*;
    pub use hbd_core::{Handle, Time};
}

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
