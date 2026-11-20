//! Application runtime.
//!
//! The app loop, schedules, and plugin system land in M1. M0 fixes the runtime
//! configuration so platform, physics, and rendering agree on the tick rate.

pub const DEFAULT_FIXED_HZ: u32 = 60;

#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    pub target_fps: Option<u32>,
    pub fixed_hz: u32,
    pub headless: bool,
    pub max_frames: Option<u64>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            target_fps: None,
            fixed_hz: DEFAULT_FIXED_HZ,
            headless: false,
            max_frames: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_runtime_is_windowing_and_fixed_tick() {
        let config = RuntimeConfig::default();
        assert!(!config.headless);
        assert_eq!(config.fixed_hz, 60);
    }
}
