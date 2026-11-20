//! Physics.
//!
//! Jolt runs on a dedicated fixed-tick thread. Simulation and rendering are
//! decoupled through interpolated body transforms.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsBackend {
    Jolt,
}

#[derive(Debug, Clone, Copy)]
pub struct PhysicsConfig {
    pub backend: PhysicsBackend,
    pub fixed_hz: u32,
    pub max_steps_per_frame: u32,
    pub interpolation: bool,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            backend: PhysicsBackend::Jolt,
            fixed_hz: 60,
            max_steps_per_frame: 3,
            interpolation: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_fixed_tick() {
        let config = PhysicsConfig::default();
        assert_eq!(config.backend, PhysicsBackend::Jolt);
        assert_eq!(config.fixed_hz, 60);
        assert!(config.interpolation);
    }
}
