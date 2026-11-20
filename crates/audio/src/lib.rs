//! Audio output and mixing.

#[derive(Debug, Clone, Copy)]
pub struct AudioConfig {
    pub master_volume: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self { master_volume: 0.8 }
    }
}

impl AudioConfig {
    /// Clamped master volume in the range `0.0..=1.0`.
    pub fn clamped_volume(&self) -> f32 {
        self.master_volume.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volume_is_clamped() {
        let config = AudioConfig { master_volume: 2.0 };
        assert_eq!(config.clamped_volume(), 1.0);
    }
}
