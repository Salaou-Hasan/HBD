//! Job system configuration.
//!
//! The runtime will use a work-stealing pool (initially `bevy_tasks`, which is
//! already used by `bevy_ecs`). Keeping the configuration here prevents the
//! engine from accidentally creating two competing thread pools.

/// Number of workers to use, leaving one core for the main thread.
pub fn default_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get().saturating_sub(1).max(1))
        .unwrap_or(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JobSystemConfig {
    pub workers: usize,
}

impl Default for JobSystemConfig {
    fn default() -> Self {
        Self {
            workers: default_worker_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_count_is_never_zero() {
        assert!(default_worker_count() >= 1);
        assert!(JobSystemConfig::default().workers >= 1);
    }
}
