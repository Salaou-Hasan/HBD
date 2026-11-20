use std::time::{Duration, Instant};

/// Frame timing state.
///
/// `Time` is deliberately small and dependency-free. The runtime ticks it once
/// per frame; systems read the delta or elapsed time.
#[derive(Debug, Clone)]
pub struct Time {
    start: Instant,
    last: Instant,
    delta: Duration,
    elapsed: Duration,
    frame: u64,
}

impl Time {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last: now,
            delta: Duration::ZERO,
            elapsed: Duration::ZERO,
            frame: 0,
        }
    }

    /// Advances the clock and returns the frame delta.
    pub fn tick(&mut self) -> Duration {
        let now = Instant::now();
        let delta = now.saturating_duration_since(self.last);

        self.last = now;
        self.delta = delta;
        self.elapsed = self.elapsed.saturating_add(delta);
        self.frame = self.frame.wrapping_add(1);

        delta
    }

    pub fn delta(&self) -> Duration {
        self.delta
    }

    pub fn delta_secs(&self) -> f32 {
        self.delta.as_secs_f32()
    }

    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed.as_secs_f64()
    }

    pub fn frame(&self) -> u64 {
        self.frame
    }

    pub fn start(&self) -> Instant {
        self.start
    }
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_advances_the_frame_counter() {
        let mut time = Time::new();
        assert_eq!(time.frame(), 0);

        time.tick();

        assert_eq!(time.frame(), 1);
        assert!(time.elapsed() >= time.delta());
    }
}
