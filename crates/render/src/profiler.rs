//! End-to-end profiler for the M1 renderer.
//!
//! Every operation is explicit and logged: enabling, GPU timestamp queries,
//! CPU frame timing, cache/skip decisions, periodic percentile reports, and
//! final report writing. Nothing is sampled or written silently.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

use ash::vk;
use tracing::{debug, info, warn};

use hbd_gfx::VulkanContext;

use crate::RenderError;

/// Order matches the timestamp markers written into each command buffer.
pub(crate) const PASS_NAMES: [&str; 8] = [
    "shadow", "geometry", "hbtr", "ssao", "bloom", "tonemap", "other", "frame",
];
const MARKERS: usize = PASS_NAMES.len();
pub(crate) const FRAME_SLOTS: usize = 8;
const HISTORY: usize = 240;

pub(crate) struct Profiler {
    device: ash::Device,
    query_pool: Option<vk::QueryPool>,
    period_ns: f32,
    pending: [bool; FRAME_SLOTS],
    gpu_samples: VecDeque<[f32; MARKERS]>,
    cpu_samples: VecDeque<f32>,
    report_path: Option<PathBuf>,
    report_interval: u32,
    since_report: u32,
    total_frames: u64,
    started: Instant,
}

impl Profiler {
    pub(crate) fn new(
        context: &VulkanContext,
        report_path: Option<PathBuf>,
        report_interval: u32,
    ) -> Result<Self, RenderError> {
        let device = context.device().clone();
        let query_pool = if context.supports_timestamps() {
            let create_info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count((MARKERS * FRAME_SLOTS) as u32);
            let pool = unsafe { device.create_query_pool(&create_info, None) }
                .map_err(|error| RenderError::Gfx(hbd_gfx::GfxError::Vulkan(error)))?;
            info!(
                queries = MARKERS * FRAME_SLOTS,
                period_ns = context.timestamp_period(),
                report_interval,
                report_path = ?report_path,
                "E2E profiler enabled with GPU timestamp queries"
            );
            Some(pool)
        } else {
            warn!(
                report_interval,
                report_path = ?report_path,
                "E2E profiler enabled in CPU-only mode: GPU timestamp queries unsupported"
            );
            None
        };

        Ok(Self {
            device,
            query_pool,
            period_ns: context.timestamp_period(),
            pending: [false; FRAME_SLOTS],
            gpu_samples: VecDeque::with_capacity(HISTORY),
            cpu_samples: VecDeque::with_capacity(HISTORY),
            report_path,
            report_interval: report_interval.max(1),
            since_report: 0,
            total_frames: 0,
            started: Instant::now(),
        })
    }

    /// Resets this frame slot's queries and writes the frame-start marker.
    /// Call at the very start of command recording, outside a render pass.
    pub(crate) fn begin_frame(
        &mut self,
        command_buffer: vk::CommandBuffer,
        slot: usize,
        frame: u64,
    ) {
        let Some(pool) = self.query_pool else {
            return;
        };
        unsafe {
            self.device.cmd_reset_query_pool(
                command_buffer,
                pool,
                (slot * MARKERS) as u32,
                MARKERS as u32,
            );
            self.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                pool,
                (slot * MARKERS) as u32,
            );
        }
        self.pending[slot] = true;
        debug!(frame, slot, "profiler: frame timestamp range armed");
    }

    /// Writes a marker after pass `marker` (1-based order in `PASS_NAMES`).
    pub(crate) fn mark(
        &mut self,
        command_buffer: vk::CommandBuffer,
        slot: usize,
        marker: usize,
        frame: u64,
    ) {
        let Some(pool) = self.query_pool else {
            return;
        };
        unsafe {
            self.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                pool,
                (slot * MARKERS + marker) as u32,
            );
        }
        debug!(
            frame,
            slot,
            pass = PASS_NAMES[marker - 1],
            "profiler: pass end marked"
        );
    }

    /// Non-blocking readback of a previously armed slot. Skips if not ready.
    ///
    /// A slot is reused only every `FRAME_SLOTS` frames, so a readback 8 frames
    /// later has time to complete even when the GPU is saturated. Two slots
    /// were not enough: at 1M instances only 1 of 20 frames resolved.
    pub(crate) fn resolve_slot(&mut self, slot: usize, frame: u64) {
        let (Some(pool), true) = (self.query_pool, self.pending[slot]) else {
            return;
        };
        let mut raw = [0u64; MARKERS];
        let result = unsafe {
            self.device.get_query_pool_results(
                pool,
                (slot * MARKERS) as u32,
                &mut raw,
                vk::QueryResultFlags::empty(),
            )
        };
        match result {
            Ok(()) => {
                self.pending[slot] = false;
                let ticks: Vec<f64> = raw.iter().map(|value| *value as f64).collect();
                let mut sample = [0.0f32; MARKERS];
                for (index, name) in PASS_NAMES.iter().enumerate() {
                    let ms = if *name == "frame" {
                        ((ticks[MARKERS - 1] - ticks[0]) * self.period_ns as f64) / 1.0e6
                    } else if *name == "other" {
                        let total = ticks[MARKERS - 1] - ticks[0];
                        let measured: f64 = (1..MARKERS - 1).map(|i| ticks[i] - ticks[i - 1]).sum();
                        ((total - measured) * self.period_ns as f64) / 1.0e6
                    } else {
                        ((ticks[index + 1] - ticks[index]) * self.period_ns as f64) / 1.0e6
                    };
                    sample[index] = ms.max(0.0) as f32;
                }
                debug!(frame, slot, ?sample, "profiler: GPU timestamps resolved");
                self.gpu_samples.push_back(sample);
                while self.gpu_samples.len() > HISTORY {
                    self.gpu_samples.pop_front();
                }
            }
            Err(vk::Result::NOT_READY) => {
                debug!(frame, slot, "profiler: timestamps not ready, will retry");
            }
            Err(error) => {
                warn!(
                    frame,
                    slot,
                    ?error,
                    "profiler: failed to read GPU timestamps"
                );
                self.pending[slot] = false;
            }
        }
    }

    pub(crate) fn record_cpu_frame(&mut self, milliseconds: f32) {
        self.cpu_samples.push_back(milliseconds);
        while self.cpu_samples.len() > HISTORY {
            self.cpu_samples.pop_front();
        }
        self.total_frames += 1;
        self.since_report += 1;
        if self.since_report >= self.report_interval {
            self.since_report = 0;
            self.report();
        }
    }

    fn percentile(values: &mut [f32], fraction: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        values.sort_by(|a, b| a.total_cmp(b));
        let index = ((values.len() - 1) as f32 * fraction).round() as usize;
        values[index]
    }

    fn report(&self) {
        let mut frame: Vec<f32> = self
            .gpu_samples
            .iter()
            .map(|sample| sample[MARKERS - 1])
            .collect();
        let cpu: Vec<f32> = self.cpu_samples.clone().into();
        info!(
            frames = self.gpu_samples.len(),
            gpu_p50_ms = Self::percentile(&mut frame, 0.50),
            gpu_p95_ms = Self::percentile(&mut frame, 0.95),
            gpu_p99_ms = Self::percentile(&mut frame, 0.99),
            cpu_p50_ms = Self::percentile(&mut cpu.clone(), 0.50),
            cpu_p99_ms = Self::percentile(&mut cpu.clone(), 0.99),
            "E2E profile summary"
        );
        for (index, name) in PASS_NAMES.iter().enumerate() {
            if *name == "frame" || *name == "other" {
                continue;
            }
            let mut values: Vec<f32> = self
                .gpu_samples
                .iter()
                .map(|sample| sample[index])
                .collect();
            info!(
                pass = name,
                p50_ms = Self::percentile(&mut values, 0.50),
                p99_ms = Self::percentile(&mut values, 0.99),
                "E2E pass timing"
            );
        }
    }

    /// Explicitly writes the CSV report. Called by the app on shutdown.
    pub(crate) fn finish(&self) {
        let elapsed = self.started.elapsed().as_secs_f64();
        info!(
            frames = self.total_frames,
            elapsed_secs = elapsed,
            gpu_samples = self.gpu_samples.len(),
            cpu_samples = self.cpu_samples.len(),
            "E2E profiler finishing"
        );
        let Some(path) = &self.report_path else {
            info!("no profile report path configured; CSV not written");
            return;
        };
        let mut csv = String::from("frame,cpu_ms,");
        csv.push_str(
            &PASS_NAMES
                .iter()
                .map(|name| format!("pass_{name}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        csv.push('\n');
        let count = self.gpu_samples.len().min(self.cpu_samples.len());
        for index in 0..count {
            csv.push_str(&format!("{},{:.4},", index, self.cpu_samples[index]));
            csv.push_str(
                &self.gpu_samples[index]
                    .iter()
                    .map(|value| format!("{value:.4}"))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            csv.push('\n');
        }
        match std::fs::write(path, csv) {
            Ok(()) => info!(path = %path.display(), rows = count, "E2E profile CSV written"),
            Err(error) => warn!(path = %path.display(), %error, "E2E profile CSV write failed"),
        }
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        if let Some(pool) = self.query_pool {
            info!("destroying GPU timestamp query pool (profiler dropped)");
            unsafe { self.device.destroy_query_pool(pool, None) };
            self.query_pool = None;
        }
    }
}
