# End-to-End Profiling (M2)

## The strict rule

**No operation is hidden.** Every profiler enable/disable, GPU query, pass record or skip, uniform upload or skip, cache invalidation, cache hit, toggle, and report write is logged through `tracing` or recorded in the CSV. There are no silent fallbacks: if GPU timestamps are unsupported, the engine logs a warning and says it is in CPU-only mode; if a report path is absent, it logs that no CSV was written.

Every optimization has an explicit A/B switch (`--no-cache` for the pass cache). If you cannot see what the engine did, that is a bug.

## Run a profile

```powershell
cargo build -p hbd-sandbox
.\target\debug\hbd-sandbox.exe --frames 900 --profile --profile-report .\profile.csv --profile-interval 240
```

Flags:

| Flag | Meaning |
|---|---|
| `--profile` | Enable GPU timestamp queries and CPU frame timing. Without it, nothing is instrumented and the log says so. |
| `--profile-report <path>` | Write a CSV when the app shuts down. The write and row count are logged. |
| `--profile-interval <n>` | Frames between percentile summaries (default 120). |
| `--no-cache` | Force the full offscreen pipeline every frame; used as the A/B control. |

Enable `RUST_LOG=debug` for per-marker logs; `info` (the default) already includes every skip, cache decision, toggle, and summary. The sandbox logs `E2E profiler requested via --profile` or `E2E profiler not requested; no GPU timestamps recorded` at startup so the mode is never ambiguous.

## What is measured

GPU timestamps are written into the command buffer with `vkCmdWriteTimestamp` (`BOTTOM_OF_PIPE`) at eight markers: frame start, after shadow, after geometry, after TAA, after SSAO, after bloom, after tonemap, frame end. Consecutive markers give per-pass GPU milliseconds; the last minus the first gives total GPU frame time. Results are read back non-blocking with retry, so profiling does not stall the queue. CPU frame time covers the full `Renderer::render` call.

CSV columns: `frame,cpu_ms,pass_shadow,pass_geometry,pass_taa,pass_ssao,pass_bloom,pass_tonemap,pass_other,pass_frame`. A skipped pass shows `0.0000`, which makes cache hits visible in the data as well as the log.

Periodic `info!` summaries report p50/p95/p99 for total GPU, CPU, and each pass over the last 240 frames.

## Measured results — Intel UHD / 1280x720 / 900 frames / 120 Hz present

Steady-state window (last 200 frames), GPU milliseconds:

| Configuration | total p50 | total p99 | shadow | geometry | SSAO | bloom | tonemap |
|---|---|---|---|---|---|---|---|
| Baseline (before M2) | 3.642 | 6.134 | 0.282 | 1.311 | 1.202 | 0.433 | 0.377 |
| Orbit, cache enabled | 4.093 | 6.167 | 0.318 | 1.503 | 1.354 | 0.495 | 0.427 |
| Static camera, cache **disabled** (`--no-cache`) | 4.074 | 6.080 | 0.321 | 1.496 | 1.339 | 0.498 | 0.428 |
| Static camera, cache **enabled** | **0.645** | **0.655** | 0 | 0 | 0 | 0 | 0.644 |

The static-camera cache reduces GPU frame work by **6.3x** with no visible change: before/after captures at the same camera pose differ by a mean of 0.0006/255 per channel (a handful of pixels differ at most). In motion the cache never hits and GPU time is unchanged — no regression.

Reported presentation FPS stays near 120 because the swapchain uses FIFO/lowest-latency present and the display is 120 Hz. The gain is GPU headroom (4.07 → 0.645 ms), not a higher vsync-capped FPS counter. On the RTX 3060-class target the same headroom is what matters for the 1440p/60 gate.

## Optimizations applied

1. **Offscreen pass cache (enabled by default).** When the camera view-projection matrix is bit-identical to the pose that produced a given swapchain image's HDR/AO/bloom/shadow data, those four passes are skipped for that image. Correctness is per-image: each swapchain image carries its own validity flag, invalidated on any camera change, cache toggle, or resize. Uniform uploads are skipped only under the same condition. Every hit prints `offscreen cache HIT`, every invalidation prints `camera changed or cache disabled`, and a 240-frame `offscreen pass cache accounting` line reports full vs cached frames. `--no-cache` disables it.
2. **Resize dedupe.** `Renderer::resize` compares the swapchain extent before and after `VulkanContext::resize`. If it is unchanged, no passes are rebuilt; the event is logged as `resize ignored: swapchain extent unchanged (no pass rebuild)`. This removes redundant WGSL recompilation during winit's DPI-settle resize events.

## Rejected experiment (kept out of the tree)

A version with 16 SSAO hemisphere offsets precomputed on the CPU into a storage buffer was implemented and measured. It removed 48 sin-based hashes per AO fragment but added storage fetches and tangent-space rotation. Steady-state GPU frame p50 went **3.642 → 3.941 ms** (SSAO 1.202 → 1.180 ms, geometry 1.311 → 1.530 ms). Because it was a net regression, it was reverted rather than shipped. This is recorded here so the decision is not silently forgotten.

## Caveats

- The development GPU is Intel integrated; the RTX 3060-class gate is still pending.
- The profiler adds eight timestamp writes and one query pool; it is off unless `--profile` is passed.
- `TemporalHistory` debug view is stale while TAA is disabled; this is unrelated to profiling.
- Valuable next targets from the data are geometry (1.3–1.5 ms) and SSAO (1.2–1.35 ms), which still dominate the moving-camera path. GPU-driven drawing and a redesigned temporal resolve are M2/M3 work.
