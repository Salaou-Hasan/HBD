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
| `--vsync` | Force FIFO presentation in the sandbox (display-locked A/B). Without it the sandbox opts into uncapped presentation. |

Enable `RUST_LOG=debug` for per-marker logs; `info` (the default) already includes every skip, cache decision, toggle, and summary. The sandbox logs `E2E profiler requested via --profile` or `E2E profiler not requested; no GPU timestamps recorded` at startup so the mode is never ambiguous.

## What is measured

GPU timestamps are written into the command buffer with `vkCmdWriteTimestamp` (`BOTTOM_OF_PIPE`) at eight markers: frame start, after shadow, after geometry, after TAA, after SSAO, after bloom, after tonemap, frame end. Consecutive markers give per-pass GPU milliseconds; the last minus the first gives total GPU frame time. Results are read back non-blocking with retry, so profiling does not stall the queue. CPU frame time covers the full `Renderer::render` call.

CSV columns: `frame,cpu_ms,pass_shadow,pass_geometry,pass_taa,pass_ssao,pass_bloom,pass_tonemap,pass_other,pass_frame`. A skipped pass shows `0.0000`, which makes cache hits visible in the data as well as the log.

Periodic `info!` summaries report p50/p95/p99 for total GPU, CPU, and each pass over the last 240 frames.

### Readback reliability (fixed in M2)

Timestamp readback uses an 8-slot ring, not 2. With 2 slots, a query was reused before the driver marked it available and `get_query_pool_results` almost always returned `NOT_READY`: a 1M-instance run resolved **1 of 20** frames, making p99 meaningless. With 8 slots the same class of run resolves **112 of 120** frames. Every resolution is still non-blocking; slots simply get 8 frames of slack.

`tools/profile-check.ps1` gates a CSV against pass p99 limits:

```powershell
powershell -ExecutionPolicy Bypass -File tools/profile-check.ps1 `
  -CsvPath .\profile.csv -MaxGpuP99Ms 9 -MaxShadowP99Ms 3 `
  -MaxGeometryP99Ms 2 -MaxSsaoP99Ms 2 -MaxBloomP99Ms 1 -MaxTonemapP99Ms 1
```

Those limits are the current Intel baseline plus margin. On the 82-cube orbit baseline (120 frames): GPU p50 **3.61 ms**, p99 **8.00 ms**; shadow p99 2.51, geometry 1.61, SSAO 1.66, bloom 0.50, tonemap 0.48 ms.

### Presentation mode: the 120 fps ceiling (measured)

`VulkanContext` exposes an explicit `PresentPolicy`: `Vsync` (engine default, tear-free FIFO) and `Uncapped` (MAILBOX when available, else IMMEDIATE). The chosen mode and the full supported-mode list are logged on every swapchain creation. The sandbox opts into `Uncapped` by default; pass `--vsync` for the display-locked comparison.

Release build, Intel, 1280x720, moving camera, 600 frames:

| Policy | FPS samples (every 120 frames) | GPU frame p50 | CPU render call p50 |
|---|---|---|---|
| `--vsync` (FIFO) | ~120 (display refresh cap) | 5.99 ms | 8.21 ms |
| uncapped (IMMEDIATE) | **422 / 466 / 426 / 461 / 492** | **1.87 ms** | **1.93 ms** |

The bottleneck was the presentation cap, not a render pass: with FIFO the CPU render call includes the wait for the previous frame's fence and present semaphore. Under IMMEDIATE the same passes cost shadow 0.17, geometry 0.39, TAA 0.50, SSAO 0.50, bloom 0.15, tonemap 0.16 ms (p50). MAILBOX is preferred when the driver exposes it: it is uncapped **and** tear-free. IMMEDIATE is only used with the explicit uncapped policy and may tear on displays without MAILBOX.

### Fullscreen is pixel-bound, not capped

Maximizing 1280x720 → 2560x1555 quadruples the fragment work. The present mode stays IMMEDIATE (verified in the log), and FPS settles at ~100–110 because the GPU frame is ~9.3 ms. Measured fullscreen p50 by pass: SSAO 2.20, TAA 2.06, geometry 1.74, bloom 0.75, tonemap 0.72, shadow 0.35 ms. The TAA resolve and its history copy are now a **single draw with two color attachments** (same output pixels, one less full-screen pass): TAA 2.46 → 2.06 ms, total 9.69 → 9.28 ms. Further gains at this resolution require either a faster GPU or an explicit render-scale/quality tradeoff; rendering at full swapchain resolution is deliberately kept lossless.

### 1M-instance stress scene

`--instances 1000000` dispatches a one-time compute pass that fills a device-local buffer with 1,000,000 distinct cubes in a paraboloid bowl (ground slab + 999,999 cubes), drawn with one `vkCmdDrawIndexedIndirect` = **12M triangles**. Intel dev GPU, 1280x720, moving camera, 20 frames: GPU frame **78.4 ms**, of which geometry **76.4 ms**; shadow 0.74, HBTR 0.50, SSAO 0.38, bloom 0.15, tonemap 0.14 ms. This is a correctness/scale demo, not a 60 fps result; the RTX 3060-class gate remains pending.

### 4M-instance verification (cluster + HZB + bindless + velocity)

`--instances 4000000 --frames 130 --cull-stats`, Intel at 1280x720, moving camera, frame 120: **1,357,643 / 4,000,000 visible instances = 16,291,716 visible triangles**, `target_10m_met=true`, ~11 fps. Visible counts move with the wall-clock camera path (an earlier run reported 1,048,055 / 12,576,660); both exceed the 10M target. Cluster/HZB are conservative: the 82-instance scene reports 28–31 visible with no over-culling, and the static-camera count (31) is unchanged after motion vectors and bindless materials were added.

### Motion-vector validation

There is no screenshot path in the sandbox, so motion-vector correctness is verified by an explicit stalled readback instead of a claimed visual result. `--animate` runs `animate.wgsl` over the compacted instances after cull (it writes poses, never the source buffer) and sets `previous_model` to the pose from the frame that wrote this image's history slot. The readback compares each compacted instance's current and previous model, so a changing compaction order cannot create a false positive:

| Run | moved | visible | max delta |
|---|---|---|---|
| `--animate` (moving camera) | 27 | 28 | 0.0434 |
| `--static-camera --animate` | 30 | 31 | 0.0434 |

The ground slab is excluded by scale, which is why `moved` is one less than `visible`. `--no-cull --animate` logs `object animation requires culling; staying static` rather than silently producing a different scene than the culled path. Static-camera and `--no-cache` runs exit 0 with no validation errors and no pass-count change.

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
3. **Cluster + HZB occlusion culling.** 256-instance clusters get one conservative sphere at init; a per-frame compute pass tests each sphere against the frustum and the previous frame's max-depth HZB before per-instance culling runs. HZB rejection is conservative (reject only when the cluster's max depth is behind the recorded max), so it cannot remove visible geometry.
4. **Bindless materials.** One `texture_2d_array` with four procedural layers; `material.w` picks the layer. No descriptor writes at draw time.
5. **Jitter-consistent motion vectors.** Velocity uses the same view-projection for both model poses (camera motion cancels and stays with HBTR's depth reprojection). HBTR stores the jittered pose per image, so history probing matches the exact projection that wrote that image's history. `--animate` provides real object motion and an explicit readback proof.

## Rejected experiment (kept out of the tree)

A version with 16 SSAO hemisphere offsets precomputed on the CPU into a storage buffer was implemented and measured. It removed 48 sin-based hashes per AO fragment but added storage fetches and tangent-space rotation. Steady-state GPU frame p50 went **3.642 → 3.941 ms** (SSAO 1.202 → 1.180 ms, geometry 1.311 → 1.530 ms). Because it was a net regression, it was reverted rather than shipped. This is recorded here so the decision is not silently forgotten.

## Caveats

- The development GPU is Intel integrated; the RTX 3060-class gate is still pending.
- The profiler adds eight timestamp writes and one query pool; it is off unless `--profile` is passed.
- `TemporalHistory` debug view shows the read slot captured at pass construction; it may not show the newest ping-pong slot.
- Valuable next targets from the data are geometry (1.3–1.5 ms) and SSAO (1.2–1.35 ms), which still dominate the moving-camera path. GPU-driven drawing and a redesigned temporal resolve are M2/M3 work.
