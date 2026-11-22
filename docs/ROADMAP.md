# HBD Roadmap

This is a living plan. A milestone is done only when its exit criteria are met on the stated hardware — not when the code merely compiles.

## Guiding rules

1. **Rendering baseline before breadth.** A correct, stable lit frame precedes tooling, content, and scale.
2. **Engine library first.** The editor is a consumer of `hbd-engine`; the engine never depends on it.
3. **Measured, not assumed.** Performance claims require numbers from the target GPU class.
4. **No silent default changes.** Rendering paths that are experimental (path tracing, HBTR until proven) stay off by default and are covered by tests.
5. **Every visual feature has a debug view.** If a buffer cannot be inspected, it cannot be trusted.

---

## M0 — Stack and configuration (done)

**Goal:** lock the foundation before writing renderer code.

**Delivered**

- Workspace of 13 crates plus the sandbox, Rust 2024 edition, resolver 3.
- Windows-only target, Vulkan 1.3 baseline via `ash` 0.38 + `gpu-allocator` 0.28.
- `winit` 0.30 windowing, `bevy_ecs` 0.20 ECS, `glam` math, `naga` 30 WGSL→SPIR-V.
- Jolt/Lua/Rust config types with unit tests for their shipping defaults.
- ADR 0001 recording the stack decisions.

**Exit criteria:** workspace builds and tests; accepted stack documented. ✅

---

## M1 — Rendering baseline (code complete, formal gate pending)

**Goal:** the first lit frame uses the full rendering grammar, not a gray triangle.

**Delivered**

- Vulkan context: device/queue selection, swapchain, FIFO present, per-image fences/semaphores, resize.
- HDR targets (`R16G16B16A16_SFLOAT`) and depth, one pair per swapchain image.
- Instanced cube grid (82 instances) with a six-color palette, PBR, procedural sky and analytic IBL.
- 3-cascade directional shadows (1024², PCF), half-res SSAO, half-res bloom.
- AgX tonemap, vignette, sRGB swapchain output.
- Perspective camera with sub-pixel jitter support; debug views `0`–`7`.
- Sandbox flags: `--probe`, `--static-camera`, `--frames N`.

**Deliberately not in the shipped M1 path**

- **HBD Temporal Resolve (HBTR)** is the default AA path (YCoCg variance clipping, confidence blend, free sharpening), replacing the old disabled HBTR. See [ANTI-ALIASING.md](ANTI-ALIASING.md) and [ADR 0005](adr/0005-hbtr.md).
- Path tracing remains research-only and off by default, enforced by a unit test.

**Exit criteria**

| Criterion | Status |
|---|---|
| Full pipeline visible in a single frame | ✅ |
| Debug view for every major buffer | ✅ |
| No shadow acne / peter-panning at 1440p | ⏳ needs target GPU |
| 0.18 gray card tonemaps to the expected value | ⏳ needs reference render |
| 1440p / 60 fps on RTX 3060-class | ⏳ needs target GPU |
| No frame over 16.6 ms on a 60 s camera path | ⏳ needs target GPU |

### M2 work started — E2E profiler and first optimizations

Delivered so far: GPU timestamp profiler with CSV export and percentile reports; explicit pass cache with per-image validity and `--no-cache` A/B; resize dedupe. Measured on the Intel dev GPU: static-camera GPU frame p50 **4.074 ms → 0.645 ms** (6.3x) with pixel-identical output; moving camera unchanged (~4.1 ms) and no regression. A precomputed-SSAO-sample experiment was measured as a regression and reverted. Full numbers and the no-hidden-operations rule are in [PROFILING.md](PROFILING.md) and [ADR 0004](adr/0004-visible-operations.md).

Remaining M2 work: the RTX 3060-class 1440p/60 gate (needs the target GPU) and the dynamic-object ghosting capture list. Bindless materials, cluster/HZB occlusion culling, and motion vectors are delivered below. The p99 regression gate now exists (`tools/profile-check.ps1`) and the profiler resolves 112/120 frames instead of 1/20.

**GPU frustum culling (per-image, safe A/B):** `cull.wgsl` tests each instance against six frustum planes and compacts survivors with `atomicAdd` into the indirect `instanceCount`. Both the compacted buffer and the indirect args are **per swapchain image**, with per-image cull and graphics descriptor sets — with 2 frames in flight an image is only reused after 3 frames, so it is guaranteed idle. The count is reset in the command buffer with `vkCmdFillBuffer` (TRANSFER → COMPUTE barrier) and published to the draw with a COMPUTE → VERTEX_SHADER|DRAW_INDIRECT barrier. `--no-cull` uses a separate graphics set and CPU-written indirect buffer bound to the *source* instances, so it can never read an unwritten compacted buffer. `--cull-stats` reports visible instances, visible triangles, and whether the 10M target is met via a one-off stalled readback.

**10M+ visible triangles — MET:** with `--instances 4000000` and a moving camera (geometry redrawn every frame, no pass-cache hits), frame 120 reports **1,357,643 / 4,000,000 visible instances = 16,291,716 visible triangles** (`target_10m_met=true`) on Intel at 1280x720, ~11 fps. Counts vary with the wall-clock camera path; the target has been met on every recorded run (earliest: 1,048,055 instances / 12,576,660 triangles).

**Cluster + HZB occlusion culling (delivered):** instances are grouped into `CLUSTER_SIZE = 256` clusters. `cluster.wgsl::init_cs` builds one conservative bounding sphere per cluster once; `cluster_cull_cs` tests each sphere against the six frustum planes and the previous frame's max-depth HZB (`HZB_MIPS = 3`, half resolution), and writes per-image visibility flags. `cull.wgsl` early-returns on a zero flag before its per-instance test. The HZB is rebuilt from the *previous* frame's depth with a conservative max reduction, so it can only reject geometry that was definitely behind last frame's occluders. Verified with `--cull-stats`: no over-culling; 82-instance scene reports 28–31 visible depending on pose, 4M scene still exceeds 10M triangles.

**Bindless material table (delivered):** the mesh pipeline samples a 4-layer `texture_2d_array` (`material_textures` / `material_sampler`, bindings 6/7) generated procedurally at startup. `Instance.material.w` selects the layer, so material changes never touch descriptors and there are no per-draw descriptor updates. `Instance` is 160 bytes: `model(64) + previous_model(64) + base_color(16) + material(16)`.

**Motion vectors (delivered, 2026-11-21):** a second color attachment `R16G16B16A16_SFLOAT` carries UV-space object motion. The vertex shader emits NDC for the current and previous model under the *same* view-projection, so camera motion cancels and is owned by HBTR's depth reprojection instead of being double-counted; NDC is `@interpolate(linear)` because it is linear in screen space. HBTR subtracts the velocity from the reprojected history UV and now stores the **jittered** pose per image, so history samples land exactly where the previous render of that image wrote them. `--animate` spins a bounded prefix of the compacted instances (`animate.wgsl`, ground slab filtered by scale) and sets `previous_model` to the pose from the frame that produced this image's history slot. Explicit readback verifies the pass actually wrote motion: `moved=27 visible=28 max_delta=0.0434` (moving camera) and `moved=30 visible=31` (static camera). `--no-cull --animate` logs that animation requires culling and stays static instead of silently differing.

**Presentation path:** `RendererConfig::hdr` is consumed via `Renderer::new_with_config`. HDR/AgX is the active default again (restored on user request); setting `hdr: false` selects the clamped-LDR presentation path that preserves the saturated palette. Both paths are compiled and explicitly logged at startup.

**ECS scene bridge (added 2026-11-20):** `hbd-scene` owns `Light`, `Camera`, and `Object` ECS components; `hbd-engine::ecs` runs an explicit schedule that extracts them into a `SceneFrame`. The sandbox spawns camera/light entities and drives every frame from `EcsScene::run()` — the camera and directional light consumed by the renderer are entity data, not local temporaries. Objects are extracted but not yet fed into the GPU instance buffer.

**Instance-count stress mode:** `--instances N` dispatches a one-time compute pass that fills a device-local instance buffer with a paraboloid **bowl of N distinct cubes** (index 0 is the ground slab). The draw is still a single `vkCmdDrawIndexedIndirect`. This is an explicit GPU-throughput stress scene, not a physics simulation or the asset pipeline; it is logged as `GPU bowl generator` at startup. Shadow cascades still use the 82-cube M1 source grid, so stress-scene shadows are intentionally not matched. Verified at 1k, 100k, and 1M instances (1M = 12M triangles, 78.4 ms GPU on Intel at 720p).

**Sky uniform layout fix:** `sky.wgsl` declared the pre-CSM `Frame` layout (three matrices), so `eye_position`/`light_direction` were read from the wrong offsets. The struct now matches `FrameUniform` member-for-member and the sky fragment writes an explicit zero velocity instead of leaving the second attachment undefined.

---

## M2 — Performance and GPU-driven rendering

**Goal:** hit the scale envelope without sacrificing the M1 look.

**Status: feature-complete on the engine side; final validation still pending.** Every work item below is implemented and measured on the Intel dev GPU. The remaining exit criteria require the RTX 3060-class target GPU (1440p/60 with the scale targets, no p99 regressions under that load, no shadow acne) and a captured no-ghosting camera-pan comparison.

**Work items**

- Replace per-instance CPU draws with indirect/multi-draw; target **1M GPU-driven draw items**. ✅ single `vkCmdDrawIndexedIndirect`, GPU bowl generator
- Target **10M+ visible triangles** with cluster/frustum culling on the GPU. ✅ 16.3M measured, `target_10m_met=true`
- Bindless descriptor sets and material/texture caching; eliminate per-draw descriptor updates. ✅ 4-layer bindless material array
- Frame-time profiling (p50/p99), GPU timestamps, and a regression dashboard. ✅ CSV + percentile reports + `tools/profile-check.ps1` gate (manual, not yet CI)
- Character/object motion vectors so HBTR can reproject dynamic objects exactly (camera motion already works). ✅ velocity attachment + jitter-consistent reprojection
- Shadows track the mesh's live instances in real time (hot-reload). ✅ The shadow pass binds the same GPU-compacted per-image buffer and indirect draw count as the geometry pass, so culled/removed/animated entities no longer leave stale shadows. `Renderer::update_instances` overrides the shadow with live ECS entity poses when present; `hbd_engine::ecs::instances_from_objects` maps `Object` entities to `render::Instance`.
- Clean, crisp image without SSAO film grain. ✅ `ssao.wgsl` uses a golden-angle hemisphere with interleaved-gradient rotation instead of a white-noise `fract(sin())` hash, and compares sampled scene depth against the actual hemisphere sample position (with the correct surface-normal orientation) instead of the centre pixel. The previous comparison reported depth discontinuities as occlusion and produced speckle on flat surfaces. AO radius/intensity are tuned to contact scale (0.7 / 0.75) so props keep their bright colours instead of being darkened by large-radius occlusion.
- HBTRU temporal upscaling (AA + upscale) with hardware work tiers. ✅ Output-resolution history with input-resolution colour/velocity/depth; `HbtrQuality` Low/Medium/High/Epic shares one shader with uniform-gated work. ✅ `--quality`, `--res WxH`, and `tools/capture-hbtr.ps1` capture harness.
- Changeable internal render resolution independent of the window size. ✅ `RendererConfig::resolution` / `Renderer::set_render_resolution` with native/1080p/1440p/2k/4k/custom presets; the sandbox exposes `--res` and F1–F5, with uncapped presentation by default and `--vsync` for A/B.
- Resize/fullscreen never loses explicit settings or produces black HBTR flicker. ✅ `RendererSettings` is the single source of truth for the cull A/B, instance count, debug view, HDR, light, pass cache, animation and live shadow instances; every pass rebuild re-applies it. A swapchain generation counter catches recreations that happen inside `VulkanContext::render` (suboptimal/out-of-date presents). HBTR carries an explicit per-image history-valid flag: freshly allocated/post-resize history is never sampled (an undefined sample poisoned the luminance-weighted resolve into black even with a zero blend), and luminance denominators are guarded against NaN.

**Exit criteria:** 1440p/60 fps on RTX 3060-class with the scale targets enabled; no p99 regressions; HBTR passes the no-ghosting camera-pan test. **Open:** all three need the target GPU and a reference capture set; none can be closed on the Intel integrated dev GPU.

---

## M3 — Editor and tooling

**Goal:** a separate editor executable that consumes the engine.

**Work items**

- `apps/editor` binary depending on `hbd-engine` only; engine gains `editor`/`hot-reload` feature gates. ✅ first slice: ImGui over the engine renderer, `render_with_overlay` hook
- Scene hierarchy, entity inspector, asset browser, viewport with the engine renderer embedded. ⏳ hierarchy/inspector read-only; asset browser pending
- Hot reload for WGSL shaders, cooked assets, and Lua scripts. ✅ WGSL via `hbd_render::shader::source()` under `editor` feature; assets/Lua pending
- Serialization through `bevy_reflect`; undo/redo and a stable `.scene` format.
- Headless cooker and CI rendering tests using the same code paths as the editor.

**Exit criteria:** editor builds independently of the engine's internals; a shader edit is visible without restarting; engine crate has no editor dependency.

---

## M4 — Runtime, assets, physics, scripting, audio

**Goal:** turn configuration placeholders into working systems.

**Work items**

- ECS schedules (fixed tick at 60 Hz, interpolation, render extraction) and plugin registration.
- Cooked `.pak` pipeline: stable `AssetId`, generational handles, chunked streaming, IO workers, per-frame budgets, LRU residency. Target 10 GB streamed world without traversal hitches.
- Jolt integration behind the physics trait; fixed-tick thread and interpolated transforms.
- Sandboxed Lua 5.4 via `mlua` with a curated API and end-of-frame mutation queue.
- Audio output and asset-backed playback.

**Exit criteria:** a moving scene with physics bodies, scripts and streamed assets runs a 60 s path with no hitch over 16.6 ms attributable to streaming or ticking.

---

## M5 — Quality, stability, and research

**Goal:** make the engine pleasant and trustworthy to use.

**Work items**

- Validation layers in CI where available; Vulkan error/warning triage.
- Shader pre-compilation and pipeline caching to remove first-frame stutter.
- Reference-scene screenshot suite (FlightHelmet, MetalRoughSpheres, Sponza, emissive interior).
- Global illumination research; path tracing stays behind an explicit opt-in.

**Exit criteria:** visual suite green on target hardware; clean validation; no first-frame stutter.

---

## Explicit non-goals

- Nanite/Lumen-class virtual geometry or GI in M1/M2.
- Path tracing as a default path, ever, without a separate decision and ADR.
- macOS/Linux or Direct3D 12/Metal backends in v1.
- AAA content quality; reference scenes are test fixtures, not shipped content.
