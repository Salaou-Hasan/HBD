# HBD Roadmap

This is a living plan. A milestone is done only when its exit criteria are met on the stated hardware — not when the code merely compiles.

## Guiding rules

1. **Rendering baseline before breadth.** A correct, stable lit frame precedes tooling, content, and scale.
2. **Engine library first.** The editor is a consumer of `hbd-engine`; the engine never depends on it.
3. **Measured, not assumed.** Performance claims require numbers from the target GPU class.
4. **No silent default changes.** Rendering paths that are experimental (path tracing, TAA until proven) stay off by default and are covered by tests.
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

- **HBD Temporal Resolve (HBTR)** is the default AA path (YCoCg variance clipping, confidence blend, free sharpening), replacing the old disabled TAA. See [ANTI-ALIASING.md](ANTI-ALIASING.md) and [ADR 0005](adr/0005-hbtr.md).
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

Remaining M2 work: GPU-driven indirect/multi-draw scale-up (1M items), 10M+ visible triangles, bindless materials, p99 regression dashboard, motion vectors for HBTR with dynamic objects.

**Motion-vector attempt (reverted):** a velocity attachment and HBTR velocity transform were implemented but produced corrupted imagery even with AA disabled, so the whole attempt was reverted to the last clean commit. No half-working code is left in the tree.

**Instance-count stress mode:** `--instances N` overrides the indirect instance count. Counts above the 82 source cubes repeat that source set modulo its count; this is an explicit GPU-throughput stress mode, not a scene of unique objects, and is logged as an override. It ran 100,000 instances on the Intel dev GPU without errors.

---

## M2 — Performance and GPU-driven rendering

**Goal:** hit the scale envelope without sacrificing the M1 look.

**Work items**

- Replace per-instance CPU draws with indirect/multi-draw; target **1M GPU-driven draw items**.
- Target **10M+ visible triangles** with cluster/frustum culling on the GPU.
- Bindless descriptor sets and material/texture caching; eliminate per-draw descriptor updates.
- Frame-time profiling (p50/p99), GPU timestamps, and a regression dashboard.
- Character/object motion vectors so HBTR can reproject dynamic objects exactly (camera motion already works).

**Exit criteria:** 1440p/60 fps on RTX 3060-class with the scale targets enabled; no p99 regressions; HBTR passes the no-ghosting camera-pan test.

---

## M3 — Editor and tooling

**Goal:** a separate editor executable that consumes the engine.

**Work items**

- `apps/editor` binary depending on `hbd-engine` only; engine gains `editor`/`hot-reload` feature gates.
- Scene hierarchy, entity inspector, asset browser, viewport with the engine renderer embedded.
- Hot reload for WGSL shaders, cooked assets, and Lua scripts.
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
