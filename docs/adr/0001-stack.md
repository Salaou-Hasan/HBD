# ADR 0001 — HBD Engine Stack

- **Status:** Accepted
- **Date:** 2026-11-19
- **Owners:** HBD
- **Supersedes:** none

## Context

HBD is a 3D game engine written in Rust. The engine is consumed as a library; the editor will be a separate executable that depends on the library. Windows is the only supported target for now.

The renderer is custom: Vulkan through `ash`, a hybrid forward + deferred pipeline, TAA as the default anti-aliasing path, and a research track for a custom AA technique. Physics is Jolt. Scripting is Lua. The runtime must be multithreaded, and the asset system must stream large worlds without frame hitches.

## Decisions

### D1 — The engine is a library; the editor is a consumer

`hbd-*` crates form the engine. `editor` will be a separate binary crate that depends on `hbd-engine`. The engine must never depend on the editor. Editor-only functionality is feature-gated (`editor`, `hot-reload`) and excluded from shipping builds.

The public API wraps third-party types (`ash`, `oxijolt`, `mlua`). Users of the engine should not need to import those crates.

The engine also supports a **headless mode** for cooking, CI, dedicated servers, and automated rendering tests.

### D2 — Windows-only, Vulkan 1.3 baseline

Target: Windows 10/11, Vulkan 1.3-capable GPU. No OpenGL fallback, no Direct3D 12 backend, no macOS or Linux support in v1.

This keeps the renderer focused and avoids a cross-platform abstraction tax. MoltenVK or a Linux port can be evaluated later; they are not design constraints now.

### D3 — `winit` for windowing and input

`winit` 0.30 is the window/input layer. GLFW was considered and rejected: it is a C dependency, its event loop is awkward to integrate with an engine runtime, and it is weaker for editor/multi-window integration. `winit` is Rust-native, actively maintained, and provides `raw-window-handle` interop for Vulkan surface creation.

### D4 — `ash` + `gpu-allocator` for Vulkan

`ash` 0.38 provides raw Vulkan bindings with no hidden abstraction. `gpu-allocator` 0.28 handles device memory suballocation through a shared allocator; GPU buffers and images free their own suballocations on drop. The allocator is dropped explicitly before `destroy_device` because its destructor frees remaining blocks through the device.

`vulkano` was considered: it is safer and faster to start with, but its abstractions fight custom synchronization, render-graph resource aliasing, bindless descriptors, and async compute. The renderer needs full control, so we own the thin safe layer.

`wgpu` was rejected because it hides Vulkan features (descriptor indexing, mesh shaders, ray tracing, sparse resources) that the renderer depends on.

### D5 — `bevy_ecs` for the ECS

`bevy_ecs` 0.20 is used standalone. It does not pull in Bevy's renderer or app framework. It provides archetype storage, parallel system scheduling, change detection, events, and a mature query API. `bevy_reflect` will be used for editor inspection and serialization.

`hecs` was considered but is too minimal for editor tooling. `flecs` is powerful but is a C library with an FFI boundary and a non-Rust-idiomatic API.

The engine wraps `bevy_ecs` behind its own facade so the choice can be revisited without rewriting gameplay code.

### D6 — Jolt Physics via `oxijolt`

`oxijolt` 1.6 tracks Jolt 5.6 through the `joltc` C API and provides a safe Rust layer. It is the most active Jolt binding available, but it is young.

Mitigation: the engine wraps it behind its own `physics` trait, vendors/pins the Jolt source, and keeps `rapier3d` as a fallback if the binding blocks progress. Physics runs on a dedicated fixed-tick thread; rendering interpolates between physics states.

### D7 — Lua via `mlua`

`mlua` 0.12 supports Lua 5.4/5.5, Luau, and LuaJIT. Start with Lua 5.4 (`vendored` feature) for compatibility, with Luau as an option if type annotations or performance justify it.

Scripts run on a single script thread initially. Mutations go through a command buffer applied at the end of the schedule. Shipping builds sandbox `os`, `io`, `package`, and `debug`. Hot reload is editor-only.

### D8 — Custom asset pipeline with cooked paks

Runtime never parses source formats. The offline cooker imports glTF, produces meshlets, compresses textures to BC7/ASTC/KTX2, and packs assets into `.pak` archives with a manifest.

Runtime uses stable `AssetId` values and generational `Handle<T>` values. Assets are chunked (texture mips/tiles, mesh LODs/clusters, audio blocks), streamed through dedicated IO threads, decompressed on workers, and uploaded on a transfer queue with per-frame budgets. The runtime never takes a global asset lock.

### D9 — Hybrid forward + deferred, TAA first

Opaque standard materials go through a deferred G-buffer. Transparent, alpha-tested, and forward-only materials go through clustered Forward+. Both paths share the light grid, shadows, IBL, and post stack.

TAA is the default anti-aliasing path because it is required infrastructure for SSR, SSGI, volumetrics, and motion blur. MSAA remains available for the forward path. A custom AA technique is a research track: it ships only if it beats TAA on the visual test suite and the frame-time budget.

### D10 — Multithreaded runtime

Main thread: OS events, ECS schedule, orchestration. Worker pool: parallel ECS systems, culling, animation, decompression, command recording. Physics thread: fixed tick. IO threads: asset reads. Render thread: submission. Audio thread.

Simulation and rendering communicate through an extraction snapshot (`RenderWorld`), not by sharing a locked world.

## Performance Envelope

Target hardware: **RTX 3060-class GPU** at **1440p, 60 fps minimum**. RTX 4070-class and above should reach 120+ fps at 1440p with scalable settings, and 60+ fps at 4K.

| Metric | Target |
|---|---|
| Minimum frame rate | 1440p / 60 fps on RTX 3060-class |
| High-end frame rate | 1440p / 120+ fps on RTX 4070-class |
| Draw items | 1,000,000 per frame |
| Visible triangles | 10,000,000+ per frame |
| VRAM budget | 8 GB |
| Streamed world size | 10 GB+ |
| Frame hitch during traversal | none over 8 ms |
| CPU render overhead | under 1 ms |
| Frame-time p99 | within 20% of median |

### Metric definitions

**“1M draw calls”** means **1,000,000 draw items submitted through GPU-driven indirect/multi-draw**, with culling and LOD on the GPU. It does **not** mean 1,000,000 CPU-issued `vkCmdDrawIndexed` calls. One million individual Vulkan draw commands per frame is not achievable at 60 fps on the target hardware; no engine does that. The metric we can and will hit is one million visible draw items.

**“10M visible triangles”** means 10M triangles surviving frustum, occlusion, and LOD selection, rendered through meshlet/cluster geometry. It does not mean 10M unique high-poly objects.

**“High fps without reducing performance”** means a scalable quality system: resolution scale, shadow quality, GI quality, AA mode, and geometry LOD are independent knobs. Performance is tracked by a benchmark harness, not by feel.

## Visual Quality Bar

See [`docs/visual-target.md`](../visual-target.md).

The short version: the first lit frame must use the full HDR/PBR/IBL/tonemap/TAA pipeline, not a gray triangle. “Looks like Unreal but better” is interpreted as matching the rendering grammar early, then competing on frame-time predictability, streaming, iteration speed, and GPU-driven scale — not on content parity, which depends on art.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `oxijolt` is young and may have API/build issues | Wrap behind our own trait; pin/vendor Jolt; keep `rapier3d` fallback |
| `ash` 0.38 is pinned to Vulkan 1.3.281 headers | Vulkan 1.3 + extensions covers our needs; regenerate bindings from `vk.xml` if 1.4 core features become necessary |
| No Vulkan SDK / `glslc` on the dev machine | Resolved for bootstrap: WGSL is compiled to SPIR-V at runtime by `naga`; the cooked asset pipeline can later move compilation offline |
| Dev GPU is Intel integrated, target is RTX 3060 | Validate correctness on Intel; validate performance on target hardware; never assume a feature is free |
| “Best in every domain” causes scope paralysis | Benchmark suite + north-star scene + explicit acceptance gates |
| TAA ghosting on thin geometry and alpha-tested foliage | TAA test suite; responsive AA; neighborhood clamping; custom AA research |
| Shader/PSO compilation stutter | Disk pipeline cache; background PSO compilation; permutation reduction; precompile in cooker |

## Consequences

- The engine is Windows/Vulkan-only in v1. Portability is deferred.
- The renderer owns its Vulkan layer; more unsafe code, but no ceiling from a wrapper.
- `bevy_ecs` brings a large dependency tree; the engine facade isolates it.
- Jolt via FFI adds a C++ build dependency (CMake/MSVC on Windows).
- The custom AA technique is not on the critical path; TAA is the shipping default until the research track wins.
- The 1M draw-item target is only meaningful with GPU-driven rendering, so meshlet/cluster pipeline work starts in M2, not “later.”

## Follow-ups

1. ~~Adopt a pure-Rust shader compiler before the first triangle.~~ Done: `naga` WGSL → SPIR-V.
2. Build the M0 Vulkan clear-screen path, then the first lit PBR frame. M0 and a depth-tested 3D bootstrap are done; the full PBR frame is still pending.
3. Define the benchmark scenes and capture baseline numbers on the Intel dev GPU and an RTX 3060.
4. Validate `oxijolt` on Windows with a falling-cube test before committing to the physics architecture.
5. Decide whether the custom AA research starts in M4 or after the editor exists.
