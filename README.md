# HBD

HBD is a Windows-first, multithreaded 3D game engine written in Rust. It is built from zero around a custom Vulkan renderer, an ECS-first data-oriented design, and a strict library/tool boundary: the engine ships as `hbd-engine`, and the future editor will be a separate executable that depends on the engine — never the other way around.

> **Current milestone: M1 — rendering baseline.** The engine renders a lit scene end to end. The editor does not exist yet, and the performance target has not been formally measured on the target GPU. See [Status](#status) and [ROADMAP.md](docs/ROADMAP.md).

## What works today

- **Vulkan 1.3 renderer** through `ash` with `gpu-allocator` RAII buffers/images. WGSL shaders are compiled to SPIR-V at runtime by `naga`, so the Vulkan SDK is not needed to build.
- **M1 render pipeline:** 3-cascade directional shadows (1024², PCF) → instanced HDR geometry with GGX/Smith/Schlick PBR, procedural sky and analytic IBL → SSAO → bloom → AgX tonemap into the sRGB swapchain.
- **Debug views** for every major buffer, selected with number keys `0`–`7` in the sandbox.
- **Validation sandbox** with an orbiting camera, a fixed-camera mode, a frame limit, and a Vulkan probe mode.
- **Stable configuration layer** for the job system, runtime tick, asset streaming budgets, physics, scripting, and renderer options, each covered by unit tests.
- **E2E profiler** with GPU timestamp queries, per-pass p50/p95/p99, and CSV export (`--profile`). All operations are logged; see [PROFILING.md](docs/PROFILING.md) and [ADR 0004](docs/adr/0004-visible-operations.md).
- **Visible offscreen pass cache**: with an unchanged camera pose, shadow/geometry/SSAO/bloom are skipped per swapchain image (6.3x GPU reduction in static mode), with `--no-cache` as A/B control.
- **CI** on Windows runs `cargo fmt --check`, `cargo clippy -D warnings`, and `cargo test --workspace`.

- **HBD Temporal Resolve** — an original temporal AA with YCoCg variance clipping and a confidence blend, enabled by default; SSAO is bilaterally denoised. See [ANTI-ALIASING.md](docs/ANTI-ALIASING.md). Path tracing is research-only and off by default.

## Status

| Area | State |
|---|---|
| Renderer (M1 visual baseline) | Implemented; formal image comparison pending |
| Windowing / Vulkan context / resize | Implemented |
| Debug views | Implemented (`0`–`7`) |
| ECS (`bevy_ecs` standalone) | World + facade implemented; gameplay systems pending |
| Assets, physics, scripting, audio, runtime schedules | Configuration/types only (placeholders) |
| Editor executable | Not started |
| GPU-driven 1M draw items / 10M triangles | Planned for M2 |
| 1440p / 60 fps on RTX 3060-class | Target defined; not yet validated locally |

The local development GPU is an Intel integrated adapter with Vulkan 1.4 and no validation layers installed. `VK_EXT_debug_utils` is used as a fallback. Performance numbers from this machine are not representative of the target GPU.

## Quick start

Requirements: Windows 10/11, Rust 1.100+ (see `rust-toolchain.toml`), a Vulkan 1.3-capable GPU/driver.

```powershell
cargo build --workspace
cargo test --workspace
cargo run -p hbd-sandbox
```

Sandbox options:

```powershell
cargo run -p hbd-sandbox -- --probe            # print Vulkan devices, no window
cargo run -p hbd-sandbox -- --static-camera    # freeze the camera for comparisons
cargo run -p hbd-sandbox -- --frames 300       # render N frames, then exit

Early (temporary) real-mesh test, documented in [EARLY-OBJ-TEST.md](docs/EARLY-OBJ-TEST.md):

```powershell
cargo run -p hbd-sandbox -- --obj "$env:TEMP\opencode\stanford-bunny.obj" --static-camera
```

In the sandbox window, press `0`–`7` for final / filmic / HDR / bloom / AO / depth / shadow cascade / TAA history.

Full instructions and troubleshooting: [BUILDING.md](docs/BUILDING.md).

## Repository layout

```text
apps/sandbox/        validation executable; not the editor
crates/core/         math, time, handles, errors
crates/jobs/         job-system configuration (work-stealing pool planned)
crates/platform/     winit window, event loop, window config
crates/gfx/          Vulkan device, queues, swapchain, sync, allocator
crates/render/       render passes, camera, shaders, debug views
crates/assets/       AssetId, handles, pak and streaming configuration
crates/scene/        ECS world and scene facade
crates/physics/      Jolt configuration and fixed-tick policy
crates/script/       Lua configuration and sandbox policy
crates/audio/        audio output placeholder
crates/runtime/      tick/runtime configuration; schedules and plugins pending
crates/engine/       public facade re-exporting all engine crates
docs/                architecture, renderer, roadmap, building, ADRs
```

Downstream code should depend on `hbd-engine` only. Internal `hbd-*` crates may move without a public API break.

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — crate map, dependency rules, runtime and data flow, subsystem status.
- [RENDERER.md](docs/RENDERER.md) — M1 pipeline, targets, debug views, TAA status, known gaps.
- [ROADMAP.md](docs/ROADMAP.md) — milestones, deliverables, exit criteria.
- [BUILDING.md](docs/BUILDING.md) — setup, commands, sandbox flags, troubleshooting.
- [EARLY-OBJ-TEST.md](docs/EARLY-OBJ-TEST.md) — temporary `--obj` hook for seeing a real mesh in the renderer (not the asset pipeline).
- [PROFILING.md](docs/PROFILING.md) — E2E profiler, measured baseline/after results, A/B procedure, rejected experiment.
- [ANTI-ALIASING.md](docs/ANTI-ALIASING.md) — how Unity/Godot/UE handle temporal noise and HBD's own HBTR variant.
- [CONTRIBUTING.md](docs/CONTRIBUTING.md) — conventions and quality gate.
- [visual-target.md](docs/visual-target.md) — the visual quality bar and M1 acceptance tests.
- [ADR 0001 — stack](docs/adr/0001-stack.md), [0002 — editor boundary](docs/adr/0002-editor-boundary.md), [0003 — TAA disabled (superseded)](docs/adr/0003-taa-disabled.md), [0004 — no hidden operations](docs/adr/0004-visible-operations.md), [0005 — HBD Temporal Resolve](docs/adr/0005-hbtr.md).

## Design targets

- Windows 10/11, Vulkan 1.3, 8 GB VRAM budget, 10 GB streamed world.
- 1M GPU-driven draw items (indirect/multi-draw), 10M+ visible triangles.
- 1440p / 60 fps on RTX 3060-class hardware.
- Predictable frame times, no shader-compilation stutter, hot reload for shaders/assets/scripts.
- Rasterization-first; path tracing is research-only.

These are goals, not claims about the current build. See [ROADMAP.md](docs/ROADMAP.md).

## License

MIT OR Apache-2.0.
