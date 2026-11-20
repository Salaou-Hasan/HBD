# HBD

HBD is a Windows-first 3D game engine written in Rust.

The engine is a library. The editor will be a separate binary that depends on the engine, never the other way around.

## Status

**M1 — complete rendering baseline.** The sandbox renders an instanced grid of 82 cubes through a full pipeline: 3-cascade shadow → HDR geometry → TAA resolve → SSAO → bloom → AgX tonemap. It has a perspective camera, metallic-roughness PBR (GGX/Smith/Schlick), 3×1024² cascaded directional shadows with PCF, a procedural sky, analytic hemisphere IBL, half-resolution SSAO and bloom, Halton-jittered TAA with reprojection and neighborhood clamping, and AgX tonemapping into the sRGB swapchain. GPU memory is suballocated through `gpu-allocator`; shaders are authored in WGSL and compiled to SPIR-V at runtime by `naga` (no Vulkan SDK required to build). Asset streaming, physics, and scripting crates are still architectural placeholders.

All M1 rendering features from the visual gate are now present. Debug views are selectable with number keys `0`–`7` in the sandbox (final, filmic, HDR, bloom, AO, depth, shadow cascade, TAA history). Path tracing is research-only and off by default. See [`docs/visual-target.md`](docs/visual-target.md).

## Stack

| Concern | Choice |
|---|---|
| Platform | Windows 10/11, Vulkan 1.3+ |
| Window + input | `winit` |
| Graphics API | Vulkan via `ash` + `gpu-allocator` |
| Math | `glam` |
| ECS | `bevy_ecs` (standalone, no Bevy renderer) |
| Physics | Jolt via `oxijolt` |
| Scripting | Lua via `mlua` |
| Renderer | custom hybrid forward + deferred |
| Anti-aliasing | TAA default, custom AA research track |
| Assets | custom cooked pak pipeline + async streaming |

## Build

```powershell
cargo build --workspace
cargo test --workspace
cargo run -p hbd-sandbox
```

Probe Vulkan without opening a window:

```powershell
cargo run -p hbd-sandbox -- --probe
```

Run a fixed number of frames and exit:

```powershell
cargo run -p hbd-sandbox -- --frames 300
```

## Layout

```text
apps/sandbox/        executable used to validate the engine
crates/core/         math, time, handles, errors
crates/jobs/         job system configuration
crates/platform/     windowing and OS integration
crates/gfx/          Vulkan device, queues, allocator
crates/render/       render graph, passes, materials, shaders
crates/assets/       asset database, cooker, pak, streaming
crates/scene/        ECS world, transforms, serialization
crates/physics/      Jolt integration, fixed tick
crates/script/       Lua VM and bindings
crates/audio/        audio output
crates/runtime/      app loop, schedules, plugins
crates/engine/       public facade
docs/                architecture decisions and targets
```

## Renderer notes

- `WindowConfig` width/height are **physical pixels**; winit uses `PhysicalSize`.
- Vulkan validation layers are requested when installed. This machine has debug utils but not `VK_LAYER_KHRONOS_validation`, so the sandbox logs a warning and continues.
- WGSL is authored y-up; `naga`'s `WriterFlags::ADJUST_COORDINATE_SPACE` produces Vulkan y-down clip-space output.
- `VulkanContext` drops the GPU allocator explicitly before destroying the device, because `gpu-allocator` frees its remaining blocks through that device.
- `RendererConfig::path_tracing` is `false` by default. Path tracing is a research-only idea and is **not** part of M1; the engine is rasterization-first.

## Docs

- [`docs/adr/0001-stack.md`](docs/adr/0001-stack.md) — accepted stack decisions and performance envelope.
- [`docs/visual-target.md`](docs/visual-target.md) — the visual quality bar and M1 acceptance gate.

## License

MIT OR Apache-2.0.
