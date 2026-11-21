# HBD Architecture

## The boundary that matters

HBD is a **library**. `hbd-engine` is the public facade; games, tools, and the future editor depend on it. The engine must never depend on an executable or on editor-only code. The editor will be `apps/editor`, a separate binary that links `hbd-engine` and enables editor-only features through Cargo features (`editor`, `hot-reload`) that are excluded from shipping builds.

Downstream code should not import `ash`, `oxijolt`, or `mlua` directly; the facade re-exports what gameplay needs. This keeps the public surface stable while internals change. See [ADR 0002](adr/0002-editor-boundary.md).

The engine also has a headless mode for cooking, CI, dedicated servers, and automated rendering tests.

## Crate map

| Crate | Responsibility | Implementation status |
|---|---|---|
| `hbd-core` | Math (`glam`), `Time`, generational `Handle<T>`, error types | Implemented |
| `hbd-jobs` | Work-stealing pool configuration; avoids two competing thread pools | Config only |
| `hbd-platform` | `winit` event loop, window, `WindowConfig` (physical pixels) | Implemented |
| `hbd-gfx` | Vulkan device/queues/swapchain/sync, `gpu-allocator` RAII buffers and images | Implemented |
| `hbd-render` | M1 render passes, camera, procedural sky, WGSL shaders, HBD Temporal Resolve, debug views | Implemented |
| `hbd-assets` | `AssetId`, `AssetKind`, handles, streaming budgets, pak design | Types/config only |
| `hbd-scene` | Standalone `bevy_ecs` world and facade | World + facade; systems pending |
| `hbd-physics` | Jolt backend selection, 60 Hz fixed tick, interpolation config | Config only |
| `hbd-script` | Sandboxed Lua 5.4 configuration and mutation policy | Config only |
| `hbd-audio` | Audio output placeholder | Placeholder |
| `hbd-runtime` | Fixed tick, target FPS, headless and frame-limit config | Config only |
| `hbd-engine` | Public facade and `prelude` | Implemented |
| `apps/sandbox` | Validation executable; **not** the editor | Implemented |

A table entry saying "config only" means the types and shipping defaults exist and are tested, but the runtime system is not wired into the app loop yet. Do not infer more from a crate's name.

## Dependency direction

```text
apps/sandbox, apps/editor (future)
        │  depends on
        ▼
   hbd-engine (facade)
        │  re-exports
        ▼
hbd-core, hbd-jobs, hbd-platform, hbd-gfx, hbd-render,
hbd-assets, hbd-scene, hbd-physics, hbd-script, hbd-audio, hbd-runtime
```

Internal crates depend downward only. `hbd-render` depends on `hbd-gfx` and `hbd-core`; it does not own the window. `hbd-platform` owns the native event loop and window but knows nothing about Vulkan. `hbd-engine` is the only crate tools should import.

## Runtime and threading

### Current (M1)

- `hbd-platform` runs the `winit` event loop. `on_window_created` builds the Vulkan context and renderer; `on_redraw` updates the camera and calls `Renderer::render`.
- GPU work is submitted from the event-loop thread. `VulkanContext::render` waits the in-flight fence for the current frame, acquires the swapchain image, records geometry/post passes, then presents.
- `MAX_FRAMES_IN_FLIGHT = 2`; the swapchain has 3 images, so each image has its own HDR/depth target, HBTR history pair, and descriptor set.
- `Time` provides elapsed/delta; `WindowConfig::max_frames` is a test-only exit condition.

### Planned

- A work-stealing job pool (`hbd-jobs`, initially `bevy_tasks` reused from `bevy_ecs`) with `workers = available_parallelism - 1`.
- A dedicated render thread extracting an immutable frame snapshot from the ECS world.
- ECS schedules: fixed 60 Hz physics tick with interpolation, variable-rate script/update, render extraction.
- Jolt on its own fixed-tick thread, publishing interpolated transforms rather than sharing mutable body state.

The current single-threaded submission is a deliberate simplification: it removes CPU/GPU races from the first renderer while the pass structure is still moving.

## Rendering data flow

```text
shadow maps (3 cascades, 1024², PCF)
        │
        ▼
HDR geometry pass (R16G16B16A16_SFLOAT + depth, per swapchain image)
        │  instanced PBR cubes, procedural sky, analytic IBL
        ├──────────────► depth ──► SSAO (half res)
        ▼
HBTR resolve (HBTR) ──► bloom (half res)
        │                                  │
        └────────► tonemap (AgX) ◄─────────┘
                       │
                       ▼
                 swapchain (sRGB)
```

Bloom and tonemap bind the HBTR-resolved views. The explicit switch is `HBTR_ENABLED` in `crates/render/src/renderer.rs` (currently enabled). Details in [RENDERER.md](RENDERER.md) and [ANTI-ALIASING.md](ANTI-ALIASING.md).

## Shader authoring

- Shaders are authored in WGSL (`crates/render/src/*.wgsl`) and compiled to SPIR-V at runtime with `naga` using `WriterFlags::ADJUST_COORDINATE_SPACE`. No Vulkan SDK is required to build.
- WGSL is authored y-up; naga produces Vulkan y-down clip-space output.
- Each pass owns its descriptor sets and framebuffers; consumers are constructed after producers so descriptor sets can bind the producer's image views once at construction.

## Memory and synchronization

- `hbd-gfx` wraps Vulkan buffers/images in RAII types that suballocate through `gpu-allocator` and free on drop. The allocator is destroyed before the device because it frees remaining blocks through that device.
- One HDR color image and one depth image per swapchain image prevents frame N+1 from overwriting targets frame N is still presenting.
- Uniform writes for the acquired image happen after its in-flight fence is waited on. This matters especially for HBTR: each image carries its own reprojection pose because it is revisited every `image_count` frames, not every frame.

## Subsystem designs (not yet implemented)

- **Assets:** source formats (glTF/PNG/FBX) are imported by an offline cooker into chunked `.pak` archives. Runtime holds stable `AssetId`s and generational handles, with IO workers, per-frame byte/time budgets, and LRU residency. Target: 10 GB streamed world without traversal hitches.
- **ECS:** `bevy_ecs` used standalone, no Bevy renderer or app. Systems operate on plain components; `bevy_reflect` will back editor serialization.
- **Physics:** Jolt behind the engine's own backend enum, fixed 60 Hz tick, max 3 steps/frame, interpolated transforms.
- **Scripting:** sandboxed Lua 5.4 with a curated API; mutations queued and applied at the end of the script schedule. Hot reload stays off in shipping defaults.
- **Runtime:** headless flag for cooker/CI/servers; plugin registration and schedules land with M3/M4.

## Known structural gaps

- `VulkanContext::new` can leak partially created Vulkan objects on early failure.
- `Renderer::resize` rebuilds all passes; it does not preserve resources if the swapchain format changes.
- `TonemapPass` captures the HBTR history views once at construction, so debug view 7 always samples its original slot.
- No GPU timing, frame graph, or pipeline cache yet.
