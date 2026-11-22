# HBD Editor (M3, scene editor)

## Stack

| Layer | Choice |
|---|---|
| Language | Rust 2024, edition workspace |
| Windowing | `winit 0.30` through `hbd_platform` |
| Graphics | Vulkan via `ash 0.38`, owned by `hbd_gfx` / `hbd_render` |
| UI | `dear-imgui-rs 0.22` (imgui-rust docking line) |
| Platform backend | `dear-imgui-winit 0.22` |
| Vulkan backend | `dear-imgui-ash 0.22` (render-pass mode) |
| Engine dependency | `hbd-engine` only — never internal crates |

There is no wgpu, SDL, or second windowing system. ImGui draw data is recorded
into the engine's own swapchain render pass.

## Build and run

```powershell
cargo build --release -p hbd-editor
.\target\release\hbd-editor.exe
```

The editor runs its event loop on a 16 MB worker thread (like the sandbox) because renderer construction allocates large descriptor arrays on the stack; the default 1 MB main-thread stack overflows in debug builds.

The editor uses vsync presentation by default. It is a normal HBD renderer:
same HDR geometry, HBTR/HBTRU, SSAO, bloom and AgX tonemap as the sandbox.

## What is in M3

- `apps/editor` binary depending only on `hbd-engine`, split into `scene`
  (editable model), `gizmo` (picking + `dear-imguizmo`), and `ui` (panels).
- `Renderer::render_with_overlay(...)` — an explicit hook that records a
  caller-supplied Vulkan overlay after tonemap, inside the presentation pass.
- Dockable panels with stable `WindowKey`s: **Hierarchy** (objects, click to
  select), **Inspector** (position/rotation/scale, color, metallic,
  roughness, AO), **Stats** (frame, resolution, quality, cull counts, debug
  views), and **Shaders** (watched WGSL files, reload, resolution/quality).
  The dockspace's central node is intentionally left **empty** with
  `PASSTHRU_CENTRAL_NODE`: Dear ImGui paints an opaque background over any
  window docked there, so the 3D viewport is that empty area. A transparent
  floating **Viewport Controls** window holds the menu and gizmo toolbar.
  Picking and the ImGuizmo transform gizmo work over the full viewport;
  `imgui.ini` persists user docking changes.
- `Renderer::update_scene_instances(...)` — the engine's live edit path. The
  edit is queued and applied with `vkCmdUpdateBuffer` to the acquired image's
  host-visible live source buffer at the top of the geometry pass, before the
  cull pass and the no-cull draw read it. Fixed-count edits only; the pass
  cache is invalidated and every update is logged. Add/delete are deferred.
- Hot reload: with the `editor` feature, `hbd_render::shader::source()` reads
  `crates/render/src/*.wgsl` from disk instead of the embedded `include_str!`.
  The editor watches modification times and calls `Renderer::rebuild_passes()`.
  A change is visible without restarting. Normal builds keep embedded shaders.

## Architecture rules

1. The editor never depends on `hbd-render`, `hbd-gfx`, or any other internal
   crate. It uses the public facade types and methods only.
2. The engine has no editor dependency. The overlay is a generic closure, so
   the engine does not know ImGui exists.
3. Every operation that changes state (shader reload, resolution change,
   quality change, debug view) is logged through `tracing`.
4. ImGui texture retirement is explicit: a retirement token is stored by the
   overlay and waited on at the start of the next frame with a visible log.

## Hot-reload workflow

1. Open `crates/render/src/mesh.wgsl` (or any watched shader) in an editor.
2. Change a colour or constant and save.
3. The editor logs `shader changed; rebuilding engine passes` and the next
   frame uses the new shader. The **Shaders** panel also has a manual reload
   button if automatic detection is disabled.

## Limitations

- Scene edits are fixed-count: transforms, colors and material parameters of
  the seeded `build_scene()` objects can be changed live, but adding or
  deleting objects is not exposed yet.
- No asset browser, serialization, undo/redo, or Lua integration yet.
- Hot reload watches source files under the repository root; packaged builds
  should ship precompiled SPIR-V and disable the feature.
- If you have an `imgui.ini` from an earlier M3 build, delete it once to pick
  up the corrected default dock tree (empty passthrough central node).
