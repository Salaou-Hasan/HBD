# Early OBJ Visualization Test

> **This is a temporary validation hook, not the asset pipeline.** HBD is not at the level of real asset importing yet. This exists so the renderer can be exercised with a recognizable mesh before M4 delivers cooked `.pak` assets, streaming, materials, and handles. Do not build production code on it.

## Purpose

See a real 3D model move through the M1 pipeline (shadow → HDR geometry → SSAO → bloom → AgX) instead of the hardcoded cube grid. It answers one question only: *does the renderer draw an arbitrary triangle mesh correctly?*

## Get a test model

Download the Stanford Bunny (69,451 triangles) from the common 3D test models repository. It is ~2.4 MB:

```powershell
$obj = "$env:TEMP\opencode\stanford-bunny.obj"
New-Item -ItemType Directory -Force -Path (Split-Path $obj) | Out-Null
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj' -OutFile $obj
```

The OBJ is **not committed to this repository**. It is third-party data with its own provenance/licensing; keep it in your local temp folder or another local assets directory. See the [source repository](https://github.com/alecjacobson/common-3d-test-models) for model details.

## Run

```powershell
cargo run -p hbd-sandbox -- --obj "$env:TEMP\opencode\stanford-bunny.obj" --static-camera
```

Omit `--static-camera` to orbit the model. Combine with `--frames N` for automated runs.

Successful startup logs this line:

```text
INFO hbd_render::mesh: mesh pipeline ready instances=1 triangles=69451 source="obj"
```

Without `--obj`, the sandbox renders the original 82-cube grid, unchanged.

## What the hook actually does

- Parses only `v` (positions) and `f` (faces) records from the OBJ. Polygon faces are fan-triangulated; negative indices and `v/vt/vn` style tokens are accepted (only the vertex index is used).
- Computes smooth normals by accumulating triangle face normals per vertex. File-supplied `vn` normals are ignored.
- Centers the mesh on its bounding box, scales its largest dimension to 5 world units, rotates it 180° around Y, and places it at `y = 1.6`.
- Renders it as **one instance** with a hardcoded blue material (`base_color = [0.06, 0.16, 0.5]`, metallic `0.18`, roughness `0.36`).
- The rest of the M1 pipeline is identical: same HDR targets, shadows, SSAO, bloom, AgX, and debug views (`0`–`7`).

Source: `crates/render/src/mesh.rs` (`load_obj`), exposed through `Renderer::new_with_obj` in `crates/render/src/renderer.rs` and the sandbox's `--obj <path>` flag. A unit test covers triangulation and normalization.

## Limitations — why this is not the asset pipeline

- **No materials, textures, UVs, or supplied normals.** Everything renders with one hardcoded color.
- **Faces must appear after the vertices they reference.** No multi-pass resolution; out-of-range indices are an error.
- **No `o`/`g`/`usemtl` grouping.** The whole file becomes a single draw instance.
- **No ground plane in OBJ mode.** All instances share one vertex buffer, so the cube-grid ground cannot be combined with the imported mesh without a second draw path. The model floats against the procedural sky.
- **CPU parse at startup.** No cooking, caching, compression, or streaming. Very large OBJs will be slow to load and are not supported in practice.
- **No handles, `AssetId`, residency, or budgets.** This does not touch `hbd-assets`.
- **Fixed framing.** Only the sandbox orbit camera; no model inspection controls.
- **This hook is expected to be removed** when the real asset pipeline lands in M4. Do not add features to it — add them to the asset pipeline instead.

## If it fails

- Check the path: the sandbox opens the file directly and reports `OBJ load failed: ...` for missing/unreadable files.
- Use a simple triangulated OBJ. Exotic exporters, split vertex-normal indexing, or faces before vertices can fail.
- Confirm the GPU/Vulkan setup first with `--probe`.
- TAA is disabled in M1; that is unrelated to OBJ loading. Debug views `2` (HDR) and `5` (depth) are useful for isolating geometry problems.
