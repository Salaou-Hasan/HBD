# Surface banding investigation (2026-11-22)

## 1. Root cause

The vertical stripes/bands were **not** a geometry, normal, lighting or AA
problem. They came from the bindless material table's demo patterns being
sampled with a world-space projection:

```wgsl
textureSample(material_textures, material_sampler,
              input.world_position.xz * 0.5, material_layer)
```

Layer 1 of the 4-layer table was a vertical stripe pattern (`x < 2` bright,
else dark) and layer 3 a gradient. The sampler used `REPEAT` wrapping. On a
vertical cube face one component of `world_position.xz` is constant and the
other varies, so the stripe/gradient pattern was projected directly onto the
face, producing the reported bands. There were no per-face UVs: the material
table was effectively a world-space decal on every surface.

## 2. Files and components responsible

- `crates/render/src/mesh.wgsl` — fragment shader material sampling.
- `crates/render/src/mesh.rs` — `Vertex` layout, `cube_mesh()`, `load_obj()`,
  pipeline vertex attribute descriptions, `create_material_table()`.

Lighting, normals, rasterization state, depth handling and the HBTR/HBTRU
pipeline were inspected and cleared; none contributed to the artifact.

## 3. Why it appeared inconsistently

`build_scene()` assigns `material.w = (x + z).rem_euclid(4)`, so only a quarter
of the cubes selected each layer. Only layers 1 and 2 produced obvious
stripes/gradients; the others were subtle. The phase of `world_position.xz`
also differed per cube, so the same pattern landed at different offsets on
different objects. Camera angle changed which part of the repeating pattern
was visible. That matches the report: multiple objects, varying visibility,
sometimes absent.

## 4. Changes implemented

1. **Real geometry UVs.** `Vertex` now carries `uv: [f32; 2]` (stride 24 → 32
   bytes). `cube_mesh()` writes per-face 0..1 UVs; `load_obj()` derives stable
   planar UVs from the normalized local position. The mesh pipeline declares
   `@location(2)` `R32G32_SFLOAT`.
2. **Geometry-relative material sampling.** The fragment shader samples the
   material array with `input.uv` instead of `world_position.xz`.
3. **Neutral default material table.** The four demo patterns were replaced
   with neutral white texels. Bindless layer selection is preserved; it becomes
   meaningful when a real asset/material pipeline supplies textures.

No object-specific or sight-line-specific code was added.

## 5. Verification

- Diagnostic: temporarily forced `texel = vec3(1.0)` and captured the same
  static scene. The amplified difference image (`band-diff.png` during the
  investigation) lit up exactly the cube sides with the bands.
- After the fix, a capture compared against that diagnostic returned
  `MAE 0.000000, p95 0.000000, ghost_score 0.000000` — the corrected image is
  pixel-identical to the version with no material texture at all.
- Before the fix, the same comparison reported `MAE 0.008486, p95 0.039216,
  ghost_score 0.068724`.
- Native, `--upscale-50`, `--no-cull` and `--quality high` runs all exit 0.
- Moving-camera capture inspected: cube faces are uniform, no bands.
- `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace`
  pass. New test `cube_mesh_has_face_local_uvs` asserts the per-face UV layout.
- Per-pass cost is unchanged within noise: geometry ~0.45-0.48 ms, HBTR
  ~0.71-0.75 ms at native 720p. Whole-frame timing was noisy on this shared
  machine (2.3 ms to 11 ms between runs), so no total-frame claim is made.

## 6. Remaining limitations

- The OBJ loader still has no real UV channel; its UVs are a planar fallback.
- The material table is a neutral placeholder until the asset pipeline exists;
  bindless selection is exercised but currently has no visible per-layer look.
- Reference screenshots from the original bug report were not available to
  this session, so the verification uses equivalent static/moving captures
  from the same test scene and a forced-neutral diagnostic.
