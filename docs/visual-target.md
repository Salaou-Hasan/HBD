# HBD Visual Target

## What “looks like Unreal but better” means

Unreal's look is not one feature. It is the combination of:

- physically based materials and lighting,
- image-based lighting and a correct HDR/color pipeline,
- soft shadows and ambient occlusion,
- a strong post stack (bloom, exposure, tonemap, grading),
- temporal stability,
- and art direction.

HBD will match that **rendering grammar from the first lit frame**. Content parity with a AAA game is not an engine milestone; it depends on assets and art. The engine's job is to make the right thing easy and fast.

“Better” is defined on measurable axes:

1. **Frame-time predictability** — fewer hitches, tighter p99, no shader-compilation stutter.
2. **Streaming** — 10 GB+ worlds with no traversal hitches.
3. **Iteration speed** — hot reload for shaders, assets, and scripts; fast editor startup.
4. **GPU-driven scale** — 1M draw items and 10M visible triangles.
5. **Openness** — the renderer is inspectable and modifiable, not a black box.

## M1 Visual Baseline — the first gate

Current state: the sandbox renders an instanced cube grid with a perspective camera, metallic-roughness PBR, 3-cascade directional shadows, a procedural sky, analytic IBL, HDR off-screen targets, half-res SSAO, half-res bloom, and AgX tonemapping. TAA is implemented but **disabled in the M1 runtime** because it produced visible artifacts; see [ADR 0003](adr/0003-taa-disabled.md) and [RENDERER.md](RENDERER.md). Formal screenshot comparison on RTX 3060-class hardware is still pending because that GPU is not available yet. Debug views are exposed through `DebugView` and the sandbox number keys `0`–`7`. Path tracing is explicitly research-only and disabled by default.

The first lit frame must use the full pipeline, not a gray triangle:

- Linear HDR rendering, correct sRGB texture decode, no banding in the sky or gradients.
- Metallic-roughness PBR with energy conservation.
- Image-based lighting: procedural sky + irradiance + prefiltered specular environment.
- Directional sun with cascaded shadow maps and soft PCF.
- SSAO at half resolution, denoised.
- Bloom, exposure, AgX tonemap, color grading, subtle vignette.
- TAA with motion vectors, neighborhood clamping, and disocclusion handling — target for the re-enable gate, not part of the current M1 path.
- A debug view for every buffer: albedo, normal, roughness, depth, motion, shadow cascades, AO, bloom, TAA history.

### Reference scenes

Use assets with known-good PBR values:

- Khronos `FlightHelmet` — material/IBL validation.
- Khronos `MetalRoughSpheres` — roughness/metalness response.
- Khronos `Sponza` — scale, shadows, alpha-tested foliage.
- A custom interior with area-light-like emissive panels — exposure and bloom validation.

No reference scene ships with the engine; they are test fixtures.

## Acceptance tests

- Screenshot comparison against a reference render at fixed exposure.
- No visible TAA ghosting on a fast horizontal camera pan through thin geometry — applies when TAA is re-enabled; tracked in ADR 0003.
- No shadow acne or peter-panning at 1440p on the reference scene.
- Correct color: a 0.18 gray card renders at the expected linear value after tonemap.
- 1440p / 60 fps on RTX 3060-class at the M1 quality preset.
- No frame over 16.6 ms during a 60-second camera path (M1 scope; streaming comes later).

## Non-goals for M1

- Nanite/Lumen-class global illumination.
- Path tracing.
- Virtual textures.
- 1M draw items / 10M triangles (M2 performance work).
- AAA content quality.

These are explicit non-goals so the visual baseline ships instead of expanding forever.
