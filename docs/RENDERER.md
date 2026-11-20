# M1 Renderer

The renderer is rasterization-first, custom, and fully inspectable. It renders an instanced grid of 82 colored cubes with a perspective camera. There is no scene/asset input yet; the scene is built in `crates/render/src/mesh.rs`.

## Pipeline order

| # | Pass | Output | Notes |
|---|---|---|---|
| 1 | Shadow | 3 × 1024² depth | Directional light, cascades at 12 / 32 / 80 world units, 3×3 PCF |
| 2 | Geometry | HDR color + depth, per swapchain image | `R16G16B16A16_SFLOAT`; instanced cubes + fullscreen procedural sky |
| 3 | TAA resolve | resolved HDR | **Bypassed** in M1; see below |
| 4 | SSAO | half-res AO | Depth-based, sampled once in the tonemap |
| 5 | Bloom | half-res bright pass + blur | Reads the HDR image currently being presented |
| 6 | Tonemap | swapchain (`B8G8R8A8_SRGB`) | AgX, bloom add, AO multiply, vignette |

The HDR color attachment ends in `SHADER_READ_ONLY_OPTIMAL`, so bloom and tonemap can sample it without a separate transition.

## Lighting and materials

- Metallic-roughness PBR: GGX distribution, Smith geometry, Schlick Fresnel, energy-conserving diffuse/specular split.
- Directional sun with 3 cascaded shadow maps and PCF. Shadowed fragments use the first cascade below 12 units, second below 32, third below 80.
- Analytic image-based lighting: hemisphere sky irradiance for diffuse and a roughness-attenuated sky reflection for specular. This is not a convolved cubemap, but it preserves the diffuse/specular energy split.
- Procedural sky uses a pale horizon `(0.55, 0.62, 0.78)`, blue zenith `(0.04, 0.14, 0.42)`, a sun disc and a soft glow.
- HDR is tonemapped with an AgX fit (`AGX_INPUT`, `AGX_OUTPUT`, smooth log-space curve) into the sRGB swapchain. Debug view 1 shows the older filmic curve for comparison.

## Targets and frame synchronization

- `MAX_FRAMES_IN_FLIGHT = 2`, 3 swapchain images, FIFO present mode.
- Per swapchain image: HDR color view, depth view, TAA resolved view, two TAA history views, SSAO view, bloom views, descriptor sets, and framebuffers.
- Uniform uploads for the acquired image occur after that frame's in-flight fence is signaled.

## Camera and jitter

`Camera::new(position, target)` builds a perspective view/projection. When TAA is enabled, the projection is jittered by a Halton(2, 3) sub-pixel sequence and the resolve reprojects using the previous frame's view-projection matrix. The sandbox orbits the camera; `--static-camera` freezes it for stable comparisons.

## Debug views

In the sandbox, press:

| Key | View |
|---|---|
| `0` | Final (AgX) |
| `1` | Filmic tonemap |
| `2` | HDR radiance (Reinhard compressed) |
| `3` | Bloom |
| `4` | Ambient occlusion |
| `5` | Depth |
| `6` | Shadow cascade 0 |
| `7` | TAA history |

`Renderer::set_debug_view(DebugView)` exposes the same modes through the engine API.

## TAA status: disabled

TAA produced artifacts (visible shake / washed-out accumulation during camera motion). The M1 runtime therefore sets `TAA_ENABLED = false` in `crates/render/src/renderer.rs`:

- no sub-pixel camera jitter,
- no TAA resolve or history work,
- bloom and tonemap bind the raw HDR target.

The TAA pass remains in the tree with a corrected per-image design: each swapchain image keeps its own previous view-projection pose, initialization flag, and history ping-pong index, because an image is revisited every `image_count` frames rather than every frame. It is kept dormant until the temporal resolve itself is redesigned. Re-enable criteria and rationale are in [ADR 0003](adr/0003-taa-disabled.md).

**SSAO and AgX are unaffected** — both remain active in the shipped path.

## Known gaps

- No motion vectors; reprojection relies on depth + view-projection only.
- `TonemapPass` captures history views at construction, so debug view 7 shows its original slot.
- No exposure control or color grading UI; exposure is implicit in the tonemap.
- No shader/pipeline cache; first frames can stutter while WGSL compiles.
- Validation layers are unavailable on the development machine; only `VK_EXT_debug_utils` is active.
- Formal RTX 3060-class 1440p/60 validation and reference-image comparison are outstanding.

See [visual-target.md](visual-target.md) for the acceptance bar and [ROADMAP.md](ROADMAP.md) for the plan to close these gaps.
