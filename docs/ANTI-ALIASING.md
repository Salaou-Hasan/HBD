# HBD Temporal Resolve (HBTR)

## What the engines do

Based on public documentation and source-level write-ups (Karis/GDC 2014, Salvi/GDC 2016, Yang et al. survey, UE5 TAA/TSR write-ups, Unity HDRP TAA, Godot 4 docs):

| Engine | Grain/temporal noise strategy |
|---|---|
| Unreal (TAA / TSR) | Jittered projection, velocity-based reprojection, history rectification with shaped YCoCg neighborhood clipping / variance clipping, depth and velocity validity checks, adaptive blend with anti-flicker, plus sharpening in TSR; TSR upsamples from multiple frames. |
| Unity (HDRP TAA) | Jittered camera, reprojection, 3×3 neighborhood AABB in YCoCg, variance clipping, adaptive blend, and an explicit sharpening kernel to counter temporal blur. |
| Godot 4 | Forward+ TAA converges jittered frames; docs acknowledge blur/ghosting under motion and recommend FSR2 or MSAA+TAA when available. |

The shared grammar is: **jitter → reproject → validate/rectify history → blend → sharpen**. The differentiator is how history is validated and how the blend is chosen. A 2024 k-DOP paper shows tighter clipping reduces ghosting further but costs noticeably more; we chose the cheaper variance-clip path.

## HBD's own variant

HBTR is implemented in `crates/render/src/taa_resolve.wgsl`. It is not a copy of any engine's shader. Its distinctive pieces:

1. **Single-pass 3×3 YCoCg moment analysis.** One neighborhood loop yields mean and variance for clipping *and* the average used for sharpening — no extra fetches for contrast recovery.
2. **Variance box + segment clipping.** History is clipped to `mean ± σ` intersected with the true neighborhood min/max, in YCoCg, by intersecting the segment from the neighborhood mean to the history color with the box (no corner accumulation).
3. **Confidence blend without a velocity buffer.** Instead of storing motion vectors, HBTR derives a per-pixel confidence from (a) how far history was reprojected in pixels and (b) how much the history color had to be clipped. Blend slides from the base 0.9 toward 0.5 when either signal is large. No extra render target, no G-buffer extension.
4. **Luminance-adaptive accumulation.** Current and history weights are divided by `1 + luma`, suppressing fireflies from bright HDR samples (Karis-style, applied to our blend).
5. **Out-of-screen / behind-camera rejection** before any history reuse, with the first frame of a history slot using blend 0.

History is per swapchain image, with its own previous view-projection pose and ping-pong slot (see `crates/render/src/taa.rs`). This is what fixed the earlier reprojection error: with three swapchain images an image's history is three frames old, not one.

## Cost and quality (Intel UHD, 1280×720, 900 frames)

| Config | GPU frame p50 | AA pass p50 |
|---|---|---|
| No AA, SSAO bilateral denoise | 2.14 ms | 0 |
| HBTR enabled | 3.34 ms | 0.89 ms |

The AA pass costs ~0.9 ms on the integrated GPU; on the RTX 3060 target this is a small fraction of a 16.6 ms budget. Present FPS stays near the 120 Hz cap. The cost buys: subpixel-smooth edges, temporal smoothing of the remaining SSAO speckle, and no ghosting on the orbit path. Debug views `0` (final), `4` (AO), `6` (shadow) and `7` (history) are available for inspection.

## Limitations

- No motion-vector buffer: moving *objects* (not yet in the engine) would need one. Camera motion is handled correctly by per-image previous matrices.
- No temporal upscaling: HBTR is native-resolution AA only.
- The pass cache skips shadow/geometry/SSAO/bloom on identical camera poses but **not** HBTR — temporal accumulation must continue.
- History debug view 7 samples the read slot captured at pass construction; it may not show the latest ping-pong slot.

## Enabling/disabling

`TAA_ENABLED` in `crates/render/src/renderer.rs` is the explicit switch (currently `true`). The older disable decision is recorded in [ADR 0003](adr/0003-taa-disabled.md) and superseded by [ADR 0005](adr/0005-hbtr.md).
