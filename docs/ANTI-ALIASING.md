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

HBTR is implemented in `crates/render/src/hbtr_resolve.wgsl`. It is not a copy of any engine's shader. Its distinctive pieces:

1. **Single-pass 3×3 YCoCg moment analysis.** One neighborhood loop yields mean and variance for clipping *and* the average used for sharpening — no extra fetches for contrast recovery.
2. **Variance box + segment clipping.** History is clipped to `mean ± σ` intersected with the true neighborhood min/max, in YCoCg, by intersecting the segment from the neighborhood mean to the history color with the box (no corner accumulation).
3. **Confidence blend plus object velocity.** Reprojection confidence comes from (a) how far history was reprojected in pixels and (b) how much the history color had to be clipped; the blend slides from 0.9 toward 0.5 when either is large. A second color attachment (`R16G16B16A16_SFLOAT`) carries **object motion only** — current and previous poses under the same view-projection — so camera motion cancels and stays with the depth reprojection, avoiding double counting.
4. **Luminance-adaptive accumulation.** Current and history weights are divided by `1 + luma`, suppressing fireflies from bright HDR samples (Karis-style, applied to our blend).
5. **History validity and rejection.** Out-of-screen/behind-camera samples are rejected, freshly allocated or post-resize history is never sampled (an explicit per-image `history-valid` flag prevents an undefined sample from poisoning the luminance-weighted blend into black), and the first frame of a history slot uses blend 0.

History is per swapchain image, with its own jittered previous view-projection pose and ping-pong slot (see `crates/render/src/hbtr.rs`). With three swapchain images an image's history is three frames old, not one; using one shared matrix was the original ghosting bug. The resolve and the history copy are now a single draw with two color attachments.

## Cost and quality (Intel UHD, release build, uncapped, moving camera)

| Internal resolution | GPU frame p50 | AA pass p50 |
|---|---|---|
| 1280×720 native | 1.85 ms | 0.45 ms |
| 2560×1440 | 9.28 ms | 2.06 ms |

The AA pass is ~0.45 ms at native 720p and ~2 ms at 1440p on the integrated GPU; on an RTX 3060-class card that is a small fraction of a 16.6 ms budget. The cost buys subpixel-smooth edges, temporal smoothing of stochastic SSAO, and no ghosting on the orbit path. Debug views `0` (final), `4` (AO), `6` (shadow) and `7` (history) are available for inspection; the sandbox also exposes `--debug <view>`.

## Limitations

- Velocity is object-motion only: both endpoints use the current view-projection, so camera motion is entirely owned by depth reprojection. `--animate` exercises the object path with an explicit readback; a formal no-ghosting reference capture is still outstanding.
- No temporal upscaling: internal resolution presets upscale with the tonemap's linear sampler only. A TAAU/FSR2-style multi-frame upscale is the natural next step (1080p internal → 1440p/4K display) and would be the largest remaining visual win.
- The pass cache skips shadow/geometry/SSAO/bloom on identical camera poses but **not** HBTR — temporal accumulation must continue.
- History debug view 7 samples the read slot captured at pass construction; it may not show the latest ping-pong slot.

## Enabling/disabling

`HBTR_ENABLED` in `crates/render/src/renderer.rs` is the explicit switch (currently `true`). The older disable decision is recorded in [ADR 0003](adr/0003-taa-disabled.md) and superseded by [ADR 0005](adr/0005-hbtr.md).
