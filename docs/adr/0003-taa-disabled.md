# ADR 0003 — TAA disabled in the M1 runtime

> Historical decision. The "TAA" in this document is the original M1 blend that
> was disabled; it is not the shipping path. HBTR (ADR 0005) supersedes it, and
> all shipping code names the resolver `hbtr`, never `taa`.

- **Status:** Superseded by [ADR 0005](0005-hbtr.md) — HBD Temporal Resolve is now the default AA path.
- **Date:** 2026-11-20
- **Supersedes:** none
- **Related:** ADR 0001, [visual-target.md](../visual-target.md)

## Context

The original M1 pipeline applied Halton-jittered temporal anti-aliasing with a 0.9 history blend. During camera orbit the image showed visible shake and a washed-out, smeared appearance. TAA history is per swapchain image: with three images, an image's history is three frames old when it comes back around. A single shared previous view-projection matrix was being used to reproject history produced by a different camera pose, so 90% of the final color came from incorrectly reprojected samples.

A correction was made in `hbd-taa`: per-image previous view-projection poses, initialization flags, and history ping-pong indices, with the uniform written only after the acquired image's in-flight fence. However, temporal accumulation itself still produced artifacts, and the decision was made that shipping a visibly unstable image is worse than shipping a sharper but noisier one.

## Decision

The M1 runtime sets `TAA_ENABLED = false` in `crates/render/src/renderer.rs`:

- camera jitter is zero,
- the TAA resolve pass is skipped,
- bloom and tonemap bind the raw HDR target instead of the resolved target.

The TAA pass, shaders, and corrected per-image bookkeeping remain in the tree, dormant. SSAO and AgX remain active; they are not part of this decision. Path tracing remains off by default under a separate rule.

## Consequences

- The image is temporally stable and colors stay distinct; fine SSAO noise is visible because no temporal accumulation smooths it.
- Edges are less anti-aliased than a working TAA would provide. Other AA approaches are a research track in M2.
- The `TemporalHistory` debug view shows a stale buffer while TAA is off.
- `RendererConfig::AntiAliasing::Taa` still describes the design target; the runtime switch is the source of truth for M1. See `docs/RENDERER.md`.

## Re-enable criteria

TAA may be turned back on only when all of the following hold:

1. No visible ghosting on a fast horizontal camera pan through thin geometry.
2. No shake or washout on the orbit path used for M1 validation.
3. Reprojection uses motion vectors or a pose that genuinely matches the history being sampled, including the sub-pixel jitter of that frame.
4. The visual suite passes on the target GPU class, with before/after captures.
5. `TAA_ENABLED` is flipped in the same commit that documents the result.
