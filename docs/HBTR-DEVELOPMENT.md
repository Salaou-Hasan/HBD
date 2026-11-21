# HBTR / HBTRU development log (living document)

This is the hand-off document for the temporal resolve work. Anyone forking HBD
should read it before touching `crates/render/src/hbtr.rs` or the resolve shader.
It records the current design, the TSR study, the performance rules, and what is
done versus planned. Update it in the same commit as any resolve change.

## Names

| Name | Meaning |
|---|---|
| **HBTR** | HBD Temporal Resolve. Anti-aliasing only, native internal resolution. |
| **HBTRU** | HBD Temporal Resolve Upscale. Anti-aliasing **and** temporal upscaling from the engine's internal render resolution to the presentation resolution. |

There is no "TAA" in the shipping code. The old implementation that was disabled
in ADR 0003 is only mentioned in historical documents. Code identifiers, shader
files, profiler markers and CSV columns use `hbtr` / `HBTR`.

## Why this exists

The M1 resolve was named TAA but was really a plain blend without history
validation; it ghosted and was disabled (ADR 0003). HBTR (ADR 0005) added the
modern grammar: jitter, reproject, rectify history, confidence blend, sharpen.
That is working and is the default path today.

The remaining weaknesses, as of 2026-11-21:

1. A 3x3 YCoCg variance clip can still ghost when a fast object uncovers background.
2. Sharpening is a fixed 0.12, conservative to avoid halos.
3. Object velocity uses the pose from the frame that produced this image's history slot (correct for our per-image history, but only as good as the velocity attachment).
4. There is no formal no-ghosting reference capture.
5. Internal resolution presets use a plain linear upsample; there is no temporal upscaling.

## TSR / FSR2 study (what we actually adopt)

Sources: Epic's UE TSR documentation, AMD FSR2 documentation and GDC material,
Karis (GDC 2014), Salvi (GDC 2016). Key ideas and our decision:

| Technique | TSR / FSR2 role | HBTR / HBTRU decision |
|---|---|---|
| Jittered camera + velocity reprojection | mandatory | already implemented; per-image history age |
| History at **display** resolution | mandatory for upscaling | **adopt** in HBTRU: history/resolved allocated at output extent, current/velocity/depth stay at render extent |
| YCoCg neighborhood rectification | TAA/HDRP style | already implemented; **extend** with tighter k-DOP planes on higher quality tiers |
| Reactive / shading-rejection mask | reduces ghosting on dynamic objects | **adopt** cheaply from existing velocity texel distance + colour variance, no new render target |
| Parallax disocclusion from depth+velocity | rejects history on reveal | **adopt** as a depth-neighbour test in the resolve shader (4 taps, only on High/Epic) |
| Flickering temporal analysis | stabilises thin geometry | defer; expensive and not needed for our scene scale |
| History resurrection | recovers temporarily occluded detail | defer; needs extra history buffers |
| Nyquist-Shannon supersampled history (200%) | removes reprojection blur | defer; 4x history cost, revisit only afterHBTRU is proven |
| Mitchell / bicubic history filtering | smoother upscale | **adopt** a 4-tap cubic reconstruction when upscale factor > 1 |
| RCAS-style sharpening | user-facing sharpness | **adopt** adaptive sharpening driven by local contrast, replacing the fixed 0.12 |
| Async parallax compute | hides cost | defer; our frame is small and the resolve is already one draw |

## Hardware scalability rule

One algorithm, several **work tiers**, never a different-looking fallback.
Tiers change the amount of work, not the grammar:

| Tier | Target | k-DOP planes | Parallax taps | History filter | Sharpening |
|---|---|---|---|---|---|
| Low | entry-level integrated | variance clip (AABB) | 0 | bilinear | 0.12 |
| Medium | mid-range iGPU / dGPU | 8 planes | 4 | bilinear | adaptive |
| High | modern dGPU | 16 planes | 4 | 4-tap cubic | adaptive |
| Epic | high-end | 16 planes + resurrection | 8 | 4-tap cubic | adaptive + stronger clip |

The default is Medium so an entry-level machine and a high-end machine produce
the same image shape; High/Epic add fidelity where the GPU can afford it. Any
new feature must be expressible as a uniform branch so Low still compiles it out
at runtime without a second shader permutation.

## Performance rules (non-negotiable)

- No hidden work: every pass, tier change, and capture is logged through `tracing`.
- Measure before and after with the release build and the CSV profiler. A change
  is an optimisation only if the after-measurement confirms it.
- Low tier must not regress versus the current 0.45 ms @ 720p / 2.06 ms @ 1440p
  HBTR cost on the Intel dev GPU by more than 10%.
- HBTRU must beat the old linear-upscale preset at the same internal resolution
  in a side-by-side capture, and its cost must be reported separately.

## Work board

| Item | State | Notes |
|---|---|---|
| Rename all TAA identifiers/files to HBTR | planned | code, profiler CSV column, docs |
| Study TSR/FSR2 and record decisions | done | this document |
| Adaptive sharpening + reactive confidence | planned | shader-only, uniform-gated |
| Tighter k-DOP history rectification | planned | quality tiers |
| HBTRU dual-extent history + cubic reconstruction | planned | pass creation + renderer extent plumbing |
| No-ghosting capture harness | planned | explicit readback, PPM dumps, documented procedure |
| Docs and ADRs refreshed | planned | same commit as code |

## Test procedure for forks

1. `cargo build --release -p hbd-sandbox`
2. `.\target\release\hbd-sandbox.exe --frames 600 --profile --profile-report .\profile.csv`
3. Compare p50/p99 of `pass_hbtr` against the table above on the same machine.
4. Capture a fast camera pan with `--capture-hbtr <dir>` (once implemented) and
   compare consecutive frames for smearing; record the result in the PR.
5. Never claim "better than TSR" without a reference capture and a cost number.
