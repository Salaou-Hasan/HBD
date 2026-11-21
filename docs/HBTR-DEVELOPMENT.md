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

Original weaknesses and what happened to them:

1. **3x3 variance clip ghosting** → k-DOP planes added for Medium+, but only as a *rejection signal* on moving pixels. Diagonal clipping that rewrites YCoCg was measured to produce coloured fringes, so the proven AABB clip stays on every tier. Open: verify on a fast-uncover reference capture.
2. **Fixed 0.12 sharpening** → now contrast-adaptive on Medium+, but capped at 1.15x. Larger boosts were measured to produce coloured fringing at cube edges on the Intel dev GPU. Open: a CAS-style luma-only curve with a proper reference set.
3. **Object velocity pose** → still the pose from the frame that produced this image's history slot; that is intentional and correct for per-image history. The `--animate` readback verifies it; a formal ghosting capture is now possible with the harness below.
4. **No reference capture** → `tools/capture-hbtr.ps1` writes numbered window PNGs plus a manifest; documented procedure in this file.
5. **No temporal upscaling** → HBTRU now allocates history/resolved at the **presentation** extent while colour/velocity/depth stay at the internal render extent. The resolve reconstructs the current sample on the input grid from the same nine taps and accumulates into output-resolution history. Still bilinear rather than cubic; see Open Work.

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
| Rename all TAA identifiers/files to HBTR | done | shader, types, variables, profiler marker, docs |
| Study TSR/FSR2 and record decisions | done | this document |
| Adaptive sharpening + reactive confidence | done | capped at 1.15x; reactive gated to Medium+ |
| Tighter k-DOP history rectification | done | rejection-only on moving pixels; AABB retained |
| HBTRU dual-extent history | done | history/resolved at output extent; grid-aligned reconstruction |
| No-ghosting capture harness | done | `tools/capture-hbtr.ps1` + `tools/compare-hbtr.ps1` |
| Cubic history reconstruction | done, experimental | 16-tap Catmull-Rom for Epic only; measured worse vs native reference and +0.44 ms, so not enabled on High |
| Docs and ADRs refreshed | in progress | this file, ANTI-ALIASING.md, ADR 0005 |

### Measured cost (Intel UHD, release, 1280x720 output, moving camera, 600 frames)

| Tier | HBTR p50 | Total frame p50 |
|---|---|---|
| Low | 0.49 ms | 1.76 ms |
| Medium | 0.51 ms | 1.79 ms |
| High | 0.57 ms | 1.87 ms |
| Epic | 0.57 ms | 1.88 ms |

The previous single-tier resolver was 0.45 ms / 1.85 ms. Low is within 10% of
the old pass and total frame time is slightly better; the extra work on higher
tiers is < 0.1 ms. Re-measure on any hardware change and update this table.

## Test procedure for forks

1. `cargo build --release -p hbd-sandbox`
2. `.\target\release\hbd-sandbox.exe --frames 600 --profile --profile-report .\profile.csv`
3. Compare p50/p99 of `pass_hbtr` against the table above on the same machine.
4. Capture a native reference and the upscaled result with the same camera path:
   `powershell -File tools/capture-hbtr.ps1 -AppArgs @('--static-camera','--res','native') -Frames 4 -OutDir .\hbtr-ref`
   then `powershell -File tools/capture-hbtr.ps1 -AppArgs @('--static-camera','--upscale-50') -Frames 4 -OutDir .\hbtr-test`.
5. Score the pair:
   `powershell -File tools/compare-hbtr.ps1 -ReferenceDir .\hbtr-ref -TestDir .\hbtr-test -OutCsv .\hbtr-compare.csv`.
   Track MAE, p95 and the edge-weighted `ghost_score`; record them in the PR.
6. Never claim "better than TSR" without a reference capture and a cost number.

## Open work (do not claim it is done)

- **Automatic upscale toggle**: `--upscale` selects `RenderResolution::Auto(50)`; `--upscale-N` selects N percent (1..=100, clamped). Both recompute the internal size on every swapchain rebuild and override `--res`. Verified 640x360 at a 1280x720 window and 1280x778 after maximizing to 2560x1555 at 50%.
- **Cubic history reconstruction**: implemented as a 16-tap Catmull-Rom history filter, Epic-only. On a static 50% upscale it scored `ghost_score` 0.0229 vs 0.0104 for bilinear Medium against a native reference, and cost +0.44 ms. Available for comparison, not claimed as better; promote it only if a future reference set prefers it.
- **Ghosting metric limits**: `tools/compare-hbtr.ps1` reports MAE, p95 and an edge-weighted difference. It is not yet an optical-flow-aware smear detector, and the capture harness uses `PrintWindow`, so use the same window state for both sequences.
- **Epic features**: history resurrection and flickering temporal analysis from
  TSR are deliberately deferred.
- **Target GPU validation**: 1440p/60 on RTX 3060-class and a reference-image
  comparison against UE TSR cannot be closed on the Intel integrated dev GPU.
