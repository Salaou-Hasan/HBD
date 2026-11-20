# ADR 0004 — No hidden operations

- **Status:** Accepted
- **Date:** 2026-11-20
- **Supersedes:** none

## Context

The user's strict rule for M2 and beyond: **no operation should be hidden ever**. An optimization that silently skips work, silently falls back, or silently writes data makes it impossible to trust the image or the performance numbers. This is especially important for a custom renderer where correctness bugs look like "the image is a bit off".

## Decision

1. Every skipped pass, uniform upload, cache hit, cache invalidation, mode toggle, fallback, profiling enable/disable, and report write must be logged through `tracing` or recorded in the profiler CSV.
2. Any behavior that changes what work is done must have an explicit A/B switch. The offscreen pass cache has `--no-cache`; profiling has `--profile`.
3. Unsupported capability must be reported, not substituted silently (GPU timestamps print a warning and switch to CPU-only mode, clearly labelled).
4. An optimization that measures worse is reverted, not kept because it sounds clever. The rejected precomputed-SSAO-sample experiment and its numbers are documented in [PROFILING.md](../PROFILING.md).
5. Documentation must state what the engine actually does, including limitations and regressions.

## Consequences

- Logs can be verbose in motion (one invalidation line per frame). This is intentional; `RUST_LOG` can filter it, but the information is never withheld by the engine.
- Every new optimization needs an observable signal and a way to turn it off before it can merge.
- Reviewers should check logs and the profiler CSV, not just the final image, when validating performance changes.
