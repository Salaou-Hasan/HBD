# Contributing to HBD

## Before you change code

Read [ARCHITECTURE.md](ARCHITECTURE.md) for the library/editor boundary, [RENDERER.md](RENDERER.md) before touching rendering, and [ROADMAP.md](ROADMAP.md) to see whether the work belongs to the current milestone.

If a change alters a public API, a rendering default, or a milestone's exit criteria, update the relevant doc **in the same commit**.

## Conventions

- Rust 2024 edition; `cargo fmt --all` on every change.
- Public items get a short doc comment explaining *why*, not just *what*.
- Prefer new values over mutation; keep functions small and nesting shallow.
- Every fallible operation has explicit error handling via `thiserror` types; no silent `unwrap()` in library code.
- Third-party engine types (`ash`, `oxijolt`, `mlua`) stay behind the `hbd-engine` facade.
- No secrets, tokens, or machine-specific absolute paths in the repository.
- No debug printing in shipped code; use `tracing`.

## Quality gate

Run all of these before pushing:

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

For renderer changes, also run:

```powershell
cargo build -p hbd-sandbox
cargo run -p hbd-sandbox -- --static-camera --frames 300
cargo run -p hbd-sandbox -- --probe
```

Visual changes need a before/after capture with `--static-camera` and a note about which debug views were inspected. Performance claims need numbers from RTX 3060-class hardware, not the Intel development GPU.

## Tests

- New behavior gets a test before or with the implementation.
- Shipping defaults are locked by unit tests (renderer path, path tracing off, physics tick, script sandbox, runtime tick). Do not weaken a default test without an ADR.
- Regression tests for bugs should reproduce the original failure condition.

## Commits

Use conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `perf:`, `ci:`. Keep a commit focused on one change; include the reasoning in the body when it is not obvious from the diff.

## Decisions

Architecture choices that constrain future work go into `docs/adr/`, numbered sequentially. An ADR records context, decision, alternatives, and consequences — including what would make us reverse the decision.
