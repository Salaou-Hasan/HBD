# ADR 0002 — Editor as a separate executable

- **Status:** Accepted
- **Date:** 2026-11-20
- **Supersedes:** none
- **Related:** ADR 0001

## Context

HBD needs both a runtime engine and an editor. Mixing them into one crate would let editor-only code leak into shipping builds, force the runtime to carry editor dependencies, and make it impossible to tell whether a bug lives in the engine or the tool.

## Decision

The engine is a library crate (`hbd-engine`) that downstream crates depend on. The editor will be a separate binary (`apps/editor`) that depends on `hbd-engine`. The engine must never depend on the editor or on editor-only modules. Editor-specific functionality is gated behind Cargo features (`editor`, `hot-reload`) that shipping builds exclude.

`apps/sandbox` is a validation executable, not the editor. It may use engine APIs but must not become a second editor.

## Consequences

- Public API changes are made in `hbd-engine`; internal `hbd-*` crates can move freely behind it.
- Editor serialization uses `bevy_reflect` exposed by the engine, not editor-private reflection code inside the engine.
- Hot reload is opt-in and off by default.
- A future editor can be replaced or forked without touching the engine.
- Cost: some duplication of window setup between sandbox and editor until a shared app-shell layer is added.
