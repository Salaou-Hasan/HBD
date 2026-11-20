# Building and Running HBD

## Requirements

- **OS:** Windows 10 or 11. HBD is Windows-only in v1.
- **Rust:** 1.100 or newer. `rust-toolchain.toml` selects stable and requires `rustfmt` and `clippy`.
- **GPU/driver:** Vulkan 1.3-capable adapter. Development has used an Intel integrated GPU with Vulkan 1.4.
- **Vulkan SDK:** not required for building. WGSL shaders are compiled to SPIR-V at runtime by `naga`.
- **Validation layers:** requested when installed. Without `VK_LAYER_KHRONOS_validation`, the engine logs a warning and continues with `VK_EXT_debug_utils`.

## Build and test

```powershell
cargo build --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

Run the validation sandbox:

```powershell
cargo run -p hbd-sandbox
```

Build without running:

```powershell
cargo build -p hbd-sandbox
.\target\debug\hbd-sandbox.exe
```

## Sandbox command line

| Flag | Effect |
|---|---|
| `--probe` | Print Vulkan loader/device information and exit without a window. |
| `--static-camera` | Freeze the camera at angle 0, useful for image comparisons. |
| `--obj <path>` | **Temporary early test**: replace the cube grid with one OBJ mesh. Not the asset pipeline; see [EARLY-OBJ-TEST.md](EARLY-OBJ-TEST.md). |
| `--instances N` | **Stress test**: override the indirect instance count. Counts above the 82 source cubes repeat that set modulo its count; logged as an override. |
| `--no-cache` | Force the full offscreen pipeline every frame (A/B control for the pass cache). |
| `--profile` | Enable GPU timestamp queries and CPU frame timing. |
| `--profile-report <path>` | Write a CSV profile on shutdown. |
| `--profile-interval N` | Frames between percentile summaries (default 120). |
| `--frames N` | Exit after N redraw frames. Useful for automated smoke tests. |

Examples:

```powershell
cargo run -p hbd-sandbox -- --probe
cargo run -p hbd-sandbox -- --static-camera --frames 300
```

Window size comes from `WindowConfig` (default `1280x720`, **physical pixels**). The debug views are bound to number keys `0`–`7`; see [RENDERER.md](RENDERER.md#debug-views).

## CI

`.github/workflows/ci.yml` runs on Windows for pushes and pull requests: checkout, stable Rust with `rustfmt`/`clippy`, Rust build cache, then format check, clippy with `-D warnings`, and the workspace test suite. Keep local changes green against these same commands before pushing.

## Troubleshooting

**PowerShell reports exit code 1 even though the build says `Finished`.** Cargo writes progress to stderr, and PowerShell can surface that as an error record when output is piped. Check for `Finished dev profile` in the output rather than trusting the shell's exit status in that specific piped form.

**`failed to remove file ... hbd-sandbox.exe: Access is denied`.** A sandbox process still has the executable open. Stop it before rebuilding:

```powershell
Get-Process hbd-sandbox -ErrorAction SilentlyContinue | Stop-Process -Force
```

**Very high reported FPS (thousands).** An occluded or minimized window often presents without meaningful GPU work. Bring the window to the foreground and use `--frames` with a settled capture before recording numbers.

**Repeated "swapchain created" / "pass ready" messages at startup.** `winit` may emit several resize events while the DPI settles; each one rebuilds the renderer. This is expected in M1 and is safe, but it means the first moments of a run are not representative.

**`Vulkan validation requested but unavailable`.** Expected on machines without the validation layers. Debug utils are used instead. Install the Vulkan SDK's validation layers if you need full validation.

**No image / black window.** Confirm the GPU supports Vulkan 1.3 with `--probe`, and check the log for pipeline errors. Path tracing is off by default; if you enabled experimental flags, revert them first.

## Headless use

`hbd-runtime::RuntimeConfig` has `headless: bool` and `max_frames`, and the platform layer can exit after a frame count. The full headless cooker/CI renderer is planned for M3/M4; today, automated visual checks use `--static-camera --frames N` plus screenshots.
