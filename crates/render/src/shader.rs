//! WGSL to SPIR-V compilation.
//!
//! The first renderer milestone uses WGSL because it is text-based, validated,
//! and compiles without an external Vulkan SDK. The cooked asset pipeline will
//! move compilation offline; the runtime boundary stays SPIR-V.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShaderError {
    #[error("WGSL parse error:\n{0}")]
    Parse(String),

    #[error("shader validation failed:\n{0}")]
    Validation(String),

    #[error("SPIR-V generation failed:\n{0}")]
    Backend(String),

    #[error("shader entry point '{0}' produced no SPIR-V")]
    Empty(String),
}

/// Returns the shader source for a pass.
///
/// Normal builds use the embedded `include_str!` bytes. With the `hot-reload`
/// feature, a file under `shaders/` wins when present, so the editor can pick
/// up a `.wgsl` edit by rebuilding passes without restarting the process.
pub fn source(name: &str, embedded: &'static str) -> String {
    #[cfg(feature = "hot-reload")]
    {
        let candidates = [
            std::path::Path::new("shaders").join(name),
            std::path::Path::new("crates/render/src").join(name),
        ];
        for path in candidates {
            if let Ok(text) = std::fs::read_to_string(&path) {
                tracing::debug!(?path, "loaded shader from disk for hot reload");
                return text;
            }
        }
        tracing::warn!(name, "hot-reload shader not found, using embedded source");
    }
    #[cfg(not(feature = "hot-reload"))]
    let _ = name;
    embedded.to_owned()
}

/// Compiles one entry point from WGSL into SPIR-V words.
pub fn compile_wgsl(
    source: &str,
    entry_point: &str,
    stage: naga::ShaderStage,
) -> Result<Vec<u32>, ShaderError> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| ShaderError::Parse(error.emit_to_string(source)))?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator
        .validate(&module)
        .map_err(|error| ShaderError::Validation(error.emit_to_string(source)))?;

    // WGSL clip space is y-up; Naga's coordinate-space adjustment produces the
    // y-down Vulkan framebuffer convention. Keep shader math in WGSL space and
    // let this single boundary handle the conversion.
    let options = naga::back::spv::Options {
        flags: naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE,
        ..Default::default()
    };
    let mut writer = naga::back::spv::Writer::new(&options)
        .map_err(|error| ShaderError::Backend(error.to_string()))?;

    let pipeline_options = naga::back::spv::PipelineOptions {
        shader_stage: stage,
        entry_point: entry_point.to_owned(),
    };
    let mut words = Vec::new();

    writer
        .write(&module, &info, Some(&pipeline_options), &None, &mut words)
        .map_err(|error| ShaderError::Backend(error.to_string()))?;

    if words.is_empty() {
        return Err(ShaderError::Empty(entry_point.to_owned()));
    }

    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SHADER: &str = r#"
        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
            var positions = array<vec2<f32>, 3>(
                vec2<f32>(0.0, -0.5),
                vec2<f32>(0.5, 0.5),
                vec2<f32>(-0.5, 0.5),
            );
            return vec4<f32>(positions[index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.5, 0.25, 1.0);
        }
    "#;

    #[test]
    fn compiles_vertex_and_fragment_entry_points() {
        let vertex =
            compile_wgsl(TEST_SHADER, "vs_main", naga::ShaderStage::Vertex).expect("vertex SPIR-V");
        let fragment = compile_wgsl(TEST_SHADER, "fs_main", naga::ShaderStage::Fragment)
            .expect("fragment SPIR-V");

        assert!(!vertex.is_empty());
        assert!(!fragment.is_empty());
    }

    #[test]
    fn reports_missing_entry_point() {
        let error = compile_wgsl(TEST_SHADER, "missing", naga::ShaderStage::Vertex)
            .expect_err("missing entry point must fail");

        assert!(matches!(error, ShaderError::Backend(_)));
    }
}
