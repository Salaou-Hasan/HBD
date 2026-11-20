struct ShadowFrame {
    light_view_projection: mat4x4<f32>,
};

struct Instance {
    model: mat4x4<f32>,
    base_color: vec4<f32>,
    material: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: ShadowFrame;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(instance_index) instance_index: u32,
) -> @builtin(position) vec4<f32> {
    return frame.light_view_projection * instances[instance_index].model
        * vec4<f32>(position, 1.0);
}

@fragment
fn fs_main() {}
