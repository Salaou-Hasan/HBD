// Temporary GPU-generation stress path for M2. This is not the shipping scene
// format and not the asset pipeline: it fills an instance buffer with a bowl
// of cubes so the indirect draw path can be exercised at scale. The cubes are
// distributed through a paraboloid volume; this is a static stress scene, not
// a physics simulation.
//
// The Instance layout must match mesh.wgsl, cull.wgsl and the Rust `Instance`
// struct: model (64) + previous_model (64) + base_color (16) + material (16)
// = 160 bytes. A stride mismatch silently corrupts vertex/cull reads.
struct Instance {
    model: mat4x4<f32>,
    previous_model: mat4x4<f32>,
    base_color: vec4<f32>,
    material: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> instances: array<Instance>;

const GOLDEN_ANGLE: f32 = 2.39996323;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total = arrayLength(&instances);
    if (index >= total) {
        return;
    }

    let radius = 4.0;
    let depth = 2.0;

    if (index == 0u) {
        let ground_model = mat4x4<f32>(
            vec4<f32>(10.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.2, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 10.0, 0.0),
            vec4<f32>(0.0, -2.1, 0.0, 1.0),
        );
        instances[index].model = ground_model;
        instances[index].previous_model = ground_model;
        instances[index].base_color = vec4<f32>(0.12, 0.13, 0.15, 1.0);
        instances[index].material = vec4<f32>(0.0, 0.92, 1.0, 0.0);
        return;
    }

    let sample = f32(index - 1u) + 0.5;
    let count = f32(total - 1u);
    let t = sample / count;
    let angle = GOLDEN_ANGLE * sample;
    let r = radius * sqrt(t);
    let x = r * cos(angle);
    let z = r * sin(angle);
    let y = depth * t * t - depth;
    let position = vec3<f32>(x, y, z);
    let scale = 0.22;
    let model = mat4x4<f32>(
        vec4<f32>(scale, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, scale, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, scale, 0.0),
        vec4<f32>(position, 1.0),
    );

    let rim = clamp(r / radius, 0.0, 1.0);
    let base_color = vec4<f32>(
        0.18 + rim * 0.62,
        0.24 + (1.0 - rim) * 0.45,
        0.78 - rim * 0.45,
        1.0,
    );

    instances[index].model = model;
    instances[index].previous_model = model;
    instances[index].base_color = base_color;
    instances[index].material = vec4<f32>(0.15, 0.42, 1.0, 0.0);
}
