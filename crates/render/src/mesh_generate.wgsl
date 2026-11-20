// Temporary GPU-generation stress path for M2. This is not the shipping scene
// format and not the asset pipeline: it fills an instance buffer with a
// deterministic grid so the indirect draw path can be exercised at scale.
struct Instance {
    model: mat4x4<f32>,
    previous_model: mat4x4<f32>,
    base_color: vec4<f32>,
    material: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> instances: array<Instance>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total = arrayLength(&instances);
    if (index >= total) {
        return;
    }

    let side = u32(ceil(pow(f32(total), 1.0 / 3.0)));
    let x = index % side;
    let y = (index / side) % side;
    let z = index / (side * side);
    let center = (f32(side) - 1.0) * 0.5;
    let position = (vec3<f32>(f32(x), f32(y), f32(z)) - vec3<f32>(center)) * 1.5;
    let scale = 0.4;
    let model = mat4x4<f32>(
        vec4<f32>(scale, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, scale, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, scale, 0.0),
        vec4<f32>(position, 1.0),
    );

    let palette_index = index % 6u;
    var base_color = vec4<f32>(0.78, 0.28, 0.22, 1.0);
    switch palette_index {
        case 1u: { base_color = vec4<f32>(0.22, 0.34, 0.82, 1.0); }
        case 2u: { base_color = vec4<f32>(0.20, 0.72, 0.34, 1.0); }
        case 3u: { base_color = vec4<f32>(0.82, 0.74, 0.20, 1.0); }
        case 4u: { base_color = vec4<f32>(0.74, 0.20, 0.74, 1.0); }
        case 5u: { base_color = vec4<f32>(0.18, 0.78, 0.78, 1.0); }
        default: { base_color = vec4<f32>(0.78, 0.28, 0.22, 1.0); }
    }

    instances[index].model = model;
    instances[index].previous_model = model;
    instances[index].base_color = base_color;
    instances[index].material = vec4<f32>(0.2, 0.4, 1.0, 0.0);
}
