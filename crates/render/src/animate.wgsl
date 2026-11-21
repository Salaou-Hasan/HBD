// Motion-vector validation pass.
//
// Spins a bounded prefix of the compacted visible instances in place so the
// velocity attachment carries real object motion. The cull pass rebuilds the
// compacted buffer from the immutable source every frame, so writing the pose
// here never accumulates. Instances far larger than a prop (the ground slab)
// stay static: rotating them would sweep the whole scene.

struct Instance {
    model: mat4x4<f32>,
    previous_model: mat4x4<f32>,
    base_color: vec4<f32>,
    material: vec4<f32>,
};

struct AnimateParams {
    angle: f32,
    previous_angle: f32,
    count: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> params: AnimateParams;
@group(0) @binding(1) var<storage, read_write> instances: array<Instance>;

fn rotation_about(translation: vec3<f32>, angle: f32) -> mat4x4<f32> {
    let c = cos(angle);
    let s = sin(angle);
    let rotated = vec3<f32>(
        c * translation.x + s * translation.z,
        translation.y,
        -s * translation.x + c * translation.z,
    );
    let offset = translation - rotated;
    return mat4x4<f32>(
        vec4<f32>(c, 0.0, -s, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(s, 0.0, c, 0.0),
        vec4<f32>(offset, 1.0),
    );
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.count) {
        return;
    }

    var instance = instances[index];
    let scale = length(instance.model[0].xyz);
    if (scale > 2.0) {
        return;
    }

    let original = instance.model;
    let translation = original[3].xyz;
    instance.model = rotation_about(translation, params.angle) * original;
    instance.previous_model = rotation_about(translation, params.previous_angle) * original;
    instances[index] = instance;
}
