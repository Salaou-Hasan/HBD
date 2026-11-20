// GPU frustum culling + compaction for the single indirect draw path.
//
// One invocation per source instance. Visible instances are compacted into a
// second storage buffer with an atomic counter, and that counter doubles as
// the `instanceCount` field of the indirect draw arguments. No CPU readback:
// the draw consumes the count directly.
//
// Instance layout matches mesh.wgsl and the Rust `Instance` struct (96 bytes).

struct Instance {
    model: mat4x4<f32>,
    previous_model: mat4x4<f32>,
    base_color: vec4<f32>,
    material: vec4<f32>,
};

struct CullUniform {
    // Six world-space planes, xyz = normal, w = distance. Inside: dot(n,p)+w >= 0.
    planes: array<vec4<f32>, 6>,
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> cull: CullUniform;
@group(0) @binding(1) var<storage, read> sources: array<Instance>;
@group(0) @binding(2) var<storage, read_write> visible: array<Instance>;
// Same buffer as the indirect draw args: [indexCount, instanceCount, ...].
@group(0) @binding(3) var<storage, read_write> draw_args: array<atomic<u32>>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= cull.count) {
        return;
    }

    let instance = sources[index];
    let center = instance.model[3].xyz;
    let sx = length(instance.model[0].xyz);
    let sy = length(instance.model[1].xyz);
    let sz = length(instance.model[2].xyz);
    // sqrt(3)/2 bounds the corner of a unit cube under non-uniform scale.
    let radius = max(max(sx, sy), sz) * 0.8660254;

    var inside = true;
    for (var plane_index = 0u; plane_index < 6u; plane_index = plane_index + 1u) {
        let plane = cull.planes[plane_index];
        if (dot(plane.xyz, center) + plane.w < -radius) {
            inside = false;
            break;
        }
    }

    if (inside) {
        let slot = atomicAdd(&draw_args[1], 1u);
        visible[slot] = instance;
    }
}
