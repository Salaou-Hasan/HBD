// Cluster-level culling. One cluster covers CLUSTER_SIZE consecutive instances.
// Stage 1 (init_cs, once per scene): conservative bounding sphere per cluster.
// Stage 2 (cluster_cull_cs, per frame): visibility flag per cluster, tested
// against the same six camera frustum planes used by the instance cull.
//
// The instance cull (cull.wgsl) consults these flags before testing individual
// instances, so whole off-screen clusters are skipped with one sphere test.

const CLUSTER_SIZE: u32 = 256u;

struct Instance {
    model: mat4x4<f32>,
    base_color: vec4<f32>,
    material: vec4<f32>,
};

struct CullUniform {
    planes: array<vec4<f32>, 6>,
    count: u32,
    cluster_count: u32,
    pad: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> sources: array<Instance>;
@group(0) @binding(1) var<storage, read_write> bounds: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> cull: CullUniform;
// One flag per cluster, written once per frame and read by cull.wgsl.
@group(0) @binding(3) var<storage, read_write> flags: array<u32>;

fn inside_frustum(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = cull.planes[i];
        if (dot(plane.xyz, center) + plane.w < -radius) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(1)
fn init_cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cluster = global_id.x;
    if (cluster >= cull.cluster_count) {
        return;
    }
    let start = cluster * CLUSTER_SIZE;
    let end = min(start + CLUSTER_SIZE, cull.count);
    var minimum = vec3<f32>(1.0e30);
    var maximum = vec3<f32>(-1.0e30);
    for (var i = start; i < end; i = i + 1u) {
        let instance = sources[i];
        let center = instance.model[3].xyz;
        let scale = max(
            max(length(instance.model[0].xyz), length(instance.model[1].xyz)),
            length(instance.model[2].xyz),
        );
        let radius = scale * 0.8660254;
        minimum = min(minimum, center - vec3<f32>(radius));
        maximum = max(maximum, center + vec3<f32>(radius));
    }
    let center = (minimum + maximum) * 0.5;
    let radius = length(maximum - center);
    bounds[cluster] = vec4<f32>(center, radius);
}

@compute @workgroup_size(64)
fn cluster_cull_cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cluster = global_id.x;
    if (cluster >= cull.cluster_count) {
        return;
    }
    let sphere = bounds[cluster];
    let visible = inside_frustum(sphere.xyz, sphere.w);
    flags[cluster] = select(0u, 1u, visible);
}
