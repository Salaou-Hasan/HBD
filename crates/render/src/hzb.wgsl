// Hierarchical Z buffer (max-depth reduction) for conservative occlusion tests.
// Mip 0 is built from the previous frame's depth (2x2 max), then each further
// mip is a 2x2 max of the level below. Max-depth is used so a cluster is only
// rejected when every sampled texel is in front of its nearest point.

struct LevelParams {
    src_mip: u32,
    src_width: u32,
    src_height: u32,
    pad: u32,
};

@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var hzb_texture: texture_2d<f32>;
@group(0) @binding(2) var hzb_output: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> level: LevelParams;

fn clamp_coord(coord: vec2<i32>) -> vec2<i32> {
    return clamp(
        coord,
        vec2<i32>(0, 0),
        vec2<i32>(i32(level.src_width) - 1, i32(level.src_height) - 1),
    );
}

@compute @workgroup_size(8, 8)
fn from_depth_cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base = vec2<i32>(global_id.xy) * 2;
    var maximum = 0.0;
    for (var y = 0; y < 2; y = y + 1) {
        for (var x = 0; x < 2; x = x + 1) {
            let coord = clamp_coord(base + vec2<i32>(x, y));
            maximum = max(maximum, textureLoad(depth_texture, coord, 0));
        }
    }
    textureStore(hzb_output, vec2<i32>(global_id.xy), vec4<f32>(maximum, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn reduce_cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base = vec2<i32>(global_id.xy) * 2;
    var maximum = 0.0;
    for (var y = 0; y < 2; y = y + 1) {
        for (var x = 0; x < 2; x = x + 1) {
            let coord = clamp_coord(base + vec2<i32>(x, y));
            maximum = max(maximum, textureLoad(hzb_texture, coord, i32(level.src_mip)).r);
        }
    }
    textureStore(hzb_output, vec2<i32>(global_id.xy), vec4<f32>(maximum, 0.0, 0.0, 0.0));
}
