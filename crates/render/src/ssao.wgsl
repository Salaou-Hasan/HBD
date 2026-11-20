@group(0) @binding(0) var depth_sampler: sampler;
@group(0) @binding(1) var depth_texture: texture_depth_2d;

struct SsaoParams {
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    // x: radius, y: bias, z: intensity, w: unused
    settings: vec4<f32>,
};
@group(0) @binding(2) var<uniform> params: SsaoParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let corner = corners[index];

    var output: VertexOutput;
    output.position = vec4<f32>(corner, 0.0, 1.0);
    output.uv = vec2<f32>(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
    return output;
}

fn read_depth(uv: vec2<f32>) -> f32 {
    return textureSample(depth_texture, depth_sampler, uv);
}

fn reconstruct(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let view_position = params.inverse_projection * ndc;
    return view_position.xyz / view_position.w;
}

fn hash(value: vec2<f32>) -> f32 {
    return fract(sin(dot(value, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let depth = read_depth(input.uv);
    if (depth >= 0.9999) {
        return vec4<f32>(1.0);
    }

    let view_position = reconstruct(input.uv, depth);
    let normal = normalize(cross(dpdx(view_position), dpdy(view_position)));
    let radius = params.settings.x;
    let bias = params.settings.y;

    var occlusion = 0.0;
    for (var i: i32 = 0; i < 16; i = i + 1) {
        let fi = f32(i);
        let r1 = hash(input.uv + vec2<f32>(fi * 0.137, 0.0)) * 2.0 - 1.0;
        let r2 = hash(input.uv.yx + vec2<f32>(0.0, fi * 0.219)) * 2.0 - 1.0;
        let r3 = hash(input.uv + vec2<f32>(fi * 0.071, fi * 0.193)) * 2.0 - 1.0;

        var direction = normalize(normal + vec3<f32>(r1, r2, r3));
        if (dot(direction, normal) < 0.0) {
            direction = -direction;
        }

        let scale = radius * (0.15 + 0.85 * (fi / 16.0));
        let sample_position = view_position + direction * scale;
        let clip = params.projection * vec4<f32>(sample_position, 1.0);
        if (clip.w <= 0.0) {
            continue;
        }

        let sample_uv = vec2<f32>(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        let sample_depth = read_depth(sample_uv);
        let sample_view = reconstruct(sample_uv, sample_depth);
        let delta = sample_view - view_position;
        let distance = length(delta);
        let range_weight = clamp(1.0 - distance / radius, 0.0, 1.0);
        let occluded = select(0.0, 1.0, sample_view.z > view_position.z + bias);
        occlusion = occlusion + occluded * range_weight;
    }

    let ao = clamp(1.0 - occlusion / 16.0 * params.settings.z, 0.0, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
}
