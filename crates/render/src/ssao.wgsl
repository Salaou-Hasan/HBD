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

// Interleaved gradient noise: only rotates the sample pattern, so it cannot
// inject white-noise variance into the AO value the way a sin() hash does.
fn rotation(uv: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(uv, vec2<f32>(0.06711056, 0.00583715))));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let depth = read_depth(input.uv);
    if (depth >= 0.9999) {
        return vec4<f32>(1.0);
    }

    let view_position = reconstruct(input.uv, depth);
    // Framebuffer y grows downward, so this order points the hemisphere out
    // of the surface (toward the camera) rather than into it.
    let normal = normalize(cross(dpdy(view_position), dpdx(view_position)));
    let radius = params.settings.x;
    let bias = params.settings.y;

    // Tangent basis for a stable hemisphere. The fallback avoids a degenerate
    // cross product when the surface faces the camera directly.
    var tangent = cross(normal, vec3<f32>(0.0, 0.0, 1.0));
    if (length(tangent) < 0.001) {
        tangent = cross(normal, vec3<f32>(0.0, 1.0, 0.0));
    }
    tangent = normalize(tangent);
    let bitangent = cross(normal, tangent);

    // Golden-angle spiral: 16 samples cover the hemisphere evenly instead of
    // clustering randomly. A cheap per-pixel rotation decorrelates neighbours
    // while keeping the variance low enough for the bilateral denoise.
    let golden_angle = 2.39996323;
    let spin = rotation(input.uv * vec2<f32>(textureDimensions(depth_texture))) * 6.2831853;
    var occlusion = 0.0;
    var total_weight = 0.0;
    for (var i: i32 = 0; i < 16; i = i + 1) {
        let fi = f32(i);
        let r = sqrt((fi + 0.5) / 16.0);
        let theta = fi * golden_angle + spin;
        let local = vec3<f32>(
            cos(theta) * r,
            sin(theta) * r,
            sqrt(max(0.0, 1.0 - r * r)),
        );
        let direction = normalize(tangent * local.x + bitangent * local.y + normal * local.z);

        let scale = radius * r;
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
        let scene_view = reconstruct(sample_uv, sample_depth);
        // Compare against the hemisphere sample we intended to test, not the
        // centre pixel: the latter reports depth discontinuities as occlusion
        // and is the source of the speckle on flat surfaces.
        let occluded = select(0.0, 1.0, scene_view.z > sample_position.z + bias);
        let range_weight = clamp(1.0 - scale / radius, 0.0, 1.0);
        occlusion = occlusion + occluded * range_weight;
        total_weight = total_weight + range_weight;
    }

    let ao = clamp(1.0 - occlusion / max(total_weight, 0.001) * params.settings.z, 0.0, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
}
