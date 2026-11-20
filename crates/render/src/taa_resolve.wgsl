@group(0) @binding(0) var taa_sampler: sampler;
@group(0) @binding(1) var current_texture: texture_2d<f32>;
@group(0) @binding(2) var history_texture: texture_2d<f32>;
@group(0) @binding(3) var depth_texture: texture_depth_2d;

struct TaaParams {
    inverse_view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
    // x: blend, y/z: texel size, w: unused
    settings: vec4<f32>,
};
@group(0) @binding(4) var<uniform> params: TaaParams;

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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let current = textureSample(current_texture, taa_sampler, input.uv).rgb;
    let depth = textureSample(depth_texture, taa_sampler, input.uv);

    var history = current;
    if (depth < 0.9999) {
        let ndc = vec4<f32>(input.uv.x * 2.0 - 1.0, 1.0 - input.uv.y * 2.0, depth, 1.0);
        let world = params.inverse_view_projection * ndc;
        let world_position = world.xyz / world.w;
        let previous_clip = params.previous_view_projection * vec4<f32>(world_position, 1.0);

        if (previous_clip.w > 0.0) {
            let previous_uv = vec2<f32>(
                previous_clip.x / previous_clip.w * 0.5 + 0.5,
                0.5 - previous_clip.y / previous_clip.w * 0.5,
            );
            if (previous_uv.x >= 0.0 && previous_uv.x <= 1.0
                && previous_uv.y >= 0.0 && previous_uv.y <= 1.0)
            {
                history = textureSample(history_texture, taa_sampler, previous_uv).rgb;
            }
        }
    }

    // Neighborhood clamping removes ghosting from disocclusions.
    var minimum = current;
    var maximum = current;
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y))
                * vec2<f32>(params.settings.y, params.settings.z);
            let neighbor = textureSample(current_texture, taa_sampler, input.uv + offset).rgb;
            minimum = min(minimum, neighbor);
            maximum = max(maximum, neighbor);
        }
    }
    history = clamp(history, minimum, maximum);

    return vec4<f32>(mix(current, history, params.settings.x), 1.0);
}
