@group(0) @binding(0) var bloom_sampler: sampler;
@group(0) @binding(1) var source_texture: texture_2d<f32>;

struct BloomParams {
    // x: horizontal texel, y: vertical texel, z: threshold, w: unused
    data: vec4<f32>,
};
@group(0) @binding(2) var<uniform> params: BloomParams;

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

fn threshold(color: vec3<f32>) -> vec3<f32> {
    let luminance = max(max(color.r, color.g), color.b);
    let contribution = max(luminance - params.data.z, 0.0) / max(luminance, 0.0001);
    return color * contribution;
}

fn sample_blur(uv: vec2<f32>, step: vec2<f32>, apply_threshold: bool) -> vec3<f32> {
    let weights = array<f32, 5>(0.227027, 0.194595, 0.121622, 0.054054, 0.016216);
    var result = vec3<f32>(0.0);
    for (var i = 0; i < 5; i = i + 1) {
        let offset = step * f32(i);
        var sample_color = textureSample(source_texture, bloom_sampler, uv + offset).rgb;
        var negative = textureSample(source_texture, bloom_sampler, uv - offset).rgb;
        if (apply_threshold) {
            sample_color = threshold(sample_color);
            negative = threshold(negative);
        }
        result = result + (sample_color + negative) * weights[i];
    }
    return result;
}

@fragment
fn fs_horizontal(input: VertexOutput) -> @location(0) vec4<f32> {
    let step = vec2<f32>(params.data.x, 0.0);
    return vec4<f32>(sample_blur(input.uv, step, true), 1.0);
}

@fragment
fn fs_vertical(input: VertexOutput) -> @location(0) vec4<f32> {
    let step = vec2<f32>(0.0, params.data.y);
    return vec4<f32>(sample_blur(input.uv, step, false), 1.0);
}
