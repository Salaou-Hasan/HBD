// Half-resolution bilateral denoise for the stochastic SSAO buffer.
@group(0) @binding(0) var ao_sampler: sampler;
@group(0) @binding(1) var ao_texture: texture_2d<f32>;
@group(0) @binding(2) var depth_texture: texture_depth_2d;

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
    let texel = 1.0 / vec2<f32>(textureDimensions(ao_texture));
    let center_depth = textureSample(depth_texture, ao_sampler, input.uv);
    var sum = textureSample(ao_texture, ao_sampler, input.uv).r;
    var total_weight = 1.0;

    let directions = array<vec2<f32>, 4>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(-1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    for (var radius: i32 = 1; radius <= 2; radius = radius + 1) {
        let spatial_weight = 1.0 / (1.0 + f32(radius) * 0.8);
        for (var i: i32 = 0; i < 4; i = i + 1) {
            let uv = input.uv + directions[i] * texel * f32(radius);
            let neighbor_depth = textureSample(depth_texture, ao_sampler, uv);
            let depth_weight = exp(-abs(neighbor_depth - center_depth) * 2000.0);
            let weight = depth_weight * spatial_weight;
            sum = sum + textureSample(ao_texture, ao_sampler, uv).r * weight;
            total_weight = total_weight + weight;
        }
    }

    let ao = clamp(sum / total_weight, 0.0, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
}
