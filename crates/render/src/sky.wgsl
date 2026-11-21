// Must match `FrameUniform` in mesh.rs member-for-member: the sky pipeline
// shares the same descriptor set and uniform buffer as the mesh pass.
struct Frame {
    view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    light_vp0: mat4x4<f32>,
    light_vp1: mat4x4<f32>,
    light_vp2: mat4x4<f32>,
    eye_position: vec4<f32>,
    light_direction: vec4<f32>,
    light_color: vec4<f32>,
    shadow_splits: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;

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

fn sky_color(ray: vec3<f32>, sun_direction: vec3<f32>) -> vec3<f32> {
    let t = clamp(ray.y * 0.5 + 0.5, 0.0, 1.0);
    let horizon = vec3<f32>(0.55, 0.62, 0.78);
    let zenith = vec3<f32>(0.04, 0.14, 0.42);
    let base = mix(horizon, zenith, t) * 1.3;
    let sun = pow(max(dot(ray, sun_direction), 0.0), 900.0) * 14.0;
    let glow = pow(max(dot(ray, sun_direction), 0.0), 16.0) * 0.8;
    return base + vec3<f32>(sun + glow);
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) velocity: vec4<f32>,
};

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    let ndc = vec4<f32>(input.uv.x * 2.0 - 1.0, 1.0 - input.uv.y * 2.0, 1.0, 1.0);
    let far_point = frame.inverse_view_projection * ndc;
    let ray = normalize(far_point.xyz / far_point.w - frame.eye_position.xyz);
    let color = sky_color(ray, normalize(frame.light_direction.xyz));
    // Sky does not write depth, so HBTR never samples its history, but writing
    // zero keeps the second attachment defined instead of undefined.
    return FragmentOutput(vec4<f32>(color, 1.0), vec4<f32>(0.0));
}
