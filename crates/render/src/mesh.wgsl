const PI: f32 = 3.14159265;

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

struct Instance {
    model: mat4x4<f32>,
    previous_model: mat4x4<f32>,
    base_color: vec4<f32>,
    material: vec4<f32>, // metallic, roughness, ao, unused
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var shadow_map0: texture_depth_2d;
@group(0) @binding(3) var shadow_map1: texture_depth_2d;
@group(0) @binding(4) var shadow_map2: texture_depth_2d;
@group(0) @binding(5) var shadow_sampler: sampler_comparison;
@group(0) @binding(6) var material_textures: texture_2d_array<f32>;
@group(0) @binding(7) var material_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) base_color: vec3<f32>,
    @location(3) light_clip0: vec4<f32>,
    @location(4) light_clip1: vec4<f32>,
    @location(5) light_clip2: vec4<f32>,
    @location(6) view_distance: f32,
    @location(7) material: vec4<f32>,
    // NDC is linear in screen space, so interpolate it linearly instead of
    // perspective-correcting clip coordinates before the divide.
    @location(8) @interpolate(linear) current_ndc: vec2<f32>,
    @location(9) @interpolate(linear) previous_ndc: vec2<f32>,
};

@vertex
fn vs_main(
    input: VertexInput,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let base_count = max(u32(frame.shadow_splits.w), 1u);
    let instance = instances[instance_index % base_count];
    let world_position = instance.model * vec4<f32>(input.position, 1.0);

    var output: VertexOutput;
    output.world_position = world_position.xyz;
    output.world_normal = normalize((instance.model * vec4<f32>(input.normal, 0.0)).xyz);
    output.base_color = instance.base_color.rgb;
    output.material = instance.material;
    // Object motion only: both endpoints use the same view-projection, so the
    // camera part cancels and HBTR's depth reprojection owns camera motion.
    let current_clip = frame.view_projection * world_position;
    let previous_clip =
        frame.view_projection * instance.previous_model * vec4<f32>(input.position, 1.0);
    output.current_ndc = current_clip.xy / current_clip.w;
    output.previous_ndc = previous_clip.xy / previous_clip.w;
    output.light_clip0 = frame.light_vp0 * world_position;
    output.light_clip1 = frame.light_vp1 * world_position;
    output.light_clip2 = frame.light_vp2 * world_position;
    output.view_distance = length(frame.eye_position.xyz - world_position.xyz);
    output.clip_position = current_clip;
    return output;
}

fn sample_shadow(map: texture_depth_2d, light_clip: vec4<f32>) -> f32 {
    let clip = light_clip.xyz / light_clip.w;
    if (clip.x < -1.0 || clip.x > 1.0 || clip.y < -1.0 || clip.y > 1.0 || clip.z > 1.0) {
        return 1.0;
    }

    let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    let bias = clip.z - 0.0015;
    let texel = 1.0 / 1024.0;
    var shadow = 0.0;

    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            shadow = shadow + textureSampleCompare(map, shadow_sampler, uv + offset, bias);
        }
    }

    return shadow / 9.0;
}

fn cascade_shadow(input: VertexOutput) -> f32 {
    if (input.view_distance < frame.shadow_splits.x) {
        return sample_shadow(shadow_map0, input.light_clip0);
    }
    if (input.view_distance < frame.shadow_splits.y) {
        return sample_shadow(shadow_map1, input.light_clip1);
    }
    return sample_shadow(shadow_map2, input.light_clip2);
}

fn environment_color(direction: vec3<f32>, sun_direction: vec3<f32>) -> vec3<f32> {
    let t = clamp(direction.y * 0.5 + 0.5, 0.0, 1.0);
    let horizon = vec3<f32>(0.55, 0.62, 0.78);
    let zenith = vec3<f32>(0.04, 0.14, 0.42);
    return mix(horizon, zenith, t) * 1.3
        + vec3<f32>(pow(max(dot(direction, sun_direction), 0.0), 16.0) * 0.5);
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denominator = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denominator * denominator, 0.0001);
}

fn geometry_schlick(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.0001);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) velocity: vec4<f32>,
};

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    let normal = normalize(input.world_normal);
    let light = normalize(frame.light_direction.xyz);
    let view = normalize(frame.eye_position.xyz - input.world_position);
    let half_vector = normalize(light + view);

    let metallic = input.material.x;
    let roughness = clamp(input.material.y, 0.05, 1.0);
    let ao = input.material.z;
    // Bindless material: instance material.w selects a texture array layer.
    let material_layer = min(u32(input.material.w), 3u);
    let texel = textureSample(
        material_textures,
        material_sampler,
        input.world_position.xz * 0.5,
        material_layer,
    );
    let albedo = input.base_color * texel.rgb;
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    let n_dot_v = max(dot(normal, view), 0.0001);
    let n_dot_l = max(dot(normal, light), 0.0);
    let n_dot_h = max(dot(normal, half_vector), 0.0);
    let v_dot_h = max(dot(view, half_vector), 0.0);

    let distribution = distribution_ggx(n_dot_h, roughness);
    let geometry = geometry_schlick(n_dot_v, roughness) * geometry_schlick(n_dot_l, roughness);
    let fresnel = fresnel_schlick(v_dot_h, f0);
    let specular_brdf = (distribution * geometry * fresnel)
        / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    let diffuse_brdf = (vec3<f32>(1.0) - fresnel) * (1.0 - metallic) * albedo / PI;

    let shadow = cascade_shadow(input);
    let direct = (diffuse_brdf + specular_brdf) * frame.light_color.rgb * n_dot_l * shadow;

    // Analytic image-based lighting: sky irradiance and a roughness-attenuated
    // environment reflection. This is not a convolved cubemap, but it obeys the
    // same energy split between diffuse and specular lobes.
    let irradiance = environment_color(normal, light) * frame.light_color.w;
    let diffuse_ibl = irradiance * albedo * (1.0 - metallic) * ao;
    let reflection = environment_color(reflect(-view, normal), light);
    let specular_ibl = reflection * f0 * (1.0 - roughness * 0.7) * ao;

    let color = direct + diffuse_ibl + specular_ibl;
    // Screen-space object motion in UV units. Y is flipped because image space
    // grows downward while NDC grows upward.
    let velocity = (input.current_ndc - input.previous_ndc) * vec2<f32>(0.5, -0.5);
    return FragmentOutput(vec4<f32>(color, 1.0), vec4<f32>(velocity, 0.0, 0.0));
}
