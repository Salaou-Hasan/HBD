// HBD Temporal Resolve (HBTR).
//
// Original variant built from the public TAA grammar: jittered sampling,
// reprojection, YCoCg variance clipping (Karis / Salvi), luminance-adaptive
// accumulation, and a motion+clip confidence blend. Unlike the old HBD pass it
// does not use a fixed history weight, and it sharpens from the same 3x3
// neighborhood used for clipping, so AA costs no extra texture fetches.

@group(0) @binding(0) var taa_sampler: sampler;
@group(0) @binding(1) var current_texture: texture_2d<f32>;
@group(0) @binding(2) var history_texture: texture_2d<f32>;
@group(0) @binding(3) var depth_texture: texture_depth_2d;

struct TaaParams {
    inverse_view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
    // x: base history blend, y/z: inverse full-res size, w: unused
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

fn rgb_to_ycocg(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * color.r + 0.5 * color.g + 0.25 * color.b,
        0.5 * color.r - 0.5 * color.b,
        -0.25 * color.r + 0.5 * color.g - 0.25 * color.b,
    );
}

fn ycocg_to_rgb(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        color.x + color.y - color.z,
        color.x,
        color.x - color.y - color.z,
    );
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Intersect the segment [average, value] with the variance box. This keeps the
// resolved color from collecting in AABB corners the way a hard clamp does.
fn clip_aabb(box_min: vec3<f32>, box_max: vec3<f32>, average: vec3<f32>, value: vec3<f32>) -> vec3<f32> {
    let ray = value - average;
    var t = 1.0;
    for (var axis: i32 = 0; axis < 3; axis = axis + 1) {
        if (ray[axis] > 0.0) {
            let limit = box_max[axis] - average[axis];
            if (limit <= 0.0) {
                t = 0.0;
            } else {
                t = min(t, limit / ray[axis]);
            }
        } else if (ray[axis] < 0.0) {
            let limit = box_min[axis] - average[axis];
            if (limit >= 0.0) {
                t = 0.0;
            } else {
                t = min(t, limit / ray[axis]);
            }
        }
    }
    return average + ray * t;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = params.settings.yz;
    let current = textureSample(current_texture, taa_sampler, input.uv).rgb;
    let depth = textureSample(depth_texture, taa_sampler, input.uv);

    // 3x3 neighborhood in YCoCg: moments, extremes, and an average for sharpening.
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var neighbor_min = vec3<f32>(1.0e9);
    var neighbor_max = vec3<f32>(-1.0e9);
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let neighbor = rgb_to_ycocg(
                textureSample(current_texture, taa_sampler, input.uv + vec2<f32>(f32(x), f32(y)) * texel).rgb,
            );
            m1 = m1 + neighbor;
            m2 = m2 + neighbor * neighbor;
            neighbor_min = min(neighbor_min, neighbor);
            neighbor_max = max(neighbor_max, neighbor);
        }
    }
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
    let box_min = max(mean - sigma, neighbor_min);
    let box_max = min(mean + sigma, neighbor_max);
    let neighborhood_average = ycocg_to_rgb(mean);

    // Reproject the current world position into the history image.
    var history = current;
    var motion_confidence = 1.0;
    var clip_confidence = 0.0;
    if (depth < 0.9999) {
        let ndc = vec4<f32>(input.uv.x * 2.0 - 1.0, 1.0 - input.uv.y * 2.0, depth, 1.0);
        let world = params.inverse_view_projection * ndc;
        let world_position = world.xyz / world.w;
        let previous_clip = params.previous_view_projection * vec4<f32>(world_position, 1.0);
        if (previous_clip.w > 0.0) {
            let history_uv = vec2<f32>(
                previous_clip.x / previous_clip.w * 0.5 + 0.5,
                0.5 - previous_clip.y / previous_clip.w * 0.5,
            );
            if (history_uv.x >= 0.0 && history_uv.x <= 1.0
                && history_uv.y >= 0.0 && history_uv.y <= 1.0)
            {
                let raw_history = textureSample(history_texture, taa_sampler, history_uv).rgb;
                let clipped_ycocg = clip_aabb(
                    box_min,
                    box_max,
                    mean,
                    rgb_to_ycocg(raw_history),
                );
                history = ycocg_to_rgb(clipped_ycocg);

                let clip_distance = length(clipped_ycocg - rgb_to_ycocg(raw_history));
                let pixels_moved = length((history_uv - input.uv) / texel);
                clip_confidence = clamp(clip_distance * 6.0, 0.0, 1.0);
                motion_confidence = clamp(pixels_moved * 0.04, 0.0, 1.0);
                motion_confidence = max(motion_confidence, clip_confidence);
            }
        }
    }

    // Confidence lowers history weight when reprojection is unreliable.
    let blend = mix(params.settings.x, 0.5, motion_confidence);
    // Recover contrast lost to temporal filtering, reusing the neighborhood mean.
    let sharpened_current = current + (current - neighborhood_average) * 0.12;
    let source_weight = (1.0 - blend) / (1.0 + luminance(sharpened_current));
    let history_weight = blend / (1.0 + luminance(history));
    let resolved = (sharpened_current * source_weight + history * history_weight)
        / max(source_weight + history_weight, 0.00001);

    return vec4<f32>(resolved, 1.0);
}
