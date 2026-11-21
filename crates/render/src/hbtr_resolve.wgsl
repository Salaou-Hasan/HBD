// HBD Temporal Resolve (HBTR).
//
// One shader for every hardware tier. Work that only helps high-end GPUs is
// gated by `params.settings2.x` (the tier), so an entry-level GPU skips it at
// runtime without a second shader permutation. The visual grammar is identical
// on all tiers: jitter -> reproject -> rectify history -> confidence blend ->
// adaptive sharpen.

@group(0) @binding(0) var hbtr_sampler: sampler;
@group(0) @binding(1) var current_texture: texture_2d<f32>;
@group(0) @binding(2) var history_texture: texture_2d<f32>;
@group(0) @binding(3) var depth_texture: texture_depth_2d;

struct HbtrParams {
    inverse_view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
    // x: base history blend, y/z: inverse input texel size, w: history-valid flag.
    settings: vec4<f32>,
    // x: tier, y: base sharpen, z: reactive strength, w: reserved.
    settings2: vec4<f32>,
    // x/y: output/input scale (1,1 = plain HBTR), z/w: reserved.
    settings3: vec4<f32>,
};
@group(0) @binding(4) var<uniform> params: HbtrParams;
@group(0) @binding(5) var velocity_texture: texture_2d<f32>;

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

// Cheap variance box for the Low tier. Higher tiers use the k-DOP below.
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
    return average + ray * clamp(t, 0.0, 1.0);
}

struct FragmentOutput {
    @location(0) resolved: vec4<f32>,
    // Written in the same draw as `resolved`: no separate copy pass.
    @location(1) history: vec4<f32>,
};

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    let texel = params.settings.yz;
    let tier = params.settings2.x;
    let scale = params.settings3.xy;
    let upscaling = scale.x > 1.01 || scale.y > 1.01;
    var current = textureSample(current_texture, hbtr_sampler, input.uv).rgb;
    let depth = textureSample(depth_texture, hbtr_sampler, input.uv);

    // 3x3 neighborhood in YCoCg: moments, extremes, sharpening average. The
    // samples are aligned to the input pixel grid so the same nine fetches can
    // reconstruct the current high-resolution sample when upscaling.
    let input_coord = input.uv / texel;
    let input_base = floor(input_coord);
    var c00 = current;
    var c10 = current;
    var c01 = current;
    var c11 = current;
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var neighbor_min = vec3<f32>(1.0e9);
    var neighbor_max = vec3<f32>(-1.0e9);
    let dirs = array<vec3<f32>, 7>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.5773503, 0.5773503, 0.5773503),
        vec3<f32>(0.5773503, 0.5773503, -0.5773503),
        vec3<f32>(0.5773503, -0.5773503, 0.5773503),
        vec3<f32>(-0.5773503, 0.5773503, 0.5773503),
    );
    // Medium: three axes + one diagonal = 8 planes. High/Epic: 14 planes.
    let planes = select(4u, 7u, tier >= 2.0);
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let sample_uv = (input_base + vec2<f32>(f32(x), f32(y)) + 0.5) * texel;
            let sample_rgb = textureSample(current_texture, hbtr_sampler, sample_uv).rgb;
            if (x == 0 && y == 0) {
                c00 = sample_rgb;
            } else if (x == 1 && y == 0) {
                c10 = sample_rgb;
            } else if (x == 0 && y == 1) {
                c01 = sample_rgb;
            } else if (x == 1 && y == 1) {
                c11 = sample_rgb;
            }
            let neighbor = rgb_to_ycocg(sample_rgb);
            m1 = m1 + neighbor;
            m2 = m2 + neighbor * neighbor;
            neighbor_min = min(neighbor_min, neighbor);
            neighbor_max = max(neighbor_max, neighbor);
        }
    }
    let mean = m1 / 9.0;
    // When upscaling, the neighborhood comes from the low-res input while the
    // history is high-res. Widen the rectification box so valid detail is not
    // clipped away as if it were a ghost.
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)))
        * select(1.0, 2.5, upscaling);
    var box_min = max(mean - sigma, neighbor_min);
    var box_max = min(mean + sigma, neighbor_max);
    if (upscaling) {
        box_min = mean - sigma;
        box_max = mean + sigma;
    }
    let neighborhood_average = ycocg_to_rgb(mean);
    if (upscaling) {
        // Reconstruct the current sample on the input grid from the four taps
        // already fetched above. The detail gain over a plain upscale comes
        // mostly from accumulating the output-resolution history over frames.
        let f = clamp(input_coord - input_base, vec2<f32>(0.0), vec2<f32>(1.0));
        current = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    }

    var kdop_min = array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var kdop_max = array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if (tier >= 1.0) {
        // Analytic variance planes: derive each k-DOP bound from the same mean
        // and sigma used by the box clip. No per-sample plane loop, so the
        // higher tiers cost a handful of dots instead of 36-63 iterations.
        for (var d: u32 = 0u; d < planes; d = d + 1u) {
            let center_projection = dot(mean, dirs[d]);
            let sigma_projection = dot(sigma, abs(dirs[d]));
            kdop_min[d] = center_projection - sigma_projection;
            kdop_max[d] = center_projection + sigma_projection;
            // Keep every plane inside the real neighborhood colour range.
            // Without this, diagonal planes can invent chroma at edges.
            let positive = max(dirs[d], vec3<f32>(0.0));
            let negative = min(dirs[d], vec3<f32>(0.0));
            kdop_min[d] = max(kdop_min[d], dot(neighbor_min, positive) + dot(neighbor_max, negative));
            kdop_max[d] = min(kdop_max[d], dot(neighbor_max, positive) + dot(neighbor_min, negative));
        }
    }

    // Reactive mask. The velocity fetch stays inside the history branch so the
    // Low tier pays only for the one fetch the reprojection already needed.
    var reactive = 0.0;

    // Parallax disocclusion: a large depth spread around this pixel means a
    // nearby object is revealing background; history must be weakened. Only
    // High/Epic pay for the four extra depth taps.
    var parallax = 0.0;
    if (tier >= 2.0) {
        let d0 = textureSample(depth_texture, hbtr_sampler, input.uv + vec2<f32>(texel.x, 0.0));
        let d1 = textureSample(depth_texture, hbtr_sampler, input.uv - vec2<f32>(texel.x, 0.0));
        let d2 = textureSample(depth_texture, hbtr_sampler, input.uv + vec2<f32>(0.0, texel.y));
        let d3 = textureSample(depth_texture, hbtr_sampler, input.uv - vec2<f32>(0.0, texel.y));
        let spread = max(max(d0, d1), max(d2, d3)) - min(min(d0, d1), min(d2, d3));
        parallax = clamp(spread * 140.0, 0.0, 1.0);
    }

    // Reproject the current world position into this image's own history slot.
    var history = current;
    var confidence = max(reactive, parallax);
    // `settings.w` is the per-image history-valid flag. A freshly allocated or
    // post-resize history slot is undefined and must never be sampled.
    if (params.settings.w > 0.5 && depth < 0.9999) {
        let ndc = vec4<f32>(input.uv.x * 2.0 - 1.0, 1.0 - input.uv.y * 2.0, depth, 1.0);
        let world = params.inverse_view_projection * ndc;
        let world_position = world.xyz / world.w;
        let previous_clip = params.previous_view_projection * vec4<f32>(world_position, 1.0);
        if (previous_clip.w > 0.0) {
            // Object motion in UV units; camera motion is already handled by
            // the reprojection above, so this never double counts.
            let velocity = textureSample(velocity_texture, hbtr_sampler, input.uv).xy;
            if (tier >= 1.0) {
                reactive = clamp(
                    length(velocity / max(texel, vec2<f32>(1.0e-6))) * 0.06 * params.settings2.z,
                    0.0,
                    1.0,
                );
                confidence = max(confidence, reactive);
            }
            var history_uv = vec2<f32>(
                previous_clip.x / previous_clip.w * 0.5 + 0.5,
                0.5 - previous_clip.y / previous_clip.w * 0.5,
            );
            history_uv = history_uv - velocity;
            if (history_uv.x >= 0.0 && history_uv.x <= 1.0
                && history_uv.y >= 0.0 && history_uv.y <= 1.0)
            {
                let raw_history = rgb_to_ycocg(
                    textureSample(history_texture, hbtr_sampler, history_uv).rgb,
                );
                // The axis-aligned variance clip stays on every tier: it is
                // what prevents stale jitter-phase colours from bleeding at
                // edges. k-DOP is an additional rejection signal, never a
                // replacement for this clip (diagonal clipping shifts chroma).
                let clipped = clip_aabb(box_min, box_max, mean, raw_history);
                let clip_distance = length(clipped - raw_history);
                var kdop_confidence = 0.0;
                if (tier >= 1.0) {
                    let ray = raw_history - mean;
                    var t = 1.0;
                    for (var d: u32 = 0u; d < planes; d = d + 1u) {
                        let denominator = dot(dirs[d], ray);
                        let center_projection = dot(dirs[d], mean);
                        if (denominator > 0.0) {
                            t = min(t, (kdop_max[d] - center_projection) / denominator);
                        } else if (denominator < 0.0) {
                            t = min(t, (kdop_min[d] - center_projection) / denominator);
                        }
                    }
                    let bounded = mean + ray * clamp(t, 0.0, 1.0);
                    kdop_confidence = clamp(length(raw_history - bounded) * 6.0, 0.0, 1.0);
                }
                history = ycocg_to_rgb(clipped);
                let pixels_moved = length((history_uv - input.uv) / texel);
                confidence = max(confidence, clamp(clip_distance * 6.0, 0.0, 1.0));
                confidence = max(confidence, clamp(pixels_moved * 0.04, 0.0, 1.0));
                // k-DOP is a *dynamic* rejection signal. At rest, jitter alone
                // makes it fire at edges and lowers the blend enough to expose
                // coloured subpixel fringes, so only use it when velocity or
                // parallax says the frame is actually changing.
                if (reactive > 0.0 || parallax > 0.0) {
                    confidence = max(confidence, kdop_confidence);
                }
            }
        }
    }

    // Confidence lowers history weight when reprojection or shading is unreliable.
    let blend = mix(params.settings.x, 0.5, clamp(confidence, 0.0, 1.0));

    // Contrast-adaptive sharpening: flat areas keep the conservative base value,
    // high-contrast edges get more recovery without inventing halos in flat areas.
    var sharpen = params.settings2.y;
    if (tier >= 1.0) {
        let contrast = max(
            max(neighbor_max.x - neighbor_min.x, neighbor_max.y - neighbor_min.y),
            neighbor_max.z - neighbor_min.z,
        );
        sharpen = mix(sharpen, sharpen * 1.15, clamp(contrast * 2.0, 0.0, 1.0));
    }
    // Luminance-only unsharp mask: per-channel sharpening on a low-res/upscaled
    // image produces coloured fringes at edges.
    let luma_delta = luminance(current) - luminance(neighborhood_average);
    let sharpened_current = current + vec3<f32>(luma_delta * sharpen);

    let source_weight = (1.0 - blend) / max(1.0 + luminance(sharpened_current), 0.00001);
    let history_weight = blend / max(1.0 + luminance(history), 0.00001);
    let resolved = (sharpened_current * source_weight + history * history_weight)
        / max(source_weight + history_weight, 0.00001);

    let output = vec4<f32>(resolved, 1.0);
    return FragmentOutput(output, output);
}
