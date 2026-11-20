@group(0) @binding(0) var hdr_sampler: sampler;
@group(0) @binding(1) var hdr_texture: texture_2d<f32>;
@group(0) @binding(2) var bloom_texture: texture_2d<f32>;
@group(0) @binding(3) var ao_texture: texture_2d<f32>;

struct DebugParams {
    // 0: AgX final, 1: filmic, 2: HDR, 3: bloom, 4: AO,
    // 5: depth, 6: shadow cascade, 7: TAA history
    mode: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};
@group(0) @binding(4) var<uniform> debug_params: DebugParams;
@group(0) @binding(5) var depth_texture: texture_depth_2d;
@group(0) @binding(6) var shadow_texture: texture_depth_2d;
@group(0) @binding(7) var history_texture: texture_2d<f32>;

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

const AGX_INPUT: mat3x3<f32> = mat3x3<f32>(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772, 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104,
);
const AGX_OUTPUT: mat3x3<f32> = mat3x3<f32>(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989618447366072, 1.15107367264116,
);

fn agx(x: vec3<f32>) -> vec3<f32> {
    let transformed = max(AGX_INPUT * x, vec3<f32>(0.0));
    let logarithmic = log2(transformed + vec3<f32>(0.000001));
    let normalized = clamp(
        (logarithmic - vec3<f32>(-12.47393)) / vec3<f32>(16.499999),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let curve = normalized * normalized * (3.0 - 2.0 * normalized);
    return clamp(AGX_OUTPUT * curve, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn filmic(x: vec3<f32>) -> vec3<f32> {
    let a = x * (x + 0.0245786) - 0.000090537;
    let b = x * (0.983729 * x + 0.4329510) + 0.238081;
    return clamp(a / b, vec3<f32>(0.0), vec3<f32>(1.0));
}

// SSAO is denoised in its own half-res pass; the tonemap samples it once.
fn sample_ao(uv: vec2<f32>) -> f32 {
    return textureSample(ao_texture, hdr_sampler, uv).r;
}

const FPS_FONT: array<u32, 50> = array<u32, 50>(
    7u, 5u, 5u, 5u, 7u,
    2u, 6u, 2u, 2u, 7u,
    7u, 1u, 7u, 4u, 7u,
    7u, 1u, 3u, 1u, 7u,
    5u, 5u, 7u, 1u, 1u,
    7u, 4u, 7u, 1u, 7u,
    7u, 4u, 7u, 5u, 7u,
    7u, 1u, 1u, 1u, 1u,
    7u, 5u, 7u, 5u, 7u,
    7u, 5u, 7u, 1u, 7u
);

fn glyph_row(digit: u32, row: i32) -> u32 {
    if (digit > 9u || row < 0 || row > 4) {
        return 0u;
    }
    return FPS_FONT[digit * 5u + u32(row)];
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_texture, hdr_sampler, input.uv).rgb;
    let bloom = textureSample(bloom_texture, hdr_sampler, input.uv).rgb;
    let ao = sample_ao(input.uv);

    let lit = (hdr * ao + bloom * 0.12) * 0.9;
    var color = vec3<f32>(0.0);

    switch debug_params.mode {
        case 0u: { color = agx(lit); }
        case 1u: { color = filmic(lit); }
        case 2u: { color = hdr / (hdr + vec3<f32>(1.0)); }
        case 3u: { color = bloom; }
        case 4u: { color = vec3<f32>(ao); }
        case 5u: {
            let d = textureSample(depth_texture, hdr_sampler, input.uv);
            color = vec3<f32>(pow(d, 40.0));
        }
        case 6u: {
            let s = textureSample(shadow_texture, hdr_sampler, input.uv);
            color = vec3<f32>(s);
        }
        case 7u: { color = textureSample(history_texture, hdr_sampler, input.uv).rgb; }
        default: { color = agx(lit); }
    }

    let offset = input.uv - vec2<f32>(0.5, 0.5);
    let vignette = 1.0 - 0.35 * dot(offset, offset);

    let fps = bitcast<f32>(debug_params.fps_bits);
    if (fps > 0.0 && debug_params.mode == 0u) {
        let fps_int = u32(round(fps));
        let digits = array<u32, 4>(
            (fps_int / 1000u) % 10u,
            (fps_int / 100u) % 10u,
            (fps_int / 10u) % 10u,
            fps_int % 10u,
        );
        let box_w = 0.06;
        let box_h = 0.06;
        let x0 = 1.0 - box_w - 0.015;
        let y0 = 0.015;
        if (input.uv.x >= x0 && input.uv.x < x0 + box_w
            && input.uv.y >= y0 && input.uv.y < y0 + box_h) {
            let gx = i32((input.uv.x - x0) / (box_w / 16.0));
            let gy = i32((input.uv.y - y0) / (box_h / 5.0));
            if (gx >= 0 && gx < 15 && gy >= 0 && gy < 5) {
                let di = u32(gx / 4);
                let lx = gx % 4;
                if (lx < 3) {
                    var show = true;
                    switch di {
                        case 0u: { show = fps_int >= 1000u; }
                        case 1u: { show = fps_int >= 100u; }
                        case 2u: { show = fps_int >= 10u; }
                        default: { show = true; }
                    }
                    if (show && ((glyph_row(digits[di], gy) >> (2u - u32(lx))) & 1u) == 1u) {
                        color = vec3<f32>(1.0, 1.0, 1.0);
                    } else {
                        color = color * 0.4;
                    }
                }
            }
        }
    }

    return vec4<f32>(clamp(color * vignette, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
