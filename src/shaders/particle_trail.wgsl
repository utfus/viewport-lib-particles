// Trail render route: sweep a camera-facing ribbon through each particle's
// recorded position history. One instance per particle; `trail_segments` quads
// (6 vertices each) per instance, newest segment at the head.
//
// The ribbon width and alpha taper from the head to the tail. Each segment
// connects two consecutive history samples; a segment is drawn only when both
// samples belong to the live particle (their stored seed matches), so recycled
// slots and dead particles collapse to nothing. `SHARED_BINDINGS_WGSL`
// (group 0, `camera`) is prepended at pipeline-build time.

struct Particle {
    position: vec3<f32>,
    lifetime: f32,
    velocity: vec3<f32>,
    max_lifetime: f32,
    colour: vec4<f32>,
    size: f32,
    seed: f32,
    pad: vec2<f32>,
};

struct HistParams {
    head: u32,
    capacity: u32,
    history_len: u32,
    pad: u32,
};

@group(1) @binding(0) var<storage, read> particles: array<Particle>;
@group(1) @binding(1) var<storage, read> history: array<vec4<f32>>;
@group(1) @binding(2) var<uniform> hist: HistParams;

@group(2) @binding(0) var ramp_tex: texture_2d<f32>;
@group(2) @binding(1) var ramp_samp: sampler;

struct DrawParams {
    stretch: f32,
    align: u32,
    trail_width: f32,
    trail_segments: u32,
};
@group(3) @binding(0) var<uniform> draw_params: DrawParams;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// Ring index `k` steps back from the head, wrapping within the history length.
fn ring(k: u32) -> u32 {
    let hl = hist.history_len;
    return (hist.head + hl - (k % hl)) % hl;
}

fn sample(pi: u32, k: u32) -> vec4<f32> {
    return history[pi * hist.history_len + ring(k)];
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let seg = vi / 6u;
    let corner = vi % 6u;
    let p = particles[ii];

    var out: VsOut;

    let s_new = sample(ii, seg);
    let s_old = sample(ii, seg + 1u);
    let ok_new = abs(s_new.w - p.seed) < 1e-6;
    let ok_old = abs(s_old.w - p.seed) < 1e-6;
    let dead = p.lifetime <= 0.0 || seg >= draw_params.trail_segments;

    if (dead || !ok_new || !ok_old) {
        // Collapse the whole segment: clip behind the camera, no fragments.
        out.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    let p_new = s_new.xyz;
    let p_old = s_old.xyz;

    let total = f32(max(draw_params.trail_segments, 1u));
    let t_new = f32(seg) / total;
    let t_old = f32(seg + 1u) / total;

    // Two triangles: (new-, old-, old+) and (new-, old+, new+).
    var is_old = false;
    var side = -1.0;
    if (corner == 1u) { is_old = true; side = -1.0; }
    else if (corner == 2u) { is_old = true; side = 1.0; }
    else if (corner == 4u) { is_old = true; side = 1.0; }
    else if (corner == 5u) { is_old = false; side = 1.0; }

    var base = p_new;
    var t = t_new;
    if (is_old) {
        base = p_old;
        t = t_old;
    }
    let w = draw_params.trail_width * (1.0 - t);

    // Width axis: perpendicular to the segment and facing the camera.
    let seg_dir = normalize(p_new - p_old + vec3<f32>(0.0, 0.0, 1e-6));
    let cam_fwd = vec3<f32>(camera.view[0][2], camera.view[1][2], camera.view[2][2]);
    var wdir = cross(seg_dir, cam_fwd);
    let wlen = length(wdir);
    if (wlen < 1e-5) {
        wdir = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    } else {
        wdir = wdir / wlen;
    }

    let world = base + wdir * (side * w);

    let ramp = textureSampleLevel(ramp_tex, ramp_samp, vec2<f32>(t, 0.5), 0.0);
    let fade = (1.0 - t) * clamp(p.lifetime / max(p.max_lifetime, 1e-4), 0.0, 1.0);

    out.pos = camera.view_proj * vec4<f32>(world, 1.0);
    out.color = vec4<f32>(p.colour.rgb * ramp.rgb, p.colour.a * fade);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let a = in.color.a;
    return vec4<f32>(in.color.rgb * a, a); // premultiplied
}
