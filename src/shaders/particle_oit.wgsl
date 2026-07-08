// OIT draw pass: the same camera-facing billboards as particle_draw.wgsl, but
// the fragment writes weighted-blended OIT outputs (accum + reveal) so
// alpha-blended particles composite order-independently and after the skybox.
//
// Additive particles stay in the main HDR draw pass; only Alpha/Premultiplied
// effects route here, so no back-to-front sort is needed. `SHARED_BINDINGS_WGSL`
// (group 0, `camera`) and `SHARED_OIT_WGSL` (`OitOutput`, `viewport_oit_pack`)
// are prepended at pipeline-build time.

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

@group(1) @binding(0) var<storage, read> particles: array<Particle>;
// Draw order: instance i draws particle `order[i]`. Identity for OIT effects
// since weighted blending is order-independent.
@group(1) @binding(1) var<storage, read> order: array<u32>;

// Group 2: the lifetime-ramp LUT. rgb multiplies colour, a multiplies size.
@group(2) @binding(0) var ramp_tex: texture_2d<f32>;
@group(2) @binding(1) var ramp_samp: sampler;

// Group 3: per-effect draw params. `stretch` (> 0) elongates the billboard
// along the screen-projected velocity; the texture fields drive sampling and
// flipbook animation in the fragment.
struct DrawParams {
    stretch: f32,
    align: u32,
    trail_width: f32,
    trail_segments: u32,
    tex_mode: u32,
    flip_cols: u32,
    flip_rows: u32,
    flip_fps: f32,
};
@group(3) @binding(0) var<uniform> draw_params: DrawParams;
// Billboard texture + sampler (a 1x1 white default when the effect has none).
@group(3) @binding(1) var sprite_tex: texture_2d<f32>;
@group(3) @binding(2) var sprite_samp: sampler;

// Flipbook: remap `uv` into the current atlas cell for this particle's age.
fn flip_uv(uv: vec2<f32>, age: f32, age_sec: f32) -> vec2<f32> {
    if (draw_params.flip_cols == 0u || draw_params.flip_rows == 0u) {
        return uv;
    }
    let cells = draw_params.flip_cols * draw_params.flip_rows;
    var frame: u32;
    if (draw_params.flip_fps > 0.0) {
        frame = u32(floor(age_sec * draw_params.flip_fps)) % cells;
    } else {
        frame = min(u32(floor(age * f32(cells))), cells - 1u);
    }
    let col = frame % draw_params.flip_cols;
    let row = frame / draw_params.flip_cols;
    let cell = vec2<f32>(1.0 / f32(draw_params.flip_cols), 1.0 / f32(draw_params.flip_rows));
    return (vec2<f32>(f32(col), f32(row)) + uv) * cell;
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) view_z: f32,
    @location(3) age: f32,
    @location(4) age_sec: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let c = corners[vi];
    let p = particles[order[ii]];

    // Normalized age (0 at spawn, 1 at death) drives the ramp lookup.
    let age = clamp(1.0 - p.lifetime / max(p.max_lifetime, 1e-4), 0.0, 1.0);
    let ramp = textureSampleLevel(ramp_tex, ramp_samp, vec2<f32>(age, 0.5), 0.0);

    // Collapse dead particles to zero area (no fragments).
    var half = p.size * 0.5 * ramp.a;
    if (p.lifetime <= 0.0) {
        half = 0.0;
    }

    let right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);

    // Screen-plane offset of this corner. Round by default; when stretched,
    // orient the long axis along the velocity projected into the camera plane.
    var off = vec2<f32>(c.x * half, c.y * half);
    let vplane = vec2<f32>(dot(p.velocity, right), dot(p.velocity, up));
    let speed2d = length(vplane);
    if (draw_params.stretch > 0.0 && speed2d > 1e-4) {
        let dirx = vplane / speed2d;
        let diry = vec2<f32>(-dirx.y, dirx.x);
        let speed = length(p.velocity);
        let len = half * (1.0 + draw_params.stretch * speed);
        off = dirx * (c.x * len) + diry * (c.y * half);
    }

    let world = p.position + right * off.x + up * off.y;

    let fade = clamp(p.lifetime / max(p.max_lifetime, 1e-4), 0.0, 1.0);

    var out: VsOut;
    out.pos = camera.view_proj * vec4<f32>(world, 1.0);
    out.color = vec4<f32>(p.colour.rgb * ramp.rgb, p.colour.a * fade);
    out.uv = c * 0.5 + vec2<f32>(0.5, 0.5);
    // View-space Z (negative in front of the camera) weights the OIT blend.
    out.view_z = (camera.view * vec4<f32>(world, 1.0)).z;
    out.age = age;
    out.age_sec = max(p.max_lifetime - p.lifetime, 0.0);
    return out;
}

@fragment
fn fs(in: VsOut) -> OitOutput {
    let texel = textureSample(sprite_tex, sprite_samp, flip_uv(in.uv, in.age, in.age_sec));
    var rgb = in.color.rgb;
    var a = in.color.a;
    if (draw_params.tex_mode == 0u) {
        let d = distance(in.uv, vec2<f32>(0.5, 0.5));
        a = a * (1.0 - smoothstep(0.35, 0.5, d));
    } else if (draw_params.tex_mode == 1u) {
        rgb = rgb * texel.rgb;
        a = a * texel.a;
    } else if (draw_params.tex_mode == 2u) {
        rgb = rgb * texel.rgb;
    } else {
        a = a * texel.r;
    }
    // `viewport_oit_pack` takes straight (non-premultiplied) colour and
    // premultiplies internally.
    return viewport_oit_pack(rgb, a, in.view_z);
}
