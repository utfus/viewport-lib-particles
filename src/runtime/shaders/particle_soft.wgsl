// Soft-particle draw pass (Phase 9): camera-facing billboards that fade their
// alpha as they approach the opaque scene surface, removing the hard clip line
// where a particle intersects geometry.
//
// Runs in the renderer's read-only-depth plugin pass (`paint_depth_read`), after
// opaque + sprites and before OIT, with the scene depth bound read-only. Group 1
// folds particles + order + draw params + sprite (so group 3 is free for the
// scene depth); group 2 is the ramp LUT; group 3 is the scene depth (matching
// the renderer's `depth_read_bgl`). `SHARED_BINDINGS_WGSL` (group 0, `camera`)
// and `SHARED_DEPTH_READ_WGSL` (depth reconstruction) are prepended at build.

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

// Group 1: particles + order + draw params + billboard texture (folded).
@group(1) @binding(0) var<storage, read> particles: array<Particle>;
@group(1) @binding(1) var<storage, read> order: array<u32>;
struct DrawParams {
    stretch: f32,
    align: u32,
    trail_width: f32,
    trail_segments: u32,
    tex_mode: u32,
    flip_cols: u32,
    flip_rows: u32,
    flip_fps: f32,
    orient: u32,
    orient_param: f32,
    screen_size: u32,
    soft_dist: f32,
};
@group(1) @binding(2) var<uniform> draw_params: DrawParams;
@group(1) @binding(3) var sprite_tex: texture_2d<f32>;
@group(1) @binding(4) var sprite_samp: sampler;

// Group 2: the lifetime-ramp LUT (row 0 colour+size, row 1 alpha).
@group(2) @binding(0) var ramp_tex: texture_2d<f32>;
@group(2) @binding(1) var ramp_samp: sampler;

// Group 3: read-only scene depth (matches the renderer's depth_read_bgl).
@group(3) @binding(0) var scene_depth: texture_depth_2d;
@group(3) @binding(1) var scene_depth_samp: sampler;

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

// Same billboard placement as particle_draw.wgsl (orientation / screen-size /
// stretch).
fn billboard_world(p: Particle, c: vec2<f32>, size_scale: f32) -> vec3<f32> {
    var half = p.size * 0.5 * size_scale;
    if (p.lifetime <= 0.0) {
        half = 0.0;
    }

    var bx = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    var by = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);

    let orient = draw_params.orient;
    if (orient == 1u) {
        let toeye = normalize(camera.eye_pos - p.position);
        bx = normalize(cross(by, toeye));
        by = cross(toeye, bx);
    } else if (orient == 3u) {
        let ca = cos(draw_params.orient_param);
        let sa = sin(draw_params.orient_param);
        let nx = bx * ca + by * sa;
        let ny = by * ca - bx * sa;
        bx = nx;
        by = ny;
    }

    var hsize = half;
    if (draw_params.screen_size == 1u) {
        let wclip = (camera.view_proj * vec4<f32>(p.position, 1.0)).w;
        hsize = half * max(wclip, 1e-3);
    }

    let vplane = vec2<f32>(dot(p.velocity, bx), dot(p.velocity, by));
    let speed2d = length(vplane);
    if (draw_params.stretch > 0.0 && speed2d > 1e-4) {
        let dirx = vplane / speed2d;
        let diry = vec2<f32>(-dirx.y, dirx.x);
        let speed = length(p.velocity);
        let len = hsize * (1.0 + draw_params.stretch * speed);
        let ax = bx * dirx.x + by * dirx.y;
        let ay = bx * diry.x + by * diry.y;
        return p.position + ax * (c.x * len) + ay * (c.y * hsize);
    } else if (orient == 2u && speed2d > 1e-4) {
        let dirx = vplane / speed2d;
        let diry = vec2<f32>(-dirx.y, dirx.x);
        let ax = bx * dirx.x + by * dirx.y;
        let ay = bx * diry.x + by * diry.y;
        return p.position + ax * (c.x * hsize) + ay * (c.y * hsize);
    }
    return p.position + bx * (c.x * hsize) + by * (c.y * hsize);
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) age: f32,
    @location(3) age_sec: f32,
    @location(4) world_pos: vec3<f32>,
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

    let age = clamp(1.0 - p.lifetime / max(p.max_lifetime, 1e-4), 0.0, 1.0);
    let ramp0 = textureSampleLevel(ramp_tex, ramp_samp, vec2<f32>(age, 0.25), 0.0);
    let ramp1 = textureSampleLevel(ramp_tex, ramp_samp, vec2<f32>(age, 0.75), 0.0);

    let world = billboard_world(p, c, ramp0.a);

    var out: VsOut;
    out.pos = camera.view_proj * vec4<f32>(world, 1.0);
    out.color = vec4<f32>(p.colour.rgb * ramp0.rgb, p.colour.a * ramp1.r);
    out.uv = c * 0.5 + vec2<f32>(0.5, 0.5);
    out.age = age;
    out.age_sec = max(p.max_lifetime - p.lifetime, 0.0);
    out.world_pos = world;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
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

    // Soft fade against the opaque scene depth. `in.pos` is the framebuffer
    // coordinate; the depth texture is the same render-target space, so its
    // dimensions give the screen UV.
    let dims = vec2<f32>(textureDimensions(scene_depth));
    let screen_uv = in.pos.xy / dims;
    let scene_ndc_z = textureSample(scene_depth, scene_depth_samp, screen_uv);
    let fade = viewport_soft_fade_from_ndc(in.world_pos, screen_uv, scene_ndc_z, draw_params.soft_dist);
    a = a * fade;

    return vec4<f32>(rgb * a, a); // premultiplied
}
