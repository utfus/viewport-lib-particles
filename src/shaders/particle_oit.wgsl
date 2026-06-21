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
// along the screen-projected velocity.
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
    @location(1) uv: vec2<f32>,
    @location(2) view_z: f32,
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
    return out;
}

@fragment
fn fs(in: VsOut) -> OitOutput {
    let d = distance(in.uv, vec2<f32>(0.5, 0.5));
    let a = in.color.a * (1.0 - smoothstep(0.35, 0.5, d));
    // `viewport_oit_pack` takes straight (non-premultiplied) colour and
    // premultiplies internally.
    return viewport_oit_pack(in.color.rgb, a, in.view_z);
}
