// Draw pass: one camera-facing billboard per live particle.
//
// Sources position, colour, and size from the simulated particle buffer at
// group 1. Dead slots collapse to a zero-area quad, so they cost nothing in the
// rasteriser. The fragment outputs premultiplied colour so one shader serves
// both additive and over-blend pipelines. `SHARED_BINDINGS_WGSL` (group 0,
// `camera`) is prepended at pipeline-build time.

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

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
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
    let p = particles[ii];

    // Collapse dead particles to zero area (no fragments).
    var half = p.size * 0.5;
    if (p.lifetime <= 0.0) {
        half = 0.0;
    }

    let right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
    let world = p.position + right * (c.x * half) + up * (c.y * half);

    let fade = clamp(p.lifetime / max(p.max_lifetime, 1e-4), 0.0, 1.0);

    var out: VsOut;
    out.pos = camera.view_proj * vec4<f32>(world, 1.0);
    out.color = vec4<f32>(p.colour.rgb, p.colour.a * fade);
    out.uv = c * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let d = distance(in.uv, vec2<f32>(0.5, 0.5));
    let a = in.color.a * (1.0 - smoothstep(0.35, 0.5, d));
    return vec4<f32>(in.color.rgb * a, a); // premultiplied
}
