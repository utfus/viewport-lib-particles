// Outline-mask pass: draw a selected effect's live particles as billboards into
// the R8 mask so the lib's edge-detect draws a selection outline around the
// cloud. The shared `viewport_mask_fs` (appended at build time) writes 1.0.
// `SHARED_BINDINGS_WGSL` (group 0, `camera`) is prepended too.

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
@group(1) @binding(1) var<storage, read> order: array<u32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let c = corners[vi];
    let p = particles[order[ii]];

    var half = p.size * 0.5;
    if (p.lifetime <= 0.0) {
        half = 0.0;
    }
    let right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
    let world = p.position + right * (c.x * half) + up * (c.y * half);

    var out: VsOut;
    out.pos = camera.view_proj * vec4<f32>(world, 1.0);
    return out;
}
