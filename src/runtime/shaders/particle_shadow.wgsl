// Shadow pass: draw each live particle as a small world-axis-aligned quad,
// depth-only, into one cascade tile of the lib's shadow atlas so particles cast
// blob shadows.
//
// Group 0 here is NOT the scene camera: the shadow pass binds a single cascade
// light-space view-projection matrix (matching the lib's `shadow_camera` bind
// group), so this shader declares its own `Light` uniform rather than pulling in
// the shared bindings. Depth-only: no fragment stage.

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

struct Light {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> light: Light;

@group(1) @binding(0) var<storage, read> particles: array<Particle>;
@group(1) @binding(1) var<storage, read> order: array<u32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> @builtin(position) vec4<f32> {
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
    // A world-axis quad (X/Y plane): cheap, casts a round-ish blob for any light.
    let world = p.position + vec3<f32>(c.x * half, c.y * half, 0.0);
    return light.view_proj * vec4<f32>(world, 1.0);
}
