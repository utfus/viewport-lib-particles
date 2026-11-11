// Simulate pass: apply modifiers to every live particle, then age it.
//
// One invocation per capacity slot. Dead slots are skipped and left for the emit
// pass to recycle. Modifiers are inlined into a fixed-size array; the count says
// how many are active this frame. The `Modifier` struct and the per-kind
// `eval_modifier` are shared with the parent-sim shader (prepended at build).

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

struct SimParams {
    misc: vec4<f32>,           // dt, time, _, modifier_count
    modifiers: array<Modifier, 8>,
    event: vec4<u32>,          // sub-emitter control (unused here; parent shader reads it)
    space: vec4<f32>,          // xyz local-space translation delta for this frame
    plane: vec4<f32>,          // plane lock: mode (0 none, 1 XY, 2 XZ), depth
    collide: vec4<f32>,        // collision: mode (0 none, 1 bounce, 2 kill, 3 stick), restitution, friction
    grid: vec4<f32>,           // collider grid origin and cell size
    grid_dims: vec4<u32>,      // collider grid cells across, down, edge count
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn sim_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&particles)) {
        return;
    }
    var p = particles[i];
    if (p.lifetime <= 0.0) {
        return;
    }

    let dt = params.misc.x;
    let time = params.misc.y;
    let count = u32(params.misc.w);
    // A particle stuck to a collider only ages.
    let stuck = p.pad.y > 0.5;
    let start = p.position;
    var accel = vec3<f32>(0.0, 0.0, 0.0);
    var drag = 0.0;
    var killed = false;

    for (var k = 0u; k < count && !stuck; k = k + 1u) {
        eval_modifier(params.modifiers[k], p.position, p.velocity, time, &accel, &drag, &killed);
    }

    if (killed) {
        p.lifetime = 0.0;
        particles[i] = p;
        return;
    }

    if (!stuck) {
        p.velocity += lock_vector(accel, params.plane) * dt;
        p.velocity *= max(0.0, 1.0 - drag * dt);
        p.velocity = lock_vector(p.velocity, params.plane);
        p.position += p.velocity * dt;
        p.position += params.space.xyz; // local-space: follow the emitter's translation
        // Reset the depth every step so drift cannot accumulate off the plane.
        p.position = lock_position(p.position, params.plane);
        _ = collide_particle(&p, start, params.plane, params.collide, params.grid, params.grid_dims);
    }
    p.lifetime -= dt;
    particles[i] = p;
}
