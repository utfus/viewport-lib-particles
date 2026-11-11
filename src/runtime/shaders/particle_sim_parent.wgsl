// Simulate pass for a parent effect: integrate and age like the normal sim, then
// append spawn events for a child effect.
//
// Identical motion to particle_sim.wgsl, plus an event buffer at binding 2. When
// a trigger fires (every step, or the step a particle dies), the parent appends
// its position and velocity into the child's event ring; the child's emit pass
// consumes them next. The curl-noise helpers are prepended at pipeline build,
// matching the normal sim.

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
    event: vec4<u32>,          // condition (0 every step, 1 on death, 2 on contact), count, child_capacity, active
    space: vec4<f32>,          // xyz local-space translation delta for this frame
    plane: vec4<f32>,          // plane lock: mode (0 none, 1 XY, 2 XZ), depth
    collide: vec4<f32>,        // collision: mode (0 none, 1 bounce, 2 kill, 3 stick), restitution, friction
    grid: vec4<f32>,           // collider grid origin and cell size
    grid_dims: vec4<u32>,      // collider grid cells across, down, edge count
};

struct SpawnEvent {
    position: vec4<f32>,
    velocity: vec4<f32>,
};

struct SpawnEvents {
    count: atomic<u32>,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    events: array<SpawnEvent>,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<storage, read_write> events: SpawnEvents;

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
    var contact = false;
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
        contact = collide_particle(&p, start, params.plane, params.collide, params.grid, params.grid_dims);
    }
    p.lifetime -= dt;
    particles[i] = p;

    // Append spawn events for the child effect. `every step` fires for every
    // live particle each frame; `on death` fires only the step lifetime crosses
    // zero (this invocation ran because the particle was alive at entry).
    if (params.event.w == 1u) {
        var trigger = false;
        if (params.event.x == 0u) {
            trigger = true;
        } else if (params.event.x == 1u) {
            trigger = p.lifetime <= 0.0;
        } else {
            trigger = contact;
        }
        if (trigger) {
            let ecount = params.event.y;
            let child_cap = params.event.z;
            for (var e = 0u; e < ecount; e = e + 1u) {
                let slot = atomicAdd(&events.count, 1u);
                if (slot < child_cap) {
                    events.events[slot].position = vec4<f32>(p.position, 0.0);
                    events.events[slot].velocity = vec4<f32>(p.velocity, 0.0);
                }
            }
        }
    }
}
