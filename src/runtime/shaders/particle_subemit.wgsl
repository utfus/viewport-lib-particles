// Emit pass for a child effect: recycle dead slots into new particles seeded
// from a parent effect's spawn events, plus the child's own spawner.
//
// Like particle_emit.wgsl, but this frame's spawn budget is the child's own
// spawn_count plus the number of events the parent appended. A claimant that
// lands in the event range seeds its particle at the event position and inherits
// a fraction of the event velocity; the rest spawn from the child's own emitter
// origin. Everything else (spawn shape, velocity distribution, lifetime, colour,
// size) comes from the child's emitter, so the child shapes the burst. The plane
// and spawn helpers are prepended at pipeline build, as for the emit kernel.

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

struct EmitParams {
    origin:  vec4<f32>,  // xyz world origin (instance position)
    spawn_a: vec4<f32>,  // shape parameters, as in particle_emit.wgsl
    spawn_b: vec4<f32>,  // box max xyz
    vel_a:   vec4<f32>,  // fixed vel xyz | box vel min xyz | cone axis xyz
    vel_b:   vec4<f32>,  // box vel max xyz | cone (half_angle, min_speed, max_speed, _)
    colour:  vec4<f32>,
    misc:    vec4<f32>,  // lifetime_min, lifetime_max, size, inherit_velocity
    ctrl:    vec4<u32>,  // rng_seed, spawn_count, capacity, _
    kinds:   vec4<u32>,  // spawn_kind, vel_kind, has_events, _
    rot:          vec4<f32>,  // unused here
    spawn_offset: vec4<f32>,  // unused here
    plane:        vec4<f32>,  // plane lock: mode (0 none, 1 XY, 2 XZ), depth
};

struct Budget { count: atomic<u32> };

struct SpawnEvent {
    position: vec4<f32>,
    velocity: vec4<f32>,
};

struct SpawnEvents {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    events: array<SpawnEvent>,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: EmitParams;
@group(0) @binding(2) var<storage, read_write> budget: Budget;
@group(0) @binding(3) var<storage, read> events: SpawnEvents;

@compute @workgroup_size(64)
fn emit_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.ctrl.z) {
        return;
    }
    if (particles[i].lifetime > 0.0) {
        return; // still alive
    }

    // Available events this frame, clamped to the child capacity (writes beyond
    // capacity were dropped by the parent).
    var ecount = 0u;
    if (params.kinds.z == 1u) {
        ecount = min(events.count, params.ctrl.z);
    }
    let own = params.ctrl.y;
    let total = own + ecount;

    let claimed = atomicAdd(&budget.count, 1u);
    if (claimed >= total) {
        return; // spawn budget for this frame exhausted
    }

    var rng = hash_u32(params.ctrl.x ^ (i * 747796405u) ^ (claimed * 2891336453u));

    // Base position and inherited velocity: event-seeded for the first `ecount`
    // claims, the emitter origin otherwise.
    var base = params.origin.xyz;
    var inherited_vel = vec3<f32>(0.0, 0.0, 0.0);
    if (claimed < ecount) {
        let e = events.events[claimed];
        base = e.position.xyz;
        // Drop the normal component so inherited velocity cannot carry a
        // child off its plane.
        inherited_vel = lock_vector(e.velocity.xyz * params.misc.w, params.plane);
    }

    // Position: base plus the child spawn shape offset, on the child's plane.
    let pos = lock_position(
        base + spawn_shape_offset(params.kinds.x, params.spawn_a, params.spawn_b, params.plane, &rng),
        params.plane,
    );

    // Velocity: the child distribution plus the inherited parent velocity.
    let vel = lock_vector(
        sample_velocity(params.kinds.y, params.vel_a, params.vel_b, &rng),
        params.plane,
    ) + inherited_vel;

    let life = mix(params.misc.x, params.misc.y, rand01(&rng));

    var np: Particle;
    np.position = pos;
    np.lifetime = life;
    np.velocity = vel;
    np.max_lifetime = life;
    np.colour = params.colour;
    np.size = params.misc.z;
    np.seed = rand01(&rng);
    np.pad = vec2<f32>(0.0, 0.0);
    particles[i] = np;
}
