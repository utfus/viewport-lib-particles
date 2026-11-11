// Emit pass: recycle dead slots into freshly spawned particles.
//
// One invocation per capacity slot. A slot whose lifetime has run out competes
// for this frame's spawn budget via an atomic counter; the first `spawn_count`
// claimants initialise a new particle. Fixed-function: the spawn shape and
// velocity distribution are selected by `kinds`, not generated per effect. The
// plane and spawn helpers (particle_plane.wgsl, particle_spawn.wgsl) are
// prepended at pipeline build.

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
    spawn_a: vec4<f32>,  // box min xyz | sphere/circle (radius, volume_flag, _, _) | cone (base_r, top_r, height, volume_flag) | ring (radius, thickness) | segment (a.xy, b.xy) | rectangle (half_size.xy)
    spawn_b: vec4<f32>,  // box max xyz
    vel_a:   vec4<f32>,  // fixed vel xyz | box vel min xyz | cone axis xyz
    vel_b:   vec4<f32>,  // box vel max xyz | cone (half_angle, min_speed, max_speed, _)
    colour:  vec4<f32>,
    misc:    vec4<f32>,  // lifetime_min, lifetime_max, size, _ | mesh: spawn_a.x = normal_velocity
    ctrl:    vec4<u32>,  // rng_seed, spawn_count, capacity, mesh_sample_count
    kinds:   vec4<u32>,  // spawn_kind, vel_kind, _, _
    rot:          vec4<f32>,  // spawn roll base, roll variance, size variance, _
    spawn_offset: vec4<f32>,  // world-space spawn offset xyz
    plane:        vec4<f32>,  // plane lock: mode (0 none, 1 XY, 2 XZ), depth
};

// One baked mesh-surface / point-cloud spawn sample.
struct SpawnSample {
    position: vec4<f32>,
    normal:   vec4<f32>,
};

struct Budget { count: atomic<u32> };

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: EmitParams;
@group(0) @binding(2) var<storage, read_write> budget: Budget;
// Mesh spawn-sample pool (a 1-entry default when the effect has no mesh shape).
@group(0) @binding(3) var<storage, read> spawn_samples: array<SpawnSample>;

@compute @workgroup_size(64)
fn emit_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.ctrl.z) {
        return;
    }
    if (particles[i].lifetime > 0.0) {
        return; // still alive
    }
    let claimed = atomicAdd(&budget.count, 1u);
    if (claimed >= params.ctrl.y) {
        return; // spawn budget for this frame exhausted
    }

    var rng = hash_u32(params.ctrl.x ^ (i * 747796405u) ^ (claimed * 2891336453u));

    // Position.
    var pos = params.origin.xyz;
    var sample_normal = vec3<f32>(0.0, 0.0, 0.0);
    let spawn_kind = params.kinds.x;
    if (spawn_kind == 5u) {
        // Mesh-surface / point-cloud: index the baked sample pool.
        let n = params.ctrl.w;
        if (n > 0u) {
            let si = hash_u32(rng * 2654435761u + 1u) % n;
            let s = spawn_samples[si];
            pos += s.position.xyz;
            sample_normal = s.normal.xyz;
        }
    } else {
        pos += spawn_shape_offset(spawn_kind, params.spawn_a, params.spawn_b, params.plane, &rng);
    }

    // Velocity.
    var vel = sample_velocity(params.kinds.y, params.vel_a, params.vel_b, &rng);

    // Mesh spawn normal lift-off: push the particle off the surface
    // along its sample normal, on top of the emitter's own velocity.
    if (spawn_kind == 5u) {
        let normal_speed = params.spawn_a.x;
        vel += sample_normal * normal_speed;
    }

    let life = mix(params.misc.x, params.misc.y, rand01(&rng));

    // Shared spawn offset, on top of the shape and the instance origin.
    pos += params.spawn_offset.xyz;
    pos = lock_position(pos, params.plane);
    vel = lock_vector(vel, params.plane);

    // Per-particle size with a symmetric random spread, clamped at zero.
    let size_var = (rand01(&rng) * 2.0 - 1.0) * params.rot.z;
    let size = max(params.misc.z + size_var, 0.0);

    // Per-particle spawn roll: base plus symmetric random spread (billboard route
    // reads pad.x; other routes ignore it).
    let roll = params.rot.x + (rand01(&rng) * 2.0 - 1.0) * params.rot.y;

    var np: Particle;
    np.position = pos;
    np.lifetime = life;
    np.velocity = vel;
    np.max_lifetime = life;
    np.colour = params.colour;
    np.size = size;
    np.seed = rand01(&rng);
    np.pad = vec2<f32>(roll, 0.0);
    particles[i] = np;
}
