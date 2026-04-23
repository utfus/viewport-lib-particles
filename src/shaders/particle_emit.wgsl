// Emit pass: recycle dead slots into freshly spawned particles.
//
// One invocation per capacity slot. A slot whose lifetime has run out competes
// for this frame's spawn budget via an atomic counter; the first `spawn_count`
// claimants initialize a new particle. Fixed-function: the spawn shape and
// velocity distribution are selected by `kinds`, not generated per effect.

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
    spawn_a: vec4<f32>,  // box min xyz | sphere (radius, volume_flag, _, _)
    spawn_b: vec4<f32>,  // box max xyz
    vel_a:   vec4<f32>,  // fixed vel xyz | box vel min xyz | cone axis xyz
    vel_b:   vec4<f32>,  // box vel max xyz | cone (half_angle, min_speed, max_speed, _)
    colour:  vec4<f32>,
    misc:    vec4<f32>,  // lifetime_min, lifetime_max, size, _
    ctrl:    vec4<u32>,  // rng_seed, spawn_count, capacity, _
    kinds:   vec4<u32>,  // spawn_kind, vel_kind, _, _
};

struct Budget { count: atomic<u32> };

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: EmitParams;
@group(0) @binding(2) var<storage, read_write> budget: Budget;

fn hash_u32(x: u32) -> u32 {
    var v = x;
    v ^= v >> 16u;
    v = v * 0x7feb352du;
    v ^= v >> 15u;
    v = v * 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}

fn rand01(state: ptr<function, u32>) -> f32 {
    *state = hash_u32(*state);
    return f32(*state) / 4294967295.0;
}

fn rand_dir(state: ptr<function, u32>) -> vec3<f32> {
    let z = rand01(state) * 2.0 - 1.0;
    let a = rand01(state) * 6.2831853;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(r * cos(a), r * sin(a), z);
}

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
    let spawn_kind = params.kinds.x;
    if (spawn_kind == 1u) {
        pos += vec3<f32>(
            mix(params.spawn_a.x, params.spawn_b.x, rand01(&rng)),
            mix(params.spawn_a.y, params.spawn_b.y, rand01(&rng)),
            mix(params.spawn_a.z, params.spawn_b.z, rand01(&rng)),
        );
    } else if (spawn_kind == 2u) {
        let dir = rand_dir(&rng);
        var r = params.spawn_a.x;
        if (params.spawn_a.y > 0.5) {
            r = r * pow(rand01(&rng), 0.3333333);
        }
        pos += dir * r;
    }

    // Velocity.
    var vel = params.vel_a.xyz;
    let vel_kind = params.kinds.y;
    if (vel_kind == 1u) {
        vel = vec3<f32>(
            mix(params.vel_a.x, params.vel_b.x, rand01(&rng)),
            mix(params.vel_a.y, params.vel_b.y, rand01(&rng)),
            mix(params.vel_a.z, params.vel_b.z, rand01(&rng)),
        );
    } else if (vel_kind == 2u) {
        let axis = normalize(params.vel_a.xyz);
        let half_angle = params.vel_b.x;
        let cos_a = mix(cos(half_angle), 1.0, rand01(&rng));
        let sin_a = sqrt(max(0.0, 1.0 - cos_a * cos_a));
        let phi = rand01(&rng) * 6.2831853;
        var up = vec3<f32>(0.0, 0.0, 1.0);
        if (abs(axis.z) > 0.9) {
            up = vec3<f32>(1.0, 0.0, 0.0);
        }
        let t = normalize(cross(up, axis));
        let b = cross(axis, t);
        let dir = axis * cos_a + (t * cos(phi) + b * sin(phi)) * sin_a;
        let speed = mix(params.vel_b.y, params.vel_b.z, rand01(&rng));
        vel = dir * speed;
    }

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
