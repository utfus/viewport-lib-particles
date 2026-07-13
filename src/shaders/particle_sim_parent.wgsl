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

struct Force {
    data0: vec4<f32>,
    data1: vec4<f32>,
};

struct SimParams {
    misc: vec4<f32>,           // dt, time, _, force_count
    forces: array<Force, 8>,
    event: vec4<u32>,          // condition (0=every step, 1=on death), count, child_capacity, active
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
    var accel = vec3<f32>(0.0, 0.0, 0.0);
    var drag = 0.0;

    for (var k = 0u; k < count; k = k + 1u) {
        let f = params.forces[k];
        let kind = u32(f.data0.w + 0.5);
        if (kind == 0u) {
            accel += f.data0.xyz;
        } else if (kind == 1u) {
            drag += f.data1.z;
        } else if (kind == 2u) {
            let to = f.data0.xyz - p.position;
            let dist = length(to);
            let soft = dist + f.data1.y;
            accel += (to / max(dist, 1e-4)) * (f.data1.x / (soft * soft));
        } else if (kind == 3u) {
            let domain = p.position * f.data1.x + vec3<f32>(time * f.data1.z);
            accel += curl_noise(domain) * f.data1.y;
        }
    }

    p.velocity += accel * dt;
    p.velocity *= max(0.0, 1.0 - drag * dt);
    p.position += p.velocity * dt;
    p.lifetime -= dt;
    particles[i] = p;

    // Append spawn events for the child effect. `every step` fires for every
    // live particle each frame; `on death` fires only the step lifetime crosses
    // zero (this invocation ran because the particle was alive at entry).
    if (params.event.w == 1u) {
        var trigger = false;
        if (params.event.x == 0u) {
            trigger = true;
        } else {
            trigger = p.lifetime <= 0.0;
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
