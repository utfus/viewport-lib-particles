// Simulate pass: integrate forces into every live particle, then age it.
//
// One invocation per capacity slot. Dead slots are skipped and left for the
// emit pass to recycle. Forces are inlined into a fixed-size array; the count
// says how many are active this frame.

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

// kind (data0.w): 0 = accel (data0.xyz), 1 = drag (data1.z),
//                 2 = attractor (data0.xyz pos, data1.x strength, data1.y falloff),
//                 3 = curl noise (data1.x scale, data1.y strength, data1.z speed).
// Curl-noise helpers (value_noise / curl_noise) are prepended at pipeline build.
struct Force {
    data0: vec4<f32>,
    data1: vec4<f32>,
};

struct SimParams {
    misc: vec4<f32>,           // dt, time, _, force_count
    forces: array<Force, 8>,
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
}
