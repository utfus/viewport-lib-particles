// Live-count pass: tally the live particles of one instance into a single
// atomic, for the host-facing readback. One invocation per capacity slot;
// runs after the simulate pass so it counts this frame's survivors. A slot is
// live when its `lifetime > 0.0` (dead / recycled slots hold `lifetime <= 0.0`).
// The count is copied to a mappable buffer and read back a frame later, so this
// pass never stalls the frame.

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

struct Stats {
    live: atomic<u32>,
};

struct CountParams {
    capacity: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> stats: Stats;
@group(0) @binding(2) var<uniform> params: CountParams;

@compute @workgroup_size(64)
fn count_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.capacity) {
        return;
    }
    if (particles[i].lifetime > 0.0) {
        atomicAdd(&stats.live, 1u);
    }
}
