// Append pass: record each particle's current position into its trail history
// ring. One invocation per capacity slot; runs after the simulate pass so the
// recorded sample is this frame's integrated position.
//
// Each particle owns `history_len` samples. `head` advances one slot per frame,
// so consecutive frames leave a contiguous path. The sample's `w` carries the
// particle's seed; the ribbon draw compares it against the live seed to discard
// history left behind by a previous occupant of a recycled slot. Dead slots
// write a -1 sentinel so no ribbon connects through them.

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

struct HistParams {
    head: u32,
    capacity: u32,
    history_len: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> history: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: HistParams;

@compute @workgroup_size(64)
fn history_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.capacity) {
        return;
    }
    let p = particles[i];
    let idx = i * params.history_len + params.head;
    if (p.lifetime > 0.0) {
        history[idx] = vec4<f32>(p.position, p.seed);
    } else {
        history[idx] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
}
