// Fixed harness spliced around every effect's generated emit / simulate bodies.
//
// The codegen pass inserts lowered modifier code at the `{{init}}` and
// `{{update}}` markers. The particle struct, bindings, and workgroup setup are
// identical across effects so only the modifier logic differs.
//
// Status: skeleton. This is the shared scaffold; the marker blocks are empty
// until the codegen phase fills them.

struct Particle {
    position: vec3<f32>,
    lifetime: f32,      // seconds remaining; <= 0 means dead
    velocity: vec3<f32>,
    max_lifetime: f32,  // initial lifetime, for fade ramps
    colour: vec4<f32>,
    size: f32,
    seed: f32,          // stable per-spawn random seed
    _pad: vec2<f32>,
};

struct SimParams {
    dt: f32,
    time: f32,
    spawn_count: u32,
    capacity: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

// Emit kernel: recycle a dead slot and run the init modifiers.
@compute @workgroup_size(64)
fn emit_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.spawn_count) {
        return;
    }
    // {{init}} lowered init-modifier bodies write particles[slot] here.
}

// Simulate kernel: sum forces, integrate, decrement lifetime.
@compute @workgroup_size(64)
fn simulate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.capacity) {
        return;
    }
    var p = particles[i];
    if (p.lifetime <= 0.0) {
        return;
    }
    // {{update}} lowered force-modifier bodies accumulate acceleration here.
    p.velocity = p.velocity; // + accel * params.dt
    p.position = p.position + p.velocity * params.dt;
    p.lifetime = p.lifetime - params.dt;
    particles[i] = p;
}
