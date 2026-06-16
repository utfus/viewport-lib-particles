// Sort setup pass: fill the draw-order index buffer with the identity
// permutation and compute each particle's sort key (distance from the camera).
//
// One invocation per padded slot. The order buffer is padded up to a power of
// two for the bitonic sort; padding slots and dead particles get sentinel keys
// that sort them behind every live particle, so the first `capacity` entries of
// the sorted order are exactly the real slots in back-to-front order.

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

struct InitParams {
    eye: vec4<f32>,   // camera eye position (xyz)
    capacity: u32,    // real particle count
    n: u32,           // padded power-of-two length
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> order: array<u32>;
@group(0) @binding(3) var<uniform> p: InitParams;

@compute @workgroup_size(64)
fn init_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) {
        return;
    }
    order[i] = i;
    if (i >= p.capacity) {
        keys[i] = -2.0; // padding: sorts behind everything
        return;
    }
    let part = particles[i];
    if (part.lifetime <= 0.0) {
        keys[i] = -1.0; // dead: behind live, ahead of padding
        return;
    }
    // Larger key = farther, drawn first (back to front) for correct over-blend.
    keys[i] = distance(p.eye.xyz, part.position);
}
