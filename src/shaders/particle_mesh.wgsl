// Mesh render route: draw one instance of an uploaded mesh per live particle.
//
// The vertex buffer holds the mesh's local-space positions; the particle buffer
// at group 1 provides each instance's world position, size, colour, and seed.
// Particles orient either to their velocity or with a stable per-particle tumble
// (chosen by `draw_params.align`). Unlit; colour comes from the particle and the
// lifetime ramp. `SHARED_BINDINGS_WGSL` (group 0, `camera`) is prepended at
// pipeline-build time.

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

@group(1) @binding(0) var<storage, read> particles: array<Particle>;
@group(2) @binding(0) var ramp_tex: texture_2d<f32>;
@group(2) @binding(1) var ramp_samp: sampler;

struct DrawParams {
    stretch: f32,
    align: u32,   // 0 = velocity-aligned, 1 = random tumble
    pad0: u32,
    pad1: u32,
};
@group(3) @binding(0) var<uniform> draw_params: DrawParams;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// Rotation that aligns local +Z to the velocity direction.
fn basis_from_velocity(vel: vec3<f32>) -> mat3x3<f32> {
    let f = normalize(vel + vec3<f32>(0.0, 0.0, 1e-5));
    var up = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(f.z) > 0.9) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let r = normalize(cross(up, f));
    let u = cross(f, r);
    return mat3x3<f32>(r, u, f);
}

// Stable per-particle tumble that advances with age (Rodrigues rotation).
fn rot_from_seed_age(seed: f32, age: f32) -> mat3x3<f32> {
    let a = seed * 6.2831853;
    let axis = normalize(vec3<f32>(sin(a * 1.7) + 0.1, cos(a * 2.3), sin(a * 0.7) + 0.2));
    let angle = a + age * 5.0;
    let s = sin(angle);
    let co = cos(angle);
    let t = 1.0 - co;
    let x = axis.x;
    let y = axis.y;
    let z = axis.z;
    return mat3x3<f32>(
        vec3<f32>(t * x * x + co, t * x * y + s * z, t * x * z - s * y),
        vec3<f32>(t * x * y - s * z, t * y * y + co, t * y * z + s * x),
        vec3<f32>(t * x * z + s * y, t * y * z - s * x, t * z * z + co),
    );
}

@vertex
fn vs(@location(0) v_pos: vec3<f32>, @builtin(instance_index) ii: u32) -> VsOut {
    let p = particles[ii];

    let age = clamp(1.0 - p.lifetime / max(p.max_lifetime, 1e-4), 0.0, 1.0);
    let ramp = textureSampleLevel(ramp_tex, ramp_samp, vec2<f32>(age, 0.5), 0.0);

    var scale = p.size * ramp.a;
    if (p.lifetime <= 0.0) {
        scale = 0.0; // collapse dead instances
    }

    var rot: mat3x3<f32>;
    if (draw_params.align == 0u) {
        rot = basis_from_velocity(p.velocity);
    } else {
        rot = rot_from_seed_age(p.seed, age);
    }

    let world = p.position + rot * (v_pos * scale);
    let fade = clamp(p.lifetime / max(p.max_lifetime, 1e-4), 0.0, 1.0);

    var out: VsOut;
    out.pos = camera.view_proj * vec4<f32>(world, 1.0);
    out.color = vec4<f32>(p.colour.rgb * ramp.rgb, p.colour.a * fade);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let a = in.color.a;
    return vec4<f32>(in.color.rgb * a, a); // premultiplied
}
