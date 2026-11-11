// Spawn shapes and velocity distributions for the fixed-function emit kernels,
// shared by particle_emit.wgsl and particle_subemit.wgsl. Prepended after
// particle_plane.wgsl, whose helpers place the flat shapes in the lock plane.
//
// Shape kinds: 0 point, 1 box, 2 sphere, 3 circle, 4 cone, 5 mesh samples
// (handled by the emit kernel, which owns the sample pool), 6 ring,
// 7 segment, 8 rectangle. Velocity kinds: 0 fixed, 1 box, 2 cone.

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

// The spawn position relative to the emitting origin.
fn spawn_shape_offset(
    kind: u32,
    a: vec4<f32>,
    b: vec4<f32>,
    lock: vec4<f32>,
    rng: ptr<function, u32>,
) -> vec3<f32> {
    if (kind == 1u) {
        return vec3<f32>(
            mix(a.x, b.x, rand01(rng)),
            mix(a.y, b.y, rand01(rng)),
            mix(a.z, b.z, rand01(rng)),
        );
    } else if (kind == 2u) {
        let dir = rand_dir(rng);
        var r = a.x;
        if (a.y > 0.5) {
            r = r * pow(rand01(rng), 0.3333333);
        }
        return dir * r;
    } else if (kind == 3u) {
        let ang = rand01(rng) * 6.2831853;
        var r = a.x;
        if (a.y > 0.5) {
            r = r * sqrt(rand01(rng)); // uniform over the disc area
        }
        return plane_offset(lock, r * cos(ang), r * sin(ang));
    } else if (kind == 4u) {
        // Cone around +Z: base at z=0, apex at z=height.
        let t = rand01(rng);
        let ring_r = mix(a.x, a.y, t);
        let ang = rand01(rng) * 6.2831853;
        var r = ring_r;
        if (a.w > 0.5) {
            r = ring_r * sqrt(rand01(rng)); // fill the cross-section
        }
        return vec3<f32>(r * cos(ang), r * sin(ang), t * a.z);
    } else if (kind == 6u) {
        // Ring: uniform over the annulus between the inner and outer radius.
        let inner = max(a.x - a.y * 0.5, 0.0);
        let outer = a.x + a.y * 0.5;
        let ang = rand01(rng) * 6.2831853;
        let r = sqrt(mix(inner * inner, outer * outer, rand01(rng)));
        return plane_offset(lock, r * cos(ang), r * sin(ang));
    } else if (kind == 7u) {
        let p = mix(a.xy, a.zw, rand01(rng));
        return plane_offset(lock, p.x, p.y);
    } else if (kind == 8u) {
        let x = (rand01(rng) * 2.0 - 1.0) * a.x;
        let y = (rand01(rng) * 2.0 - 1.0) * a.y;
        return plane_offset(lock, x, y);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

fn sample_velocity(kind: u32, a: vec4<f32>, b: vec4<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    if (kind == 1u) {
        return vec3<f32>(
            mix(a.x, b.x, rand01(rng)),
            mix(a.y, b.y, rand01(rng)),
            mix(a.z, b.z, rand01(rng)),
        );
    } else if (kind == 2u) {
        let axis = normalize(a.xyz);
        let cos_a = mix(cos(b.x), 1.0, rand01(rng));
        let sin_a = sqrt(max(0.0, 1.0 - cos_a * cos_a));
        let phi = rand01(rng) * 6.2831853;
        var up = vec3<f32>(0.0, 0.0, 1.0);
        if (abs(axis.z) > 0.9) {
            up = vec3<f32>(1.0, 0.0, 0.0);
        }
        let t = normalize(cross(up, axis));
        let bt = cross(axis, t);
        let dir = axis * cos_a + (t * cos(phi) + bt * sin(phi)) * sin_a;
        return dir * mix(b.y, b.z, rand01(rng));
    }
    return a.xyz;
}
