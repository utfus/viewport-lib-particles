// Screen-space refraction post-process: a fullscreen pass that samples the
// rendered scene colour and re-samples it with a radial displacement, producing
// an expanding shockwave / heat-haze lens.
//
// Self-contained (a `GpuPlugin` gets no shared bindings): group 0 is the scene
// colour texture + sampler + this effect's uniform. The vertex stage emits one
// full-screen triangle; the fragment stage displaces the sample point by a
// signed radial profile centered on the ring.

struct Refract {
    center: vec2<f32>,   // ring center in uv space (0..1)
    radius: f32,         // current ring radius in uv units
    width: f32,          // ring band half-width
    strength: f32,       // displacement scale
    aspect: f32,         // viewport width / height, keeps the ring circular
    pad: vec2<f32>,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
@group(0) @binding(2) var<uniform> u: Refract;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    // Oversized triangle covering the whole clip rectangle.
    var xy = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let p = xy[vi];
    var out: VsOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    // uv origin at the top-left, matching texture space.
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    // Aspect-correct distance from the ring center so the ring stays circular.
    let d_vec = (uv - u.center) * vec2<f32>(u.aspect, 1.0);
    let d = length(d_vec);

    // Signed lens profile: x * exp(-x^2) peaks just inside and outside the ring
    // and is ~0 elsewhere, so the scene is untouched away from the wavefront.
    let x = (d - u.radius) / max(u.width, 1e-4);
    let profile = x * exp(-x * x);

    let dir = normalize(uv - u.center + vec2<f32>(1e-6, 1e-6));
    let offset = dir * (profile * u.strength);

    return textureSampleLevel(src_tex, src_samp, uv + offset, 0.0);
}
