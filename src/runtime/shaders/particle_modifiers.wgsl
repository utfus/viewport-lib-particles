// Shared modifier definitions for the fixed-function simulate shaders.
//
// One `case` per Modifier kind, matching the `Modifier` enum order and the
// packing in `build_sim_params`. Prepended to both particle_sim.wgsl and
// particle_sim_parent.wgsl (after the noise helpers, so `curl_noise` is in
// scope). Adding a modifier is one `case` here plus the matching enum variant
// and packing arm on the Rust side.

struct Modifier {
    kind: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    data0: vec4<f32>,
    data1: vec4<f32>,
    data2: vec4<f32>,
};

// Apply one modifier to a particle: accumulate into `accel` / `drag`, or set
// `killed`. A pure function of the modifier and the particle's position and
// velocity, so both simulate shaders share it.
fn eval_modifier(
    m: Modifier,
    pos: vec3<f32>,
    vel: vec3<f32>,
    time: f32,
    accel: ptr<function, vec3<f32>>,
    drag: ptr<function, f32>,
    killed: ptr<function, bool>,
) {
    switch m.kind {
        // Accel: constant acceleration (data0.xyz).
        case 0u: {
            *accel += m.data0.xyz;
        }
        // Drag: velocity-proportional damping coefficient (data0.x).
        case 1u: {
            *drag += m.data0.x;
        }
        // PointAttractor: softened inverse-square pull toward data0.xyz.
        case 2u: {
            let to = m.data0.xyz - pos;
            let dist = length(to);
            let soft = dist + m.data1.y;
            *accel += (to / max(dist, 1e-4)) * (m.data1.x / (soft * soft));
        }
        // CurlNoise: divergence-free turbulence.
        case 3u: {
            let domain = pos * m.data1.x + vec3<f32>(time * m.data1.z);
            *accel += curl_noise(domain) * m.data1.y;
        }
        // Radial: push out from (or, negative, pull toward) center data0.xyz.
        case 4u: {
            let to = pos - m.data0.xyz;
            let dist = length(to);
            let dir = to / max(dist, 1e-4);
            var mag = m.data1.x;
            if (m.data1.y > 0.0) {
                mag = mag / (1.0 + dist / m.data1.y);
            }
            *accel += dir * mag;
        }
        // Vortex: tangential swirl around axis data1.xyz through center data0.xyz.
        case 5u: {
            let rel = pos - m.data0.xyz;
            let axis = normalize(m.data1.xyz);
            let along = axis * dot(rel, axis);
            let dist = length(rel - along); // distance from the axis
            let tangent = normalize(cross(axis, rel));
            var mag = m.data2.x;
            if (m.data2.y > 0.0) {
                mag = mag / (1.0 + dist / m.data2.y);
            }
            *accel += tangent * mag;
        }
        // Conform: pull onto a sphere shell (center data0.xyz, radius data0.w)
        // within an influence band; sticky adds drag there.
        case 6u: {
            let delta = m.data0.xyz - pos; // center - pos
            let dist = length(delta);
            let dir = delta / max(dist, 1e-4);
            let shell = dist - m.data0.w; // + outside shell, - inside
            if (abs(shell) < m.data1.y) {
                *accel += dir * sign(shell) * m.data1.x;
                *drag += m.data1.z;
            }
        }
        // KillSphere: kill inside (data1.x) or outside a sphere.
        case 7u: {
            let inside = distance(pos, m.data0.xyz) < m.data0.w;
            let kill_inside = m.data1.x > 0.5;
            if (inside == kill_inside) {
                *killed = true;
            }
        }
        // KillAabb: kill inside (data2.x) or outside an axis-aligned box.
        case 8u: {
            let inside = all(pos >= m.data0.xyz) && all(pos <= m.data1.xyz);
            let kill_inside = m.data2.x > 0.5;
            if (inside == kill_inside) {
                *killed = true;
            }
        }
        default: {}
    }
}
