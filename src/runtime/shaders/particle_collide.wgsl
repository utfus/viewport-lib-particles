// 2D collision against a host collider set, for the fixed-function simulate
// kernels. Prepended after particle_plane.wgsl.
//
// The set is binned into a uniform grid in plane coordinates: each cell holds a
// start and a count into an index list of edges. A step's swept segment is
// tested against the edges in the cells its bounding box covers, at most
// MAX_SWEEP_CELLS a side, so a fast particle does not pass through a thin wall.

struct CollideCell {
    start: u32,
    count: u32,
};

@group(0) @binding(3) var<storage, read> collider_edges: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> collider_cells: array<CollideCell>;
@group(0) @binding(5) var<storage, read> collider_index: array<u32>;

const MAX_SWEEP_CELLS: i32 = 8;

// Where along the step `p0 -> p1` it crosses the edge `a -> b`, from 0 to 1;
// 2 when it does not.
fn step_crossing(p0: vec2<f32>, p1: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = p1 - p0;
    let e = b - a;
    let denom = d.x * e.y - d.y * e.x;
    if (abs(denom) < 1e-12) {
        return 2.0;
    }
    let w = a - p0;
    let t = (w.x * e.y - w.y * e.x) / denom;
    let u = (w.x * d.y - w.y * d.x) / denom;
    if (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0) {
        return t;
    }
    return 2.0;
}

// The first edge the step `p0 -> p1` crosses: `(t, normal)`, with the normal
// facing the side the step came from; t is 2 when it crosses none. `grid` is
// the grid origin and cell size, `dims` the cells across and down and the edge
// count.
fn collide_step(p0: vec2<f32>, p1: vec2<f32>, grid: vec4<f32>, dims: vec4<u32>) -> vec3<f32> {
    var best = vec3<f32>(2.0, 0.0, 0.0);
    if (dims.z == 0u) {
        return best;
    }
    let lo = min(p0, p1);
    let hi = max(p0, p1);
    let top = grid.xy + grid.zw * vec2<f32>(dims.xy);
    if (any(hi < grid.xy) || any(lo > top)) {
        return best;
    }
    let last = vec2<i32>(dims.xy) - vec2<i32>(1);
    let c0 = clamp(vec2<i32>(floor((lo - grid.xy) / grid.zw)), vec2<i32>(0), last);
    var c1 = clamp(vec2<i32>(floor((hi - grid.xy) / grid.zw)), vec2<i32>(0), last);
    c1 = min(c1, c0 + vec2<i32>(MAX_SWEEP_CELLS - 1));
    for (var cy = c0.y; cy <= c1.y; cy = cy + 1) {
        for (var cx = c0.x; cx <= c1.x; cx = cx + 1) {
            let cell = collider_cells[u32(cy) * dims.x + u32(cx)];
            for (var k = 0u; k < cell.count; k = k + 1u) {
                let e = collider_edges[collider_index[cell.start + k]];
                let t = step_crossing(p0, p1, e.xy, e.zw);
                if (t < best.x) {
                    let dir = e.zw - e.xy;
                    var n = normalize(vec2<f32>(-dir.y, dir.x));
                    if (dot(n, p0 - e.xy) < 0.0) {
                        n = -n;
                    }
                    best = vec3<f32>(t, n);
                }
            }
        }
    }
    return best;
}

// Collide one particle's step from `start` to its new position, in the lock
// plane. `collide` is the response (mode 1 bounce, 2 kill, 3 stick; then
// restitution and friction). Returns true on contact.
fn collide_particle(
    p: ptr<function, Particle>,
    start: vec3<f32>,
    lock: vec4<f32>,
    collide: vec4<f32>,
    grid: vec4<f32>,
    dims: vec4<u32>,
) -> bool {
    if (collide.x < 0.5 || lock.x < 0.5) {
        return false;
    }
    let ya = plane_y_axis(lock);
    let q0 = vec2<f32>(start.x, dot(start, ya));
    let q1 = vec2<f32>((*p).position.x, dot((*p).position, ya));
    let hit = collide_step(q0, q1, grid, dims);
    if (hit.x > 1.0) {
        return false;
    }
    let n = hit.yz;
    // Stop just short of the edge, on the side the step came from.
    let q = mix(q0, q1, hit.x) + n * 1e-3;
    (*p).position = lock_position(plane_offset(lock, q.x, q.y), lock);
    let n3 = plane_offset(lock, n.x, n.y);
    let mode = u32(collide.x + 0.5);
    if (mode == 2u) {
        (*p).lifetime = 0.0;
    } else if (mode == 3u) {
        (*p).velocity = vec3<f32>(0.0, 0.0, 0.0);
        (*p).pad.y = 1.0; // stuck: the kernels only age it from now on
    } else {
        let v = (*p).velocity;
        let vn = dot(v, n3);
        if (vn < 0.0) {
            let tangent = v - n3 * vn;
            (*p).velocity = tangent * (1.0 - collide.z) - n3 * vn * collide.y;
        }
    }
    return true;
}
