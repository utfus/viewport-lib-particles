// Plane lock helpers, shared by the fixed-function kernels and generated ones.
//
// A lock is a vec4: x is 0 for none, 1 for the XY plane, 2 for the XZ plane;
// y is the depth, the elevation along the plane normal. The planes match
// viewport-lib's `WorldPlane`: XY has normal +Z, XZ has normal -Y.

fn plane_normal(lock: vec4<f32>) -> vec3<f32> {
    if (lock.x > 1.5) {
        return vec3<f32>(0.0, -1.0, 0.0);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

// World direction of plane y: +Y on XY (and with no lock), +Z on XZ.
fn plane_y_axis(lock: vec4<f32>) -> vec3<f32> {
    if (lock.x > 1.5) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

// A world offset for plane coordinates (x, y).
fn plane_offset(lock: vec4<f32>, x: f32, y: f32) -> vec3<f32> {
    return vec3<f32>(x, 0.0, 0.0) + plane_y_axis(lock) * y;
}

// Put a position on the locked plane at its depth.
fn lock_position(p: vec3<f32>, lock: vec4<f32>) -> vec3<f32> {
    if (lock.x < 0.5) {
        return p;
    }
    let n = plane_normal(lock);
    return p - n * (dot(p, n) - lock.y);
}

// Drop a vector's component along the locked plane's normal.
fn lock_vector(v: vec3<f32>, lock: vec4<f32>) -> vec3<f32> {
    if (lock.x < 0.5) {
        return v;
    }
    let n = plane_normal(lock);
    return v - n * dot(v, n);
}
