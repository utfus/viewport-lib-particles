// One bitonic-sort stage over the draw-order index buffer.
//
// The host dispatches this once per (k, j) stage with the stage constants in a
// dynamic-offset uniform, each in its own compute pass so writes from the prior
// stage are visible. Sorts the order indices by descending key, so the draw
// reads far particles first.

struct SortParams {
    k: u32,   // outer bitonic sequence size
    j: u32,   // compare distance
    n: u32,   // padded length
    pad: u32,
};

@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(1) var<storage, read_write> order: array<u32>;
@group(0) @binding(2) var<uniform> sp: SortParams;

@compute @workgroup_size(64)
fn sort_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sp.n) {
        return;
    }
    let l = i ^ sp.j;
    // Only the lower index of each compare pair does the swap, so no two
    // invocations write the same slot.
    if (l <= i) {
        return;
    }
    let a = order[i];
    let b = order[l];
    let ka = keys[a];
    let kb = keys[b];
    // Ascending sub-sequences when (i & k) == 0; invert for a descending final
    // order (largest key first).
    let ascending = (i & sp.k) == 0u;
    let swap = select(ka > kb, ka < kb, ascending);
    if (swap) {
        order[i] = b;
        order[l] = a;
    }
}
