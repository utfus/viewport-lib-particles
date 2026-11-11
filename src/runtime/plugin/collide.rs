//! 2D collider sets: the host's level geometry, binned into a uniform grid on
//! the CPU and uploaded as three storage buffers the simulate kernels read.
//!
//! The grid is a compact cell list: each cell holds a start and a count into an
//! index list of edges, so a cell has no edge cap. A step's swept segment is
//! tested against the edges in the cells its bounding box covers, at most
//! `MAX_SWEEP_CELLS` a side (see `particle_collide.wgsl`).

use viewport_lib::wgpu as vwgpu;

use crate::effect::{Collider2dSet, Collider2dSetId};

use super::*;

/// Most cells a side the grid has.
const MAX_GRID_CELLS: u32 = 64;

/// A collider set's edges binned into a grid, as the simulate kernels read it.
#[derive(Debug, Default)]
pub(crate) struct ColliderGrid {
    /// `[a.x, a.y, b.x, b.y]` per edge.
    pub(crate) edges: Vec<[f32; 4]>,
    /// `[start, count]` into `index` per cell, row by row.
    pub(crate) cells: Vec<[u32; 2]>,
    /// Edge indices, grouped by cell.
    pub(crate) index: Vec<u32>,
    /// Grid origin and cell size, in plane coordinates.
    pub(crate) origin: [f32; 2],
    pub(crate) cell: [f32; 2],
    /// Cells across and down.
    pub(crate) dims: [u32; 2],
}

/// Cost figures for a registered collider set, from
/// [`ParticlePlugin::collider_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColliderStats {
    /// Edges in the set.
    pub edges: u32,
    /// Grid cells across and down.
    pub cells: [u32; 2],
    /// The most edges any one cell holds: the per-cell cost bound of a
    /// collision test, times the cells a step crosses.
    pub max_edges_per_cell: u32,
}

impl ColliderGrid {
    pub(crate) fn stats(&self) -> ColliderStats {
        ColliderStats {
            edges: self.edges.len() as u32,
            cells: self.dims,
            max_edges_per_cell: self.cells.iter().map(|c| c[1]).max().unwrap_or(0),
        }
    }
}

/// Bin `set`'s edges into a grid of at most `MAX_GRID_CELLS` a side over their
/// bounds, sized for about two edges a cell. An edge is listed in every cell
/// its bounding box touches.
pub(crate) fn build_collider_grid(set: &Collider2dSet) -> ColliderGrid {
    let edges: Vec<[f32; 4]> = set
        .edges()
        .iter()
        .map(|e| [e.a[0], e.a[1], e.b[0], e.b[1]])
        .collect();
    if edges.is_empty() {
        return ColliderGrid::default();
    }
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for e in &edges {
        for k in 0..2 {
            lo[k] = lo[k].min(e[k]).min(e[k + 2]);
            hi[k] = hi[k].max(e[k]).max(e[k + 2]);
        }
    }
    // A margin so edges on the bounds sit inside cells, and a floor on the
    // extent so a set of collinear edges still has an area.
    let extent = [(hi[0] - lo[0]).max(1e-3), (hi[1] - lo[1]).max(1e-3)];
    let margin = extent[0].max(extent[1]) * 1e-3 + 1e-3;
    let origin = [lo[0] - margin, lo[1] - margin];
    let span = [extent[0] + 2.0 * margin, extent[1] + 2.0 * margin];
    // About two edges a cell over the area, square cells where the cap allows.
    let target = (span[0] * span[1] / (edges.len() as f32 * 0.5)).sqrt();
    let dims = [
        ((span[0] / target).ceil() as u32).clamp(1, MAX_GRID_CELLS),
        ((span[1] / target).ceil() as u32).clamp(1, MAX_GRID_CELLS),
    ];
    let cell = [span[0] / dims[0] as f32, span[1] / dims[1] as f32];

    let to_cell = |v: f32, k: usize| {
        (((v - origin[k]) / cell[k]).floor() as i64).clamp(0, dims[k] as i64 - 1) as u32
    };
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); (dims[0] * dims[1]) as usize];
    for (i, e) in edges.iter().enumerate() {
        let (x0, x1) = (to_cell(e[0].min(e[2]), 0), to_cell(e[0].max(e[2]), 0));
        let (y0, y1) = (to_cell(e[1].min(e[3]), 1), to_cell(e[1].max(e[3]), 1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                lists[(y * dims[0] + x) as usize].push(i as u32);
            }
        }
    }
    let mut cells = Vec::with_capacity(lists.len());
    let mut index = Vec::new();
    for list in lists {
        cells.push([index.len() as u32, list.len() as u32]);
        index.extend(list);
    }
    ColliderGrid {
        edges,
        cells,
        index,
        origin,
        cell,
        dims,
    }
}

/// A storage buffer of at least `bytes`, never empty.
fn storage(device: &vwgpu::Device, label: &str, bytes: u64) -> vwgpu::Buffer {
    device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage: vwgpu::BufferUsages::STORAGE | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl ColliderSetGpu {
    /// Buffers sized for `grid`, unwritten.
    pub(crate) fn new(device: &vwgpu::Device, grid: &ColliderGrid) -> Self {
        let sizes = grid_bytes(grid);
        Self {
            edges: storage(device, "particle_collider_edges", sizes[0]),
            cells: storage(device, "particle_collider_cells", sizes[1]),
            index: storage(device, "particle_collider_index", sizes[2]),
            capacity: sizes,
            generation: 0,
            header: grid_header(grid),
            stats: grid.stats(),
        }
    }

    /// Upload `grid`, growing the buffers (and bumping `generation`, so bind
    /// groups over the old ones rebuild) when it does not fit.
    pub(crate) fn write(&mut self, device: &vwgpu::Device, queue: &vwgpu::Queue, grid: &ColliderGrid) {
        let sizes = grid_bytes(grid);
        if (0..3).any(|k| sizes[k] > self.capacity[k]) {
            let generation = self.generation + 1;
            *self = Self::new(device, grid);
            self.generation = generation;
        }
        queue.write_buffer(&self.edges, 0, bytemuck::cast_slice(&grid.edges));
        queue.write_buffer(&self.cells, 0, bytemuck::cast_slice(&grid.cells));
        queue.write_buffer(&self.index, 0, bytemuck::cast_slice(&grid.index));
        self.header = grid_header(grid);
        self.stats = grid.stats();
    }
}

fn grid_bytes(grid: &ColliderGrid) -> [u64; 3] {
    [
        (grid.edges.len() * 16) as u64,
        (grid.cells.len() * 8) as u64,
        (grid.index.len() * 4) as u64,
    ]
}

/// The grid lanes of `SimParamsGpu`: origin and cell size, then cells across,
/// cells down and the edge count (0 skips collision).
fn grid_header(grid: &ColliderGrid) -> ([f32; 4], [u32; 4]) {
    (
        [grid.origin[0], grid.origin[1], grid.cell[0], grid.cell[1]],
        [grid.dims[0], grid.dims[1], grid.edges.len() as u32, 0],
    )
}

impl ParticlePlugin {
    /// Register a collider set for effects to collide with (see
    /// [`Collide2d`](crate::Collide2d)). The shapes are binned and uploaded
    /// now; replace them with
    /// [`replace_collider_set`](Self::replace_collider_set) or
    /// [`ParticleItems::replace_collider_set`] when the level changes.
    pub fn add_collider_set(
        &mut self,
        device: &vwgpu::Device,
        queue: &vwgpu::Queue,
        set: &Collider2dSet,
    ) -> Collider2dSetId {
        let grid = build_collider_grid(set);
        let mut gpu = ColliderSetGpu::new(device, &grid);
        gpu.write(device, queue, &grid);
        self.collider_sets.push(gpu);
        Collider2dSetId(self.collider_sets.len() as u32 - 1)
    }

    /// Replace a collider set's shapes. Effects colliding with it see the new
    /// shapes from the next simulated step. A no-op for an unknown id.
    pub fn replace_collider_set(
        &mut self,
        device: &vwgpu::Device,
        queue: &vwgpu::Queue,
        id: Collider2dSetId,
        set: &Collider2dSet,
    ) {
        let grid = build_collider_grid(set);
        if let Some(gpu) = self.collider_sets.get_mut(id.0 as usize) {
            gpu.write(device, queue, &grid);
        }
    }

    /// Cost figures for a registered collider set.
    pub fn collider_stats(&self, id: Collider2dSetId) -> Option<ColliderStats> {
        self.collider_sets.get(id.0 as usize).map(|g| g.stats)
    }

    /// The collider set effect `idx` collides with, when it collides: a set it
    /// names that exists, under a plane lock.
    pub(crate) fn effect_collider(&self, idx: usize) -> Option<(usize, &ColliderSetGpu)> {
        let collide = self.effects.get(idx)?.asset.collide?;
        self.effective_plane_lock(idx)?;
        let set = collide.set.0 as usize;
        self.collider_sets.get(set).map(|g| (set, g))
    }

    /// The three collider buffers effect `idx`'s simulate bind group binds:
    /// its set's, or the shared empty ones.
    pub(crate) fn collider_buffers(&self, idx: usize) -> [&vwgpu::Buffer; 3] {
        let g = match self.effect_collider(idx) {
            Some((_, g)) => g,
            None => self.default_collider.as_ref().unwrap(),
        };
        [&g.edges, &g.cells, &g.index]
    }

    /// The set and generation effect `idx`'s simulate bind groups were built
    /// against, so they rebuild when the set's buffers grow.
    pub(crate) fn collider_binding(&self, idx: usize) -> Option<(usize, u32)> {
        self.effect_collider(idx).map(|(set, g)| (set, g.generation))
    }

    /// Rebuild the simulate bind groups of instances whose collider set grew
    /// (new buffers) since they were built.
    pub(crate) fn refresh_collider_bindings(&mut self, device: &vwgpu::Device) {
        let stale: Vec<InstanceKey> = self
            .instances
            .iter()
            .filter(|(k, inst)| inst.collider != self.collider_binding(k.effect as usize))
            .map(|(k, _)| *k)
            .collect();
        for ikey in stale {
            let idx = ikey.effect as usize;
            let binding = self.collider_binding(idx);
            let sim_bgl = self.sim_bgl.clone().unwrap();
            let parent_bgl = self.sim_parent_bgl.clone().unwrap();
            let buffers: [vwgpu::Buffer; 3] = self.collider_buffers(idx).map(|b| b.clone());
            let buffers = [&buffers[0], &buffers[1], &buffers[2]];
            let Some(inst) = self.instances.get_mut(&ikey) else {
                continue;
            };
            inst.collider = binding;
            let EffectGpu::Fixed(g) = &mut inst.gpu else {
                continue;
            };
            g.sim_bg = build_sim_bg(device, &sim_bgl, &g.particle_buf, &g.sim_buf, None, buffers);
            if let Some(se) = self.subemitters.get_mut(&idx) {
                se.parent_sim_bg = build_sim_bg(
                    device,
                    &parent_bgl,
                    &g.particle_buf,
                    &g.sim_buf,
                    Some(&se.event_buf),
                    buffers,
                );
            }
        }
    }
}

/// A fixed-function simulate bind group: particles (0), sim params (1), the
/// parent's event buffer (2, parent kernel only) and the collider edges, cells
/// and index (3, 4, 5).
pub(crate) fn build_sim_bg(
    device: &vwgpu::Device,
    bgl: &vwgpu::BindGroupLayout,
    particle_buf: &vwgpu::Buffer,
    sim_buf: &vwgpu::Buffer,
    event_buf: Option<&vwgpu::Buffer>,
    colliders: [&vwgpu::Buffer; 3],
) -> vwgpu::BindGroup {
    let mut entries = vec![
        vwgpu::BindGroupEntry {
            binding: 0,
            resource: particle_buf.as_entire_binding(),
        },
        vwgpu::BindGroupEntry {
            binding: 1,
            resource: sim_buf.as_entire_binding(),
        },
    ];
    if let Some(events) = event_buf {
        entries.push(vwgpu::BindGroupEntry {
            binding: 2,
            resource: events.as_entire_binding(),
        });
    }
    for (k, buf) in colliders.iter().enumerate() {
        entries.push(vwgpu::BindGroupEntry {
            binding: 3 + k as u32,
            resource: buf.as_entire_binding(),
        });
    }
    device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some(if event_buf.is_some() {
            "particle_sim_parent_bg"
        } else {
            "particle_sim_bg"
        }),
        layout: bgl,
        entries: &entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::{Collider2d, ColliderEdge};

    #[test]
    fn every_edge_is_listed_in_the_cells_it_touches() {
        let mut shapes = vec![Collider2d::Segment {
            a: [0.0, 0.0],
            b: [100.0, 0.0],
        }];
        for i in 0..40 {
            let x = i as f32 * 2.5;
            shapes.push(Collider2d::Edges(vec![ColliderEdge {
                a: [x, 10.0],
                b: [x + 1.0, 11.0],
            }]));
        }
        let grid = build_collider_grid(&Collider2dSet::new(shapes));
        assert_eq!(grid.edges.len(), 41);
        assert!(grid.dims[0] > 1 && grid.dims[0] <= MAX_GRID_CELLS);
        // The long floor runs through every cell of the bottom row.
        for x in 0..grid.dims[0] {
            let [start, count] = grid.cells[x as usize];
            let listed = &grid.index[start as usize..(start + count) as usize];
            assert!(listed.contains(&0), "the floor should be in bottom cell {x}");
        }
        // Each short edge is listed somewhere, in a cell its box overlaps.
        for i in 1..41u32 {
            assert!(grid.index.contains(&i), "edge {i} should be binned");
        }
        assert!(grid.stats().max_edges_per_cell >= 1);
    }

    #[test]
    fn an_empty_set_has_no_cells() {
        let grid = build_collider_grid(&Collider2dSet::default());
        assert!(grid.edges.is_empty() && grid.cells.is_empty());
        assert_eq!(grid_header(&grid).1[2], 0, "a zero edge count skips collision");
    }
}
