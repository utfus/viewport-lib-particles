//! Showcase 17: Mesh spawn.
//!
//! Particles spawn from an uploaded mesh, either **area-weighted over its
//! surface** (`SpawnShape::MeshSurface`) or **from its vertices as a point
//! cloud** (`SpawnShape::PointCloud`), and lift off along the surface normal, like a
//! model dissolving / assembling into particles. A live slider drives the
//! lift-off speed via a per-frame emitter override (the baked sample pool is
//! reused, only the normal velocity changes).

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, Gradient, ParticleBlend, ParticleItem,
    ParticleItems, ParticleMeshId, ParticlePlugin, SpawnRate, SpawnShape, VelocityDist,
};

/// Build a torus mesh (positions + triangle indices) so the spawn surface is a
/// recognisable shape.
fn torus(major: f32, minor: f32, seg_u: u32, seg_v: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut positions = Vec::new();
    for i in 0..seg_u {
        let u = i as f32 / seg_u as f32 * std::f32::consts::TAU;
        let (cu, su) = (u.cos(), u.sin());
        for j in 0..seg_v {
            let v = j as f32 / seg_v as f32 * std::f32::consts::TAU;
            let (cv, sv) = (v.cos(), v.sin());
            // Z-up torus lying in the XY plane.
            let x = (major + minor * cv) * cu;
            let y = (major + minor * cv) * su;
            let z = minor * sv;
            positions.push([x, y, z]);
        }
    }
    let mut indices = Vec::new();
    for i in 0..seg_u {
        for j in 0..seg_v {
            let a = i * seg_v + j;
            let b = ((i + 1) % seg_u) * seg_v + j;
            let c = i * seg_v + (j + 1) % seg_v;
            let d = ((i + 1) % seg_u) * seg_v + (j + 1) % seg_v;
            indices.extend_from_slice(&[a, b, c, c, b, d]);
        }
    }
    (positions, indices)
}

fn base_emitter(spawn: SpawnShape) -> Emitter {
    Emitter {
        rate: SpawnRate::PerSecond(3000.0),
        lifetime: (0.8, 1.6),
        spawn,
        // The mesh normal lift-off supplies the outward motion; keep the base
        // velocity distribution small so the surface shape stays legible.
        velocity: VelocityDist::UniformCone {
            axis: [0.0, 0.0, 1.0],
            half_angle: std::f32::consts::PI,
            min_speed: 0.0,
            max_speed: 0.2,
        },
        colour: [1.0, 1.0, 1.0, 1.0],
        size: 0.03,
    }
}

fn surface_effect(mesh: ParticleMeshId) -> EffectAsset {
    EffectAsset::new("meshspawn-surface")
        .with_capacity(40_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(base_emitter(SpawnShape::MeshSurface {
            mesh,
            normal_velocity: 0.6,
        }))
        .force(ForceModifier::Drag(0.6))
        .with_gradient(
            Gradient::new().with_colour(vec![
                (0.0, [0.4, 1.4, 1.8]),
                (0.5, [0.2, 0.7, 1.2]),
                (1.0, [0.05, 0.1, 0.4]),
            ]),
        )
}

fn points_effect(mesh: ParticleMeshId) -> EffectAsset {
    EffectAsset::new("meshspawn-points")
        .with_capacity(40_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(base_emitter(SpawnShape::PointCloud {
            mesh,
            normal_velocity: 0.6,
        }))
        .force(ForceModifier::Drag(0.6))
        .with_gradient(
            Gradient::new().with_colour(vec![
                (0.0, [1.8, 1.2, 0.4]),
                (0.5, [1.2, 0.5, 0.15]),
                (1.0, [0.4, 0.1, 0.02]),
            ]),
        )
}

pub struct Ids {
    pub surface: EffectId,
    pub points: EffectId,
    mesh: ParticleMeshId,
}

pub fn register(plugin: &mut ParticlePlugin, device: &viewport_lib::wgpu::Device) -> Ids {
    let (positions, indices) = torus(1.2, 0.45, 48, 24);
    // Upload once; the surface effect samples the triangles, the point-cloud
    // effect samples the vertices (indices ignored).
    let mesh = plugin.upload_mesh(device, &positions, &indices);
    let surface = plugin.add_effect(device, surface_effect(mesh));
    let points = plugin.add_effect(device, points_effect(mesh));
    Ids {
        surface,
        points,
        mesh,
    }
}

/// UI state.
pub struct State {
    /// Draw the point-cloud variant instead of the surface variant.
    pub points: bool,
    /// Normal lift-off speed (live-tunable via a per-frame emitter override).
    pub lift: f32,
    /// Emission rate multiplier.
    pub rate: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            points: false,
            lift: 0.6,
            rate: 1.0,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.checkbox(&mut state.points, "Point cloud (vertices)");
    ui.add(egui::Slider::new(&mut state.lift, 0.0..=3.0).text("normal lift-off"));
    ui.add(egui::Slider::new(&mut state.rate, 0.1..=3.0).text("emission rate"));
    ui.separator();
    ui.label("Particles spawn over an uploaded torus (area-weighted across its surface, or from its vertices) and lift off along the surface normal.");
}

/// Submit the selected variant with the live lift / rate override.
pub fn items(ids: &Ids, state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    let (id, shape) = if state.points {
        (
            ids.points,
            SpawnShape::PointCloud {
                mesh: ids.mesh,
                normal_velocity: state.lift,
            },
        )
    } else {
        (
            ids.surface,
            SpawnShape::MeshSurface {
                mesh: ids.mesh,
                normal_velocity: state.lift,
            },
        )
    };
    let mut e = base_emitter(shape);
    if let SpawnRate::PerSecond(r) = e.rate {
        e.rate = SpawnRate::PerSecond(r * state.rate);
    }
    items.push(ParticleItem::new(id).with_emitter(e));
    items
}
