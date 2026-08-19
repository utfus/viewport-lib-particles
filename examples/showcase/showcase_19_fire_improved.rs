//! Showcase 19: Fire (improved) — the layered fire rebuilt on Phases 9, 15, 22.
//!
//! Same campfire idea as "15: Fire", now using the features that landed since:
//! - **Phase 22 (mesh-surface spawn):** flames, smoke, and embers spawn over the
//!   surface of two crossed logs and lift off along their normals, so the fire
//!   conforms to the wood instead of a generic cone.
//! - **Phase 15 (HDR colour + per-key alpha):** black-body HDR ramps that bloom,
//!   and per-key alpha so each particle fades *in* at birth and out at death
//!   (smoother than the lifetime-fade-only look).
//! - **Phase 9 (soft particles):** flames and smoke fade against the logs and the
//!   ground (soft additive for the flame, so it keeps its glow) instead of
//!   showing a hard clip line where they meet geometry.
//!
//! `main.rs` submits the ground plane, the logs (as an opaque occluder for the
//! soft fade), and a sun light for this mode.

use eframe::egui;
use viewport_lib::renderer::SceneRenderItem;
use viewport_lib::resources::MeshData;
use viewport_lib::scene::material::{Material, ShadingModel};
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, Gradient, ParticleBlend, ParticleItem,
    ParticleItems, ParticleMeshId, ParticleTextureId, ParticlePlugin, SpawnRate, SpawnShape,
    TextureMode, VelocityDist,
};

// --- Procedural wisp texture ------------------------------------------------
// A soft round blob broken up by fractal value noise, so billboards read as
// irregular fluffy puffs instead of clean circles.

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn hash2(x: f32, y: f32) -> f32 {
    let h = (x * 127.1 + y * 311.7).sin() * 43758.5453;
    h - h.floor()
}

fn vnoise(x: f32, y: f32) -> f32 {
    let (xi, yi) = (x.floor(), y.floor());
    let (xf, yf) = (x - xi, y - yi);
    let u = xf * xf * (3.0 - 2.0 * xf);
    let v = yf * yf * (3.0 - 2.0 * yf);
    let a = hash2(xi, yi);
    let b = hash2(xi + 1.0, yi);
    let c = hash2(xi, yi + 1.0);
    let d = hash2(xi + 1.0, yi + 1.0);
    let ab = a + (b - a) * u;
    let cd = c + (d - c) * u;
    ab + (cd - ab) * v
}

fn fbm(mut x: f32, mut y: f32) -> f32 {
    let mut f = 0.0;
    let mut amp = 0.5;
    for _ in 0..4 {
        f += amp * vnoise(x, y);
        x *= 2.0;
        y *= 2.0;
        amp *= 0.5;
    }
    f
}

/// A 64x64 white RGBA sprite whose alpha is a soft blob modulated by noise.
fn wisp_rgba() -> (u32, u32, Vec<u8>) {
    let n = 64u32;
    let mut px = vec![0u8; (n * n * 4) as usize];
    for y in 0..n {
        for x in 0..n {
            let u = x as f32 / (n - 1) as f32;
            let v = y as f32 / (n - 1) as f32;
            let d = 2.0 * ((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt();
            // Soft round core, eroded at the rim and broken up by noise.
            let core = 1.0 - smoothstep(0.15, 1.0, d);
            let noise = fbm(u * 5.0, v * 5.0);
            let a = (core * (0.3 + 1.0 * noise) * smoothstep(1.05, 0.45, d)).clamp(0.0, 1.0);
            let i = ((y * n + x) * 4) as usize;
            px[i] = 255;
            px[i + 1] = 255;
            px[i + 2] = 255;
            px[i + 3] = (a * 255.0) as u8;
        }
    }
    (n, n, px)
}

// --- Crossed-logs geometry (shared by the plugin spawn mesh and the occluder) -

fn push_box(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
    c: [f32; 3],
    h: [f32; 3],
) {
    let base = positions.len() as u32;
    let corners = [
        [c[0] - h[0], c[1] - h[1], c[2] - h[2]],
        [c[0] + h[0], c[1] - h[1], c[2] - h[2]],
        [c[0] + h[0], c[1] + h[1], c[2] - h[2]],
        [c[0] - h[0], c[1] + h[1], c[2] - h[2]],
        [c[0] - h[0], c[1] - h[1], c[2] + h[2]],
        [c[0] + h[0], c[1] - h[1], c[2] + h[2]],
        [c[0] + h[0], c[1] + h[1], c[2] + h[2]],
        [c[0] - h[0], c[1] + h[1], c[2] + h[2]],
    ];
    for p in corners {
        positions.push(p);
        // Per-vertex normal from the box center: fine for a dark, matte log.
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-4);
        normals.push([d[0] / l, d[1] / l, d[2] / l]);
    }
    let tris = [
        0u32, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 3, 7, 7, 4, 0, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7,
        3, 0, 4, 5, 5, 1, 0,
    ];
    indices.extend(tris.iter().map(|i| base + i));
}

/// Two crossed logs resting on the ground (z = 0), each ~0.3 tall.
fn logs_geometry() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    let (mut p, mut n, mut i) = (Vec::new(), Vec::new(), Vec::new());
    push_box(&mut p, &mut n, &mut i, [0.0, 0.0, 0.16], [1.25, 0.16, 0.16]);
    push_box(&mut p, &mut n, &mut i, [0.0, 0.0, 0.16], [0.16, 1.25, 0.16]);
    (p, n, i)
}

/// The logs as a scene mesh for the renderer (opaque occluder the fire fades into).
pub fn logs_mesh_data() -> MeshData {
    let (positions, normals, indices) = logs_geometry();
    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}

/// A dark-brown dirt ground for the campfire (uses the shared ground cuboid,
/// top surface at z = 0). Its own item so the grey interaction/soft ground is
/// untouched.
pub fn ground_item(mesh: viewport_lib::MeshId) -> SceneRenderItem {
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model =
        glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.25)).to_cols_array_2d();
    item.material = Material::from_colour([0.14, 0.09, 0.05]); // dark brown dirt
    item.material.shading_model = ShadingModel::Pbr;
    item.material.roughness = 0.98;
    item
}

/// A dark-wood scene item for the uploaded logs mesh.
pub fn logs_item(mesh: viewport_lib::MeshId) -> SceneRenderItem {
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.material = Material::from_colour([0.09, 0.06, 0.04]);
    item.material.shading_model = ShadingModel::Pbr;
    item.material.roughness = 0.95;
    item
}

// --- Particle effects -------------------------------------------------------

fn flame_emitter(logs: ParticleMeshId) -> Emitter {
    Emitter {
        rate: SpawnRate::PerSecond(2600.0),
        lifetime: (0.45, 0.9),
        spawn: SpawnShape::MeshSurface {
            mesh: logs,
            normal_velocity: 0.8,
        },
        velocity: VelocityDist::UniformCone {
            axis: [0.0, 0.0, 1.0],
            half_angle: 0.5,
            min_speed: 0.3,
            max_speed: 0.9,
        },
        colour: [1.0, 1.0, 1.0, 1.0],
        size: 0.3,
    }
}

fn flame(logs: ParticleMeshId, wisp: ParticleTextureId) -> EffectAsset {
    EffectAsset::new("fire2-flame")
        .with_capacity(28_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(flame_emitter(logs))
        .with_texture(wisp)
        .with_texture_mode(TextureMode::Modulate)
        .force(ForceModifier::Accel([0.0, 0.0, 2.4])) // buoyancy
        .force(ForceModifier::Drag(0.8))
        .force(ForceModifier::CurlNoise {
            scale: 2.0,
            strength: 2.2,
            speed: 0.9,
        })
        // Soft additive: keeps the emissive glow while fading against the logs.
        .with_soft(0.22)
        .with_gradient(
            Gradient::new()
                .with_colour(vec![
                    (0.0, [1.7, 1.4, 0.75]),   // hot white-yellow (HDR)
                    (0.25, [1.5, 0.7, 0.18]),  // orange
                    (0.55, [1.0, 0.24, 0.05]), // red
                    (0.85, [0.35, 0.06, 0.02]),
                    (1.0, [0.05, 0.01, 0.0]),
                ])
                // Fade in fast at the wood, hold, then fade out at the tip.
                .with_alpha(vec![(0.0, 0.0), (0.1, 1.0), (0.6, 0.85), (1.0, 0.0)])
                .with_size(vec![(0.0, 0.5), (0.3, 1.0), (1.0, 0.2)]),
        )
}

fn smoke_emitter(logs: ParticleMeshId) -> Emitter {
    Emitter {
        rate: SpawnRate::PerSecond(70.0),
        lifetime: (1.5, 2.6),
        spawn: SpawnShape::MeshSurface {
            mesh: logs,
            normal_velocity: 0.35,
        },
        velocity: VelocityDist::UniformCone {
            axis: [0.0, 0.0, 1.0],
            half_angle: 0.4,
            min_speed: 0.7,
            max_speed: 1.2,
        },
        colour: [0.2, 0.2, 0.22, 0.5],
        size: 0.45,
    }
}

fn smoke(logs: ParticleMeshId, wisp: ParticleTextureId) -> EffectAsset {
    EffectAsset::new("fire2-smoke")
        .with_capacity(8_000)
        .with_blend(ParticleBlend::Alpha)
        .with_emitter(smoke_emitter(logs))
        .with_texture(wisp)
        .with_texture_mode(TextureMode::Modulate)
        .force(ForceModifier::Accel([0.0, 0.0, 0.7]))
        .force(ForceModifier::Drag(0.4))
        .force(ForceModifier::CurlNoise {
            scale: 0.7,
            strength: 1.2,
            speed: 0.4,
        })
        // Soft over-blend: smoke dissolves into the logs and ground.
        .with_soft(0.45)
        .with_gradient(
            Gradient::new()
                .with_colour(vec![(0.0, [0.18, 0.18, 0.2]), (1.0, [0.07, 0.07, 0.08])])
                .with_alpha(vec![(0.0, 0.0), (0.3, 0.55), (1.0, 0.0)])
                .with_size(vec![(0.0, 0.4), (1.0, 1.9)]),
        )
}

fn embers_emitter(logs: ParticleMeshId) -> Emitter {
    Emitter {
        rate: SpawnRate::PerSecond(130.0),
        lifetime: (0.8, 1.6),
        spawn: SpawnShape::MeshSurface {
            mesh: logs,
            normal_velocity: 1.2,
        },
        velocity: VelocityDist::UniformCone {
            axis: [0.0, 0.0, 1.0],
            half_angle: 0.6,
            min_speed: 0.6,
            max_speed: 2.2,
        },
        colour: [1.0, 1.0, 1.0, 1.0],
        size: 0.045,
    }
}

fn embers(logs: ParticleMeshId) -> EffectAsset {
    EffectAsset::new("fire2-embers")
        .with_capacity(4_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(embers_emitter(logs))
        .force(ForceModifier::Accel([0.0, 0.0, 0.6]))
        .force(ForceModifier::Drag(0.2))
        .force(ForceModifier::CurlNoise {
            scale: 2.2,
            strength: 2.0,
            speed: 1.0,
        })
        .with_gradient(
            Gradient::new()
                .with_colour(vec![
                    (0.0, [2.2, 1.5, 0.5]),
                    (0.5, [1.6, 0.5, 0.1]),
                    (1.0, [0.5, 0.05, 0.0]),
                ])
                .with_alpha(vec![(0.0, 0.0), (0.15, 1.0), (1.0, 0.0)])
                .with_size(vec![(0.0, 1.0), (1.0, 0.0)]),
        )
}

pub struct Ids {
    pub flame: EffectId,
    pub smoke: EffectId,
    pub embers: EffectId,
    logs: ParticleMeshId,
}

pub fn register(
    plugin: &mut ParticlePlugin,
    device: &viewport_lib::wgpu::Device,
    queue: &viewport_lib::wgpu::Queue,
) -> Ids {
    let (positions, _normals, indices) = logs_geometry();
    let logs = plugin.upload_mesh(device, &positions, &indices);
    let (w, h, px) = wisp_rgba();
    let wisp = plugin.upload_texture(device, queue, w, h, &px);
    Ids {
        flame: plugin.add_effect(device, flame(logs, wisp)),
        smoke: plugin.add_effect(device, smoke(logs, wisp)),
        embers: plugin.add_effect(device, embers(logs)),
        logs,
    }
}

/// UI state.
pub struct State {
    /// Emission-rate multiplier applied to every layer.
    pub intensity: f32,
    /// Horizontal wind (x acceleration) bending the fire.
    pub wind: f32,
    /// Draw the smoke layer.
    pub smoke: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            wind: 0.0,
            smoke: true,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.add(egui::Slider::new(&mut state.intensity, 0.2..=2.5).text("intensity"));
    ui.add(egui::Slider::new(&mut state.wind, -3.0..=3.0).text("wind"));
    ui.checkbox(&mut state.smoke, "Smoke");
    ui.separator();
    ui.label("Fire spawned over crossed logs (mesh surface), HDR + per-key alpha, soft-fading into the wood and ground.");
}

fn scaled(mut e: Emitter, mult: f32) -> Emitter {
    if let SpawnRate::PerSecond(r) = e.rate {
        e.rate = SpawnRate::PerSecond(r * mult);
    }
    e
}

/// Submit all three layers, applying the live controls.
pub fn items(ids: &Ids, state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    // Flame: live intensity + wind via per-frame overrides (the mesh spawn shape
    // is preserved, only rate / forces change).
    items.push(
        ParticleItem::new(ids.flame)
            .with_emitter(scaled(flame_emitter(ids.logs), state.intensity))
            .with_forces(vec![
                ForceModifier::Accel([state.wind, 0.0, 2.4]),
                ForceModifier::Drag(0.8),
                ForceModifier::CurlNoise {
                    scale: 2.0,
                    strength: 2.2,
                    speed: 0.9,
                },
            ]),
    );
    if state.smoke {
        items.push(
            ParticleItem::new(ids.smoke)
                .with_emitter(scaled(smoke_emitter(ids.logs), state.intensity))
                .with_forces(vec![
                    ForceModifier::Accel([state.wind * 1.5, 0.0, 0.7]),
                    ForceModifier::Drag(0.4),
                    ForceModifier::CurlNoise {
                        scale: 0.7,
                        strength: 1.2,
                        speed: 0.4,
                    },
                ]),
        );
    }
    items.push(
        ParticleItem::new(ids.embers)
            .with_emitter(scaled(embers_emitter(ids.logs), state.intensity)),
    );
    items
}
