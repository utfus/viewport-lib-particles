//! Showcase 20: Billboard rotation.
//!
//! Per-particle sprite rotation: each particle spawns at a random roll (a wide
//! birth variance) and spins at a constant rate advanced by its age. An arrow
//! sprite makes the spin legible. The spin rate is chosen from a few registered
//! variants (the rate is baked into the draw pipeline, so it is a discrete choice
//! rather than a live slider); tumbling embers show the same feature drawn small.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, Gradient, ParticleBlend, ParticleItem,
    ParticleItems, ParticlePlugin, SpawnRate, SpawnShape, TextureMode, VelocityDist,
};

/// A 32x32 white arrow (triangle) pointing up on a transparent field, so a
/// billboard's roll is legible.
fn arrow_rgba() -> (u32, u32, Vec<u8>) {
    let n = 32u32;
    let mut px = vec![0u8; (n * n * 4) as usize];
    for y in 0..n {
        for x in 0..n {
            let fx = x as f32 / (n - 1) as f32;
            let fy = y as f32 / (n - 1) as f32;
            let inside = (fx - 0.5).abs() < 0.45 * fy && fy > 0.08;
            let i = ((y * n + x) * 4) as usize;
            if inside {
                px[i] = 255;
                px[i + 1] = 255;
                px[i + 2] = 255;
                px[i + 3] = 255;
            }
        }
    }
    (n, n, px)
}

/// Spin rates (radians/sec) offered as registered variants.
pub const SPINS: [f32; 4] = [0.0, 3.0, 8.0, -6.0];

fn arrow_variant(tex: viewport_lib_particles::ParticleTextureId, spin: f32) -> EffectAsset {
    EffectAsset::new("rotation-arrow")
        .with_capacity(2_000)
        .with_blend(ParticleBlend::Alpha)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(20.0),
            lifetime: (1.8, 2.4),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.5,
                min_speed: 1.4,
                max_speed: 2.2,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.5,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -0.6]))
        .with_texture(tex)
        .with_texture_mode(TextureMode::Modulate)
        // Random birth roll (full turn) plus the chosen constant spin.
        .with_rotation(0.0, std::f32::consts::PI, spin)
        .with_gradient(
            Gradient::new().with_colour(vec![(0.0, [0.6, 0.9, 1.0]), (1.0, [0.2, 0.4, 0.8])]),
        )
}

/// Small tumbling shards: the same spin at a smaller scale. Textured with the
/// arrow sprite (an asymmetric shape) so the tumble reads; a round dot would show
/// no rotation at all.
fn shards(tex: viewport_lib_particles::ParticleTextureId, spin: f32) -> EffectAsset {
    EffectAsset::new("rotation-shards")
        .with_capacity(3_000)
        .with_blend(ParticleBlend::Alpha)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(45.0),
            lifetime: (1.4, 2.0),
            spawn: SpawnShape::Circle {
                radius: 0.5,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.4,
                min_speed: 0.8,
                max_speed: 1.6,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.22,
        })
        .force(ForceModifier::Accel([0.0, 0.0, 0.4]))
        .with_texture(tex)
        .with_texture_mode(TextureMode::Modulate)
        .with_rotation(0.0, std::f32::consts::PI, spin)
        .with_gradient(
            Gradient::new().with_colour(vec![(0.0, [1.0, 0.8, 0.4]), (1.0, [0.9, 0.4, 0.2])]),
        )
}

/// Registered ids: one arrow and one shard variant per spin rate.
pub struct Ids {
    pub arrows: Vec<EffectId>,
    pub shards: Vec<EffectId>,
}

pub fn register(
    plugin: &mut ParticlePlugin,
    device: &viewport_lib::wgpu::Device,
    queue: &viewport_lib::wgpu::Queue,
) -> Ids {
    let (w, h, px) = arrow_rgba();
    let tex = plugin.upload_texture(device, queue, w, h, &px);
    let arrows = SPINS
        .iter()
        .map(|&s| plugin.add_effect(device, arrow_variant(tex, s)))
        .collect();
    let shards = SPINS
        .iter()
        .map(|&s| plugin.add_effect(device, shards(tex, s)))
        .collect();
    Ids { arrows, shards }
}

/// UI state.
pub struct State {
    /// Index into [`SPINS`].
    pub spin: usize,
    /// Draw the small tumbling shards alongside the arrows.
    pub shards: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            spin: 2,
            shards: true,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Spin rate (radians / sec):");
    for (i, s) in SPINS.iter().enumerate() {
        ui.radio_value(&mut state.spin, i, format!("{s:+.0}"));
    }
    ui.separator();
    ui.checkbox(&mut state.shards, "Tumbling shards");
    ui.separator();
    ui.label("Each particle spawns at a random roll and spins at the chosen rate. Both use an asymmetric arrow sprite so the roll is visible (a round dot would show nothing).");
}

/// Submit the selected spin variant.
pub fn items(ids: &Ids, state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    let idx = state.spin.min(SPINS.len() - 1);
    if let Some(&id) = ids.arrows.get(idx) {
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
    }
    if state.shards {
        if let Some(&id) = ids.shards.get(idx) {
            items.push(ParticleItem::new(id).at([1.8, 0.0, 0.0]));
        }
    }
    items
}
