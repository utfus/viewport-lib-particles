//! Showcase 12: Per-instance simulation (Phase 24).
//!
//! One registered "brazier" effect drives an N×N grid of **independent**
//! instances, each keyed by `ParticleItem::with_key` and simulated at its own
//! anchor — twenty braziers are twenty distinct fires, not one drawn twenty
//! times. Toggles hide one instance (it vanishes while the rest burn on) and
//! wiggle one (it carries its own particles; its neighbours are undisturbed),
//! demonstrating that per-instance state is truly independent.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, Gradient, ParticleBlend, ParticleItem,
    ParticleItems, SpawnRate, SpawnShape, VelocityDist,
};

fn brazier() -> EffectAsset {
    EffectAsset::new("brazier")
        .with_capacity(2_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(700.0),
            lifetime: (0.6, 1.0),
            spawn: SpawnShape::Sphere {
                radius: 0.08,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.4,
                min_speed: 0.7,
                max_speed: 1.3,
            },
            colour: [1.0, 0.5, 0.15, 1.0],
            size: 0.08,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -0.5]))
        .with_gradient(
            Gradient::new().with_colour(vec![(0.0, [1.0, 0.8, 0.3]), (1.0, [1.0, 0.15, 0.05])]),
        )
}

/// One effect; the grid of instances is built per frame in `items`.
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![("brazier", brazier())]
}

/// UI state for the per-instance showcase.
pub struct State {
    /// Grid side length; `grid * grid` independent instances.
    pub grid: usize,
    /// Anchor spacing in world units.
    pub spacing: f32,
    /// Hide the first instance (it freezes / vanishes; the rest are unaffected).
    pub hide_one: bool,
    /// Sweep the first instance back and forth (it carries its own particles).
    pub move_one: bool,
    /// Internal clock, advanced by the app each frame.
    pub t: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            grid: 4,
            spacing: 1.6,
            hide_one: false,
            move_one: false,
            t: 0.0,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.add(egui::Slider::new(&mut state.grid, 1..=6).text("grid (N×N)"));
    ui.add(egui::Slider::new(&mut state.spacing, 0.8..=3.0).text("spacing"));
    ui.separator();
    ui.label(format!("{} independent instances", state.grid * state.grid));
    ui.separator();
    ui.checkbox(&mut state.hide_one, "Hide one (others burn on)");
    ui.checkbox(&mut state.move_one, "Move one (neighbours undisturbed)");
}

/// Build this frame's grid submission — one keyed instance per anchor.
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    let Some(&id) = effect_ids.first() else {
        return items;
    };
    let n = state.grid.max(1);
    let half = (n as f32 - 1.0) * 0.5;
    let mut key = 0u64;
    for iy in 0..n {
        for ix in 0..n {
            let mut x = (ix as f32 - half) * state.spacing;
            let y = (iy as f32 - half) * state.spacing;
            if key == 0 && state.move_one {
                x += (state.t * 1.5).sin() * state.spacing * 0.9;
            }
            let mut item = ParticleItem::new(id).with_key(key).at([x, y, 0.0]);
            if key == 0 && state.hide_one {
                item.settings.hidden = true;
            }
            items.push(item);
            key += 1;
        }
    }
    items
}
