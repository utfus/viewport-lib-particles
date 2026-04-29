//! Showcase 1: Emitters.
//!
//! Demonstrates the Phase 2 fixed-function feature set through a set of preset
//! effects: point / box / sphere spawn shapes, fixed / cone velocity, gravity /
//! drag / attractor forces, and additive / alpha blend. Every preset is
//! registered at startup (effects cannot be added after the plugin is handed to
//! the renderer); the UI switches which one is submitted.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, ParticleBlend, ParticleItem, ParticleItems,
    SpawnRate, SpawnShape, VelocityDist,
};

/// The preset effects this showcase registers, in menu order. Returned to the
/// app at startup so it can register each one and keep the resulting
/// [`EffectId`]s aligned by index.
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![
        (
            "Additive fountain",
            EffectAsset::new("fountain")
                .with_capacity(40_000)
                .with_blend(ParticleBlend::Additive)
                .with_emitter(Emitter {
                    rate: SpawnRate::PerSecond(12_000.0),
                    lifetime: (1.2, 2.4),
                    spawn: SpawnShape::Sphere {
                        radius: 0.2,
                        volume: true,
                    },
                    velocity: VelocityDist::UniformCone {
                        axis: [0.0, 0.0, 1.0],
                        half_angle: 0.35,
                        min_speed: 3.0,
                        max_speed: 5.5,
                    },
                    colour: [1.0, 0.55, 0.15, 1.0],
                    size: 0.14,
                })
                // Z-up: gravity pulls along -Z.
                .force(ForceModifier::Accel([0.0, 0.0, -4.0]))
                .force(ForceModifier::Drag(0.05)),
        ),
        (
            "Isotropic burst",
            EffectAsset::new("burst")
                .with_capacity(30_000)
                .with_blend(ParticleBlend::Additive)
                .with_emitter(Emitter {
                    rate: SpawnRate::PerSecond(9_000.0),
                    lifetime: (1.0, 2.0),
                    spawn: SpawnShape::Sphere {
                        radius: 0.15,
                        volume: false,
                    },
                    // Full-sphere cone (half-angle pi) approximates radial spread.
                    velocity: VelocityDist::UniformCone {
                        axis: [0.0, 0.0, 1.0],
                        half_angle: std::f32::consts::PI,
                        min_speed: 1.5,
                        max_speed: 3.5,
                    },
                    colour: [0.3, 0.7, 1.0, 1.0],
                    size: 0.13,
                })
                .force(ForceModifier::Drag(0.4)),
        ),
        (
            "Rain (alpha)",
            EffectAsset::new("rain")
                .with_capacity(24_000)
                .with_blend(ParticleBlend::Alpha)
                .with_emitter(Emitter {
                    rate: SpawnRate::PerSecond(6_000.0),
                    lifetime: (1.4, 1.8),
                    spawn: SpawnShape::Box {
                        min: [-4.0, -4.0, 4.0],
                        max: [4.0, 4.0, 5.0],
                    },
                    velocity: VelocityDist::Fixed([0.0, 0.0, -6.0]),
                    colour: [0.7, 0.8, 1.0, 0.6],
                    size: 0.06,
                })
                .force(ForceModifier::Accel([0.0, 0.0, -3.0])),
        ),
        (
            "Attractor swirl",
            EffectAsset::new("swirl")
                .with_capacity(40_000)
                .with_blend(ParticleBlend::Additive)
                .with_emitter(Emitter {
                    rate: SpawnRate::PerSecond(14_000.0),
                    lifetime: (2.0, 3.5),
                    spawn: SpawnShape::Point,
                    velocity: VelocityDist::UniformCone {
                        axis: [1.0, 0.0, 0.2],
                        half_angle: 0.5,
                        min_speed: 3.0,
                        max_speed: 5.0,
                    },
                    colour: [0.85, 0.4, 1.0, 1.0],
                    size: 0.11,
                })
                .force(ForceModifier::PointAttractor {
                    position: [0.0, 0.0, 1.5],
                    strength: 12.0,
                    falloff: 0.6,
                })
                .force(ForceModifier::Drag(0.08)),
        ),
    ]
}

/// UI state for the emitters showcase.
pub struct State {
    /// Index into [`presets`] of the effect being shown.
    pub selected: usize,
    /// When true, the effect is submitted `hidden` so it stops emitting/drawing.
    pub paused: bool,
    /// Emitter origin in world space (moved by the sliders).
    pub position: [f32; 3],
}

impl Default for State {
    fn default() -> Self {
        Self {
            selected: 0,
            paused: false,
            position: [0.0, 0.0, 0.0],
        }
    }
}

/// Left-panel controls for the emitters showcase.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Preset:");
    let names: Vec<&'static str> = presets().into_iter().map(|(n, _)| n).collect();
    for (i, name) in names.iter().enumerate() {
        ui.selectable_value(&mut state.selected, i, *name);
    }
    ui.separator();
    ui.checkbox(&mut state.paused, "Pause");
    ui.separator();
    ui.label("Emitter position:");
    ui.add(egui::Slider::new(&mut state.position[0], -5.0..=5.0).text("x"));
    ui.add(egui::Slider::new(&mut state.position[1], -5.0..=5.0).text("y"));
    ui.add(egui::Slider::new(&mut state.position[2], -2.0..=5.0).text("z"));
    ui.separator();
    ui.label("Each preset is a separate registered effect.");
    ui.label("Live emitter tuning (rate, lifetime, ...) needs per-frame");
    ui.label("emitter params; see the plan.");
}

/// Build this frame's submission for the selected preset.
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    if let Some(&id) = effect_ids.get(state.selected) {
        let mut item = ParticleItem::new(id).at(state.position);
        item.settings.hidden = state.paused;
        items.push(item);
    }
    items
}
