//! Showcase 10: Simulation space.
//!
//! A single emitter sweeps back and forth. With **global** (world-space)
//! simulation the already-spawned particles stay where they were emitted, so the
//! cloud smears into a trail behind the moving emitter; with **local** simulation
//! the whole cloud rigidly follows the emitter and stays a compact blob. The
//! simulation space is fixed on the `EffectAsset`, so two effects are registered
//! (one of each space) and the toggle chooses which is submitted.
//!
//! The spawn-shape selector also exercises the [`SpawnShape::Circle`] and
//! [`SpawnShape::Cone`] shapes live, applied through the per-frame emitter
//! override.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, ParticleBlend, ParticleItem, ParticleItems,
    SimulationSpace, SpawnRate, SpawnShape, VelocityDist,
};

/// The four selectable spawn shapes, in menu order.
const SHAPES: &[&str] = &["Point", "Sphere", "Circle", "Cone"];

fn spawn_shape(i: usize) -> SpawnShape {
    match i {
        1 => SpawnShape::Sphere {
            radius: 0.25,
            volume: true,
        },
        2 => SpawnShape::Circle {
            radius: 0.6,
            volume: true,
        },
        3 => SpawnShape::Cone {
            base_radius: 0.5,
            top_radius: 0.05,
            height: 1.2,
            volume: true,
        },
        _ => SpawnShape::Point,
    }
}

fn base_emitter(shape: usize) -> Emitter {
    Emitter {
        rate: SpawnRate::PerSecond(4_000.0),
        lifetime: (1.2, 1.8),
        spawn: spawn_shape(shape),
        // Near-zero velocity so the trailing / following distinction is about the
        // simulation space, not the particles' own motion.
        velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
        colour: [1.0, 0.6, 0.2, 1.0],
        size: 0.09,
    }
}

fn base_effect(space: SimulationSpace) -> EffectAsset {
    EffectAsset::new("simspace")
        .with_capacity(30_000)
        .with_blend(ParticleBlend::Additive)
        .with_simulation_space(space)
        .with_emitter(base_emitter(0))
        .force(ForceModifier::Drag(0.1))
}

/// Two presets: the same emitter simulated in global and in local space.
/// `items` submits index 0 (global) or 1 (local) per the toggle.
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![
        ("global", base_effect(SimulationSpace::Global)),
        ("local", base_effect(SimulationSpace::Local)),
    ]
}

/// UI state for the simulation-space showcase.
pub struct State {
    /// `true` submits the local-space effect, `false` the global one.
    pub local: bool,
    /// When true, the emitter sweeps automatically; otherwise the slider drives
    /// its x position.
    pub auto_move: bool,
    /// Index into [`SHAPES`] of the current spawn shape.
    pub shape: usize,
    /// Internal sweep clock, advanced by the app each frame.
    pub t: f32,
    /// Manual emitter x when `auto_move` is off.
    pub manual_x: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            local: false,
            auto_move: true,
            shape: 1,
            t: 0.0,
            manual_x: 0.0,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Simulation space:");
    ui.selectable_value(&mut state.local, false, "Global (world space)");
    ui.selectable_value(&mut state.local, true, "Local (follows emitter)");
    ui.separator();
    ui.checkbox(&mut state.auto_move, "Auto-sweep emitter");
    if !state.auto_move {
        ui.add(egui::Slider::new(&mut state.manual_x, -3.0..=3.0).text("emitter x"));
    }
    ui.separator();
    ui.label("Spawn shape:");
    for (i, name) in SHAPES.iter().enumerate() {
        ui.selectable_value(&mut state.shape, i, *name);
    }
    ui.separator();
    ui.label(if state.local {
        "Local: the cloud follows the emitter."
    } else {
        "Global: particles stay put, so a trail forms."
    });
}

/// The emitter x position this frame.
fn emitter_x(state: &State) -> f32 {
    if state.auto_move {
        2.5 * (state.t * 0.9).sin()
    } else {
        state.manual_x
    }
}

/// Build this frame's submission.
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    let idx = usize::from(state.local);
    if let Some(&id) = effect_ids.get(idx) {
        let item = ParticleItem::new(id)
            .at([emitter_x(state), 0.0, 0.0])
            .with_emitter(base_emitter(state.shape));
        items.push(item);
    }
    items
}
