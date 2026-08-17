//! Showcase 14: Runtime lifecycle + deterministic seed (Phases 27 & 28).
//!
//! Phase 27: the effect's registered asset is hot-reloaded every frame from the
//! panel sliders via `ParticleItems::replace_effect` — a *structural-preserving*
//! change (only colour / rate / size differ), so the live stream recolours and
//! retimes without resetting. A "Remove" button reclaims the effect's GPU state
//! (one-way until restart). Phase 28: a seed makes spawning reproducible for a
//! fixed seed + `dt` sequence (`ParticleItem::with_seed`).

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, ParticleBlend, ParticleItem, ParticleItems,
    SpawnRate, SpawnShape, VelocityDist,
};

/// The base effect; its emitter is rebuilt from `State` and hot-reloaded.
fn base_asset(colour: [f32; 3], rate: f32, size: f32) -> EffectAsset {
    EffectAsset::new("editable")
        .with_capacity(20_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(rate),
            lifetime: (1.0, 1.8),
            spawn: SpawnShape::Sphere {
                radius: 0.15,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.4,
                min_speed: 2.0,
                max_speed: 4.0,
            },
            colour: [colour[0], colour[1], colour[2], 1.0],
            size,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -3.0]))
        .force(ForceModifier::Drag(0.05))
}

/// One effect, hot-reloaded per frame from the sliders.
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![("editable", base_asset([0.4, 0.8, 1.0], 8_000.0, 0.12))]
}

/// UI state.
pub struct State {
    pub colour: [f32; 3],
    pub rate: f32,
    pub size: f32,
    /// Phase 27: once true, the effect is removed (reclaimed) and stays gone.
    pub removed: bool,
    /// Phase 28: use a deterministic spawn seed.
    pub deterministic: bool,
    pub seed: u32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            colour: [0.4, 0.8, 1.0],
            rate: 8_000.0,
            size: 0.12,
            removed: false,
            deterministic: false,
            seed: 1,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Phase 27 — hot-reload (replace_effect):");
    ui.color_edit_button_rgb(&mut state.colour);
    ui.add(egui::Slider::new(&mut state.rate, 0.0..=20_000.0).text("rate"));
    ui.add(egui::Slider::new(&mut state.size, 0.02..=0.3).text("size"));
    ui.label("The live stream recolours / retimes without resetting.");
    ui.separator();
    if state.removed {
        ui.label("Effect removed (GPU state reclaimed). Restart to restore.");
    } else if ui.button("Remove effect").clicked() {
        state.removed = true;
    }
    ui.separator();
    ui.label("Phase 28 — deterministic seed:");
    ui.checkbox(&mut state.deterministic, "Seeded spawning");
    if state.deterministic {
        ui.add(egui::Slider::new(&mut state.seed, 1..=9999).text("seed"));
    }
}

/// Build this frame's submission, queuing a hot-reload (or a removal).
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    let Some(&id) = effect_ids.first() else {
        return items;
    };
    if state.removed {
        items.remove_effect(id);
        return items;
    }
    // Hot-reload the registered asset from the sliders (uniform-only, so the
    // particle stream is preserved).
    items.replace_effect(id, base_asset(state.colour, state.rate, state.size));
    let mut item = ParticleItem::new(id).at([0.0, 0.0, 0.0]);
    if state.deterministic {
        item = item.with_seed(state.seed as u64);
    }
    items.push(item);
    items
}
