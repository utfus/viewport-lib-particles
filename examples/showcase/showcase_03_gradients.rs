//! Showcase 3: Gradients & curves.
//!
//! Colour-over-lifetime and size-over-lifetime ramps, baked into a lookup
//! texture the draw samples by particle age. The emitter colour is white so the
//! ramp colour comes through directly; the size ramp grows then shrinks each
//! particle over its life.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, Gradient, ParticleBlend, ParticleItem,
    ParticleItems, SpawnRate, SpawnShape, VelocityDist,
};

/// A rising fire plume: white-hot at birth, cooling to orange then dark; size
/// swells then fades.
fn fire() -> EffectAsset {
    EffectAsset::new("fire")
        .with_capacity(40_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(11_000.0),
            lifetime: (0.8, 1.6),
            spawn: SpawnShape::Sphere {
                radius: 0.25,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.4,
                min_speed: 2.0,
                max_speed: 4.0,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.3,
        })
        // Slight buoyancy up plus drag so the plume slows as it rises.
        .force(ForceModifier::Accel([0.0, 0.0, 1.0]))
        .force(ForceModifier::Drag(0.3))
        .with_gradient(
            Gradient::new()
                .with_colour(vec![
                    (0.0, [1.6, 1.4, 0.7]),
                    (0.4, [1.4, 0.5, 0.1]),
                    (1.0, [0.25, 0.04, 0.02]),
                ])
                .with_size(vec![(0.0, 0.5), (0.3, 1.2), (1.0, 0.15)]),
        )
}

/// A fountain of cooling sparks: yellow to red, shrinking to nothing, falling
/// under gravity.
fn sparks() -> EffectAsset {
    EffectAsset::new("sparks")
        .with_capacity(30_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(6_000.0),
            lifetime: (1.2, 2.0),
            spawn: SpawnShape::Sphere {
                radius: 0.1,
                volume: false,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.6,
                min_speed: 3.5,
                max_speed: 6.0,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.1,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -6.0]))
        .with_gradient(
            Gradient::new()
                .with_colour(vec![
                    (0.0, [1.6, 1.3, 0.5]),
                    (0.6, [1.2, 0.3, 0.05]),
                    (1.0, [0.2, 0.02, 0.0]),
                ])
                .with_size(vec![(0.0, 1.0), (1.0, 0.0)]),
        )
}

/// The preset effects this showcase registers, in menu order.
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![("Fire plume", fire()), ("Cooling sparks", sparks())]
}

/// UI state for the gradients showcase.
pub struct State {
    pub selected: usize,
    pub paused: bool,
    pub position: [f32; 3],
    /// When true, the ramp is rebuilt from the sliders below and applied as a
    /// per-frame gradient override (Phase 11 live gradient-key editing). When
    /// false, the preset's baked ramp is used.
    pub live_gradient: bool,
    /// Colour ramp endpoints (hue) and their HDR intensity multipliers.
    pub birth: [f32; 3],
    pub birth_intensity: f32,
    pub death: [f32; 3],
    pub death_intensity: f32,
    /// Mid-life size-scale peak.
    pub size_peak: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            selected: 0,
            paused: false,
            position: [0.0, 0.0, 0.0],
            live_gradient: false,
            birth: [1.0, 0.85, 0.4],
            birth_intensity: 1.6,
            death: [0.4, 0.06, 0.03],
            death_intensity: 0.7,
            size_peak: 1.2,
        }
    }
}

/// Left-panel controls.
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
    ui.checkbox(&mut state.live_gradient, "Live gradient keys");
    ui.add_enabled_ui(state.live_gradient, |ui| {
        ui.horizontal(|ui| {
            ui.color_edit_button_rgb(&mut state.birth);
            ui.label("birth");
        });
        ui.add(egui::Slider::new(&mut state.birth_intensity, 0.0..=3.0).text("birth glow"));
        ui.horizontal(|ui| {
            ui.color_edit_button_rgb(&mut state.death);
            ui.label("death");
        });
        ui.add(egui::Slider::new(&mut state.death_intensity, 0.0..=3.0).text("death glow"));
        ui.add(egui::Slider::new(&mut state.size_peak, 0.1..=2.5).text("size peak"));
    });
    ui.separator();
    ui.label("Colour and size follow a ramp over each particle's life,");
    ui.label("sampled from a lookup texture by normalized age.");
}

/// The ramp built from the live sliders (birth -> death colour, swelling size).
fn live_ramp(state: &State) -> Gradient {
    let scale = |c: [f32; 3], k: f32| [c[0] * k, c[1] * k, c[2] * k];
    Gradient::new()
        .with_colour(vec![
            (0.0, scale(state.birth, state.birth_intensity)),
            (1.0, scale(state.death, state.death_intensity)),
        ])
        .with_size(vec![(0.0, 0.5), (0.3, state.size_peak), (1.0, 0.15)])
}

/// Build this frame's submission for the selected preset.
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    if let Some(&id) = effect_ids.get(state.selected) {
        let mut item = ParticleItem::new(id).at(state.position);
        if state.live_gradient {
            item = item.with_gradient(live_ramp(state));
        }
        item.settings.hidden = state.paused;
        items.push(item);
    }
    items
}
