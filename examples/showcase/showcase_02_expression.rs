//! Showcase 2: Expression effects.
//!
//! Effects authored through the expression graph and modifier stack, compiled
//! to per-effect emit/simulate WGSL. These do things the fixed-function emitter
//! cannot: per-particle random colour, and an update force that reads the
//! particle's own position (a spring pulling it back toward the emitter).

use eframe::egui;
use viewport_lib_particles::{
    Attribute, EffectAsset, EffectId, EffectProgram, Module, ParticleBlend, ParticleItem,
    ParticleItems, SpawnRate, UpdateOp,
};

/// A cloud of randomly-tinted particles launched isotropically and sprung back
/// toward the emitter, so they orbit and mix instead of flying away.
fn random_hue_cloud() -> EffectAsset {
    let mut m = Module::new();

    // Colour = colA + (colB - colA) * rand, per particle.
    let ca = m.lit_vec3([1.0, 0.4, 0.1]);
    let cb = m.lit_vec3([0.3, 0.6, 1.0]);
    let r = m.rand();
    let sr = m.splat3(r);
    let diff = m.sub(cb, ca);
    let scaled = m.mul(diff, sr);
    let colour = m.add(ca, scaled);

    // Velocity = random unit direction * speed.
    let dir = m.rand_unit();
    let speed = m.lit(2.5);
    let velocity = m.mul(dir, speed);

    let size = m.lit(0.12);

    // Update: spring toward the emitter, accel = (origin - position) * k.
    let origin = m.attr("origin");
    let position = m.attr("position");
    let toward = m.sub(origin, position);
    let k = m.lit(6.0);
    let spring = m.mul(toward, k);

    let mut program = EffectProgram::new()
        .with_rate(SpawnRate::PerSecond(9_000.0))
        .with_lifetime(1.8, 2.8);
    program.module = m;
    let program = program
        .set(Attribute::Colour, colour)
        .set(Attribute::Velocity, velocity)
        .set(Attribute::Size, size)
        .update(UpdateOp::Accelerate(spring));

    EffectAsset::new("random_hue_cloud")
        .with_capacity(40_000)
        .with_blend(ParticleBlend::Additive)
        .with_program(program)
}

/// The preset effects this showcase registers, in menu order.
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![("Random-hue cloud", random_hue_cloud())]
}

/// UI state for the expression showcase.
pub struct State {
    pub selected: usize,
    pub paused: bool,
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
    ui.label("These effects are compiled from an expression graph:");
    ui.label("random per-particle colour and a position-reading spring");
    ui.label("force, neither of which the fixed emitter can express.");
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
