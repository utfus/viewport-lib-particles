//! Showcase 23: Expression op set + built-ins (Phase 13).
//!
//! A swirling "galaxy" authored entirely from the expression graph, exercising
//! the expanded op set and built-in inputs:
//!
//! - **`time`** drives a rolling rainbow: each particle's spawn colour is
//!   `0.5 + 0.5*sin(time + phase + offset)` per channel, so freshly spawned
//!   particles sweep through hue as the clock advances.
//! - **Vector constructors + swizzles** build a tangential swirl force from the
//!   particle's own position: `vec3(-position.y, position.x, 0)`.
//! - **`time` again, in the update kernel**, pulses the swirl strength as
//!   `0.5 + 0.5*sin(time * speed)`, multiplied by a live **runtime property** so
//!   the slider and the animated pulse compose.
//! - A linear spring toward the origin keeps the disc bound (safe at the origin,
//!   unlike a normalize).
//!
//! The program is type-checked with `EffectProgram::validate()` when built, so an
//! authoring mistake surfaces here rather than as a WGSL compile error.

use eframe::egui;
use viewport_lib_particles::{
    Attribute, EffectAsset, EffectId, EffectProgram, Module, ParticleBlend, ParticleItem,
    ParticleItems, PropertyValue, SpawnRate, UpdateOp,
};

/// A rainbow disc that swirls around the vertical axis, its swirl strength both
/// pulsing with `time` and scaled by the live `swirl` property.
fn expr_galaxy() -> EffectAsset {
    let mut m = Module::new();

    // --- spawn: spread across a disc, drifting outward slightly ---------------
    let origin = m.attr("origin");
    let spawn_dir = m.rand_unit();
    let radius = m.lit(1.2);
    let offset = m.mul(spawn_dir, radius);
    let position = m.add(origin, offset);

    let vel_dir = m.rand_unit();
    let speed = m.lit(0.4);
    let velocity = m.mul(vel_dir, speed);

    // --- spawn colour: rolling rainbow from time + a per-particle phase --------
    let time = m.time();
    let phase01 = m.rand();
    let tau = m.lit(6.2831855);
    let phase = m.mul(phase01, tau);
    let base = m.add(time, phase);
    let half = m.lit(0.5);
    // r/g/b = 0.5 + 0.5*sin(base + channel offset)
    let off_r = m.lit(0.0);
    let br = m.add(base, off_r);
    let sr = m.sin(br);
    let sr_h = m.mul(sr, half);
    let r = m.add(sr_h, half);
    let off_g = m.lit(2.0944); // 2π/3
    let bg = m.add(base, off_g);
    let sg = m.sin(bg);
    let sg_h = m.mul(sg, half);
    let g = m.add(sg_h, half);
    let off_b = m.lit(4.1888); // 4π/3
    let bb = m.add(base, off_b);
    let sb = m.sin(bb);
    let sb_h = m.mul(sb, half);
    let b = m.add(sb_h, half);
    let colour = m.vec3(r, g, b);

    let smin = m.lit(0.06);
    let smax = m.lit(0.14);
    let size_rand = m.rand();
    let size = m.mix(smin, smax, size_rand);

    // --- update: tangential swirl (vec constructor + swizzles) ----------------
    let pos = m.attr("position");
    let px = m.x(pos);
    let py = m.y(pos);
    let zero = m.lit(0.0);
    let neg = m.lit(-1.0);
    let neg_py = m.mul(py, neg);
    let tangent = m.vec3(neg_py, px, zero); // perpendicular in the XY plane

    // Swirl strength pulses with time and scales by the `swirl` property.
    let ut = m.time();
    let pulse_speed = m.lit(1.5);
    let pt = m.mul(ut, pulse_speed);
    let spt = m.sin(pt);
    let spt_h = m.mul(spt, half);
    let pulse = m.add(spt_h, half); // 0..1
    let swirl_prop = m.property("swirl");
    let swirl_k = m.mul(swirl_prop, pulse);
    let swirl_force = m.mul(tangent, swirl_k);

    // Linear spring toward the origin: -pull * position (zero at origin, safe).
    let pull_prop = m.property("pull");
    let neg_pull = m.mul(pull_prop, neg);
    let inward = m.mul(pos, neg_pull);

    let accel = m.add(swirl_force, inward);

    let program = EffectProgram::new()
        .with_module(m)
        .with_rate(SpawnRate::PerSecond(16_000.0))
        .with_lifetime(2.5, 3.5)
        .property("swirl", PropertyValue::F32(5.0))
        .property("pull", PropertyValue::F32(1.5))
        .set(Attribute::Position, position)
        .set(Attribute::Velocity, velocity)
        .set(Attribute::Colour, colour)
        .set(Attribute::Size, size)
        .update(UpdateOp::Accelerate(accel));

    // Phase 13: type-check the graph up front. A mistake (e.g. a scalar where a
    // vec3 is required) fails here with a node-level message, not in WGSL.
    debug_assert!(
        program.validate().is_ok(),
        "expr_galaxy failed validation: {:?}",
        program.validate()
    );

    EffectAsset::new("expr_galaxy")
        .with_capacity(60_000)
        .with_blend(ParticleBlend::Additive)
        .with_program(program)
}

/// The preset effects this showcase registers, in menu order.
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![("Expr galaxy", expr_galaxy())]
}

/// UI state for the expression-ops showcase.
pub struct State {
    pub selected: usize,
    pub paused: bool,
    pub position: [f32; 3],
    /// Live swirl strength, multiplied by the time pulse each frame.
    pub swirl: f32,
    /// Live inward-spring strength.
    pub pull: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            selected: 0,
            paused: false,
            position: [0.0, 0.0, 0.0],
            swirl: 5.0,
            pull: 1.5,
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
    ui.label("Swirl strength (property × time pulse):");
    ui.add(egui::Slider::new(&mut state.swirl, 0.0..=15.0).text("swirl"));
    ui.label("Inward spring:");
    ui.add(egui::Slider::new(&mut state.pull, 0.0..=6.0).text("pull"));
    ui.separator();
    ui.label("Authored from the expression graph (Phase 13):");
    ui.label("• time drives a rolling rainbow at spawn");
    ui.label("• vec3(-pos.y, pos.x, 0) builds the swirl (constructor + swizzle)");
    ui.label("• swirl = property × (0.5 + 0.5·sin(time))");
    ui.label("The graph is type-checked before it compiles to WGSL.");
}

/// Build this frame's submission for the selected preset.
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    if let Some(&id) = effect_ids.get(state.selected) {
        let mut item = ParticleItem::new(id)
            .at(state.position)
            .with_property("swirl", PropertyValue::F32(state.swirl))
            .with_property("pull", PropertyValue::F32(state.pull));
        item.settings.hidden = state.paused;
        items.push(item);
    }
    items
}
