//! Showcase 21: Timed bursts.
//!
//! One effect holds a schedule of timed bursts on top of its continuous rate, so
//! a single emitter fires staged pops. Here three bursts fire at t = 0, 0.35, and
//! 0.7 s after the effect starts. The whole staged sequence is submitted once as a
//! fire-and-forget transient and replays on a loop, so you see the three pops in
//! succession each cycle.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, Gradient, ParticleBlend, ParticleItem,
    ParticleItems, SpawnRate, SpawnShape, VelocityDist,
};

/// A staged spark explosion: no continuous rate, three timed bursts.
fn staged() -> EffectAsset {
    EffectAsset::new("staged-explosion")
        .with_capacity(3_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(0.0),
            lifetime: (0.5, 0.8),
            spawn: SpawnShape::Sphere {
                radius: 0.05,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: std::f32::consts::PI, // full sphere
                min_speed: 2.0,
                max_speed: 4.0,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.07,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -1.5]))
        .add_burst(300, 0.0)
        .add_burst(300, 0.35)
        .add_burst(300, 0.7)
        .with_gradient(
            Gradient::new()
                .with_colour(vec![(0.0, [1.0, 0.95, 0.6]), (1.0, [1.0, 0.3, 0.1])])
                .with_size(vec![(0.0, 1.0), (1.0, 0.0)]),
        )
}

pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![("staged-explosion", staged())]
}

/// UI state.
pub struct State {
    /// Seconds between replays of the staged sequence.
    pub period: f32,
    /// Internal clock, advanced by the app each frame.
    pub t: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            period: 2.0,
            t: 0.0,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.add(egui::Slider::new(&mut state.period, 1.0..=5.0).text("replay period (s)"));
    ui.separator();
    ui.label("One emitter fires three bursts at t = 0, 0.35, 0.7 s. The staged sequence is submitted once as a transient and replays each period.");
}

/// Fire the staged sequence once each time the timer crosses a `period` boundary.
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    let Some(&id) = effect_ids.first() else {
        return items;
    };
    let period = state.period.max(0.1);
    let prev = ((state.t - dt).max(0.0) / period).floor();
    let now = (state.t / period).floor();
    // Fire on the very first frame so the demo plays immediately, then once each
    // time the timer crosses a `period` boundary.
    let first_frame = state.t <= dt;
    if first_frame || now > prev {
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.5]).burst());
    }
    items
}
