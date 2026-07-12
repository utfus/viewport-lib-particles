//! Showcase 8: Turbulence.
//!
//! A rising cloud stirred by a divergence-free curl-noise field, giving the
//! smooth folding, swirling flow of smoke or plasma. Scale / strength / speed
//! sliders drive the `CurlNoise` force live each frame through the per-frame
//! force override, so the turbulence can be tuned while it runs.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, Gradient, ParticleBlend, ParticleItem,
    ParticleItems, SpawnRate, SpawnShape, VelocityDist,
};

/// The turbulence effect: an additive cloud with a cool blue-to-magenta ramp,
/// drifting up. Forces are supplied per frame from the sliders, so the asset
/// itself carries none.
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    let effect = EffectAsset::new("turbulence")
        .with_capacity(60_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(18_000.0),
            lifetime: (2.0, 3.5),
            spawn: SpawnShape::Sphere {
                radius: 0.35,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.5,
                min_speed: 0.4,
                max_speed: 1.0,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.16,
        })
        .with_gradient(
            Gradient::new()
                .with_colour(vec![
                    (0.0, [0.4, 0.7, 1.4]),
                    (0.5, [0.6, 0.3, 1.2]),
                    (1.0, [0.1, 0.02, 0.15]),
                ])
                .with_size(vec![(0.0, 0.6), (0.4, 1.2), (1.0, 0.3)]),
        );
    vec![("Curl-noise cloud", effect)]
}

/// UI state for the turbulence showcase.
pub struct State {
    pub paused: bool,
    pub position: [f32; 3],
    /// Curl-noise field frequency (higher = finer, more chaotic).
    pub scale: f32,
    /// Curl-noise acceleration magnitude.
    pub strength: f32,
    /// How fast the field scrolls over time.
    pub speed: f32,
    /// Upward buoyancy.
    pub buoyancy: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            paused: false,
            position: [0.0, 0.0, -1.0],
            scale: 0.8,
            strength: 6.0,
            speed: 0.5,
            buoyancy: 1.5,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.checkbox(&mut state.paused, "Pause");
    ui.separator();
    ui.label("Curl noise:");
    ui.add(egui::Slider::new(&mut state.scale, 0.2..=3.0).text("scale"));
    ui.add(egui::Slider::new(&mut state.strength, 0.0..=25.0).text("strength"));
    ui.add(egui::Slider::new(&mut state.speed, 0.0..=2.5).text("speed"));
    ui.add(egui::Slider::new(&mut state.buoyancy, 0.0..=5.0).text("buoyancy"));
    if ui.button("Reset").clicked() {
        *state = State::default();
    }
    ui.separator();
    ui.label("Emitter position:");
    ui.add(egui::Slider::new(&mut state.position[0], -5.0..=5.0).text("x"));
    ui.add(egui::Slider::new(&mut state.position[1], -5.0..=5.0).text("y"));
    ui.add(egui::Slider::new(&mut state.position[2], -3.0..=5.0).text("z"));
    ui.separator();
    ui.label("A divergence-free curl-noise field advects the cloud, so it");
    ui.label("folds and swirls like a fluid instead of just spreading.");
}

/// Build this frame's submission, driving the curl-noise force from the sliders.
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    if let Some(&id) = effect_ids.first() {
        let mut item = ParticleItem::new(id).at(state.position).with_forces(vec![
            ForceModifier::Accel([0.0, 0.0, state.buoyancy]),
            ForceModifier::Drag(0.15),
            ForceModifier::CurlNoise {
                scale: state.scale,
                strength: state.strength,
                speed: state.speed,
            },
        ]);
        item.settings.hidden = state.paused;
        items.push(item);
    }
    items
}
