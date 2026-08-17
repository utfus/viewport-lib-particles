//! Showcase 15: Realistic fire.
//!
//! A layered campfire built entirely from parts the system already ships — no
//! bespoke shaders:
//! - **flame** — a dense additive cone with a black-body colour ramp
//!   (hot white-yellow core → orange → deep red → out) and a grow-then-shrink
//!   size ramp, stirred by curl noise so it licks. HDR-bright (values > 1) so it
//!   blooms on a bloom-enabled renderer.
//! - **smoke** — sparse alpha-blended puffs that rise above the flame, grow, and
//!   fade, on coarser slower curl noise.
//! - **embers** — tiny bright additive sparks flung up and swirled, fading out.
//!
//! Live controls scale the emission rate, the turbulence, and a horizontal wind.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, Gradient, ParticleBlend, ParticleItem,
    ParticleItems, SpawnRate, SpawnShape, VelocityDist,
};

fn flame_emitter() -> Emitter {
    Emitter {
        rate: SpawnRate::PerSecond(2400.0),
        lifetime: (0.5, 1.1),
        spawn: SpawnShape::Cone {
            base_radius: 0.35,
            top_radius: 0.12,
            height: 0.2,
            volume: true,
        },
        velocity: VelocityDist::UniformCone {
            axis: [0.0, 0.0, 1.0],
            half_angle: 0.25,
            min_speed: 1.2,
            max_speed: 2.4,
        },
        colour: [1.0, 1.0, 1.0, 1.0],
        size: 0.5,
    }
}

fn flame() -> EffectAsset {
    EffectAsset::new("fire-flame")
        .with_capacity(24_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(flame_emitter())
        .force(ForceModifier::Accel([0.0, 0.0, 1.6])) // buoyancy
        .force(ForceModifier::Drag(0.6))
        .force(ForceModifier::CurlNoise {
            scale: 1.6,
            strength: 2.6,
            speed: 0.7,
        })
        .with_gradient(
            Gradient::new()
                .with_colour(vec![
                    (0.0, [1.6, 1.35, 0.7]),  // hot white-yellow (HDR)
                    (0.25, [1.5, 0.7, 0.18]), // orange
                    (0.55, [1.0, 0.25, 0.05]), // red
                    (0.85, [0.35, 0.06, 0.02]),
                    (1.0, [0.05, 0.01, 0.0]),
                ])
                .with_size(vec![(0.0, 0.5), (0.25, 1.0), (1.0, 0.15)]),
        )
}

fn smoke_emitter() -> Emitter {
    Emitter {
        // Sparse and wispy: real smoke is thin, translucent grey, and lives
        // *above* the flame, not over it. Submitted at a raised origin (see
        // `items`) so it rises from the flame tip rather than covering the core.
        rate: SpawnRate::PerSecond(55.0),
        lifetime: (1.3, 2.2),
        spawn: SpawnShape::Cone {
            base_radius: 0.12,
            top_radius: 0.06,
            height: 0.25,
            volume: true,
        },
        velocity: VelocityDist::UniformCone {
            axis: [0.0, 0.0, 1.0],
            half_angle: 0.22,
            min_speed: 0.8,
            max_speed: 1.4,
        },
        // Low alpha (w) keeps it translucent; a mid grey reads as smoke rather
        // than a black hole over the dark background.
        colour: [0.14, 0.14, 0.15, 0.16],
        size: 0.4,
    }
}

fn smoke() -> EffectAsset {
    EffectAsset::new("fire-smoke")
        .with_capacity(8_000)
        .with_blend(ParticleBlend::Alpha)
        .with_emitter(smoke_emitter())
        .force(ForceModifier::Accel([0.0, 0.0, 0.8]))
        .force(ForceModifier::Drag(0.4))
        .force(ForceModifier::CurlNoise {
            scale: 0.8,
            strength: 1.4,
            speed: 0.4,
        })
        .with_gradient(
            Gradient::new()
                // Fade the tint up as it rises and thins; keep it grey, not black.
                .with_colour(vec![
                    (0.0, [0.16, 0.16, 0.17]),
                    (0.5, [0.10, 0.10, 0.11]),
                    (1.0, [0.05, 0.05, 0.06]),
                ])
                .with_size(vec![(0.0, 0.4), (1.0, 1.8)]),
        )
}

fn embers_emitter() -> Emitter {
    Emitter {
        rate: SpawnRate::PerSecond(140.0),
        lifetime: (0.8, 1.6),
        spawn: SpawnShape::Sphere {
            radius: 0.18,
            volume: true,
        },
        velocity: VelocityDist::UniformCone {
            axis: [0.0, 0.0, 1.0],
            half_angle: 0.5,
            min_speed: 1.5,
            max_speed: 3.5,
        },
        colour: [1.0, 1.0, 1.0, 1.0],
        size: 0.05,
    }
}

fn embers() -> EffectAsset {
    EffectAsset::new("fire-embers")
        .with_capacity(4_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(embers_emitter())
        .force(ForceModifier::Accel([0.0, 0.0, 0.4]))
        .force(ForceModifier::Drag(0.2))
        .force(ForceModifier::CurlNoise {
            scale: 2.2,
            strength: 2.0,
            speed: 1.0,
        })
        .with_gradient(
            Gradient::new()
                .with_colour(vec![
                    (0.0, [2.0, 1.4, 0.5]),
                    (0.5, [1.6, 0.5, 0.1]),
                    (1.0, [0.5, 0.05, 0.0]),
                ])
                .with_size(vec![(0.0, 1.0), (1.0, 0.0)]),
        )
}

/// Three layers, in draw order (flame, smoke, embers).
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![
        ("fire-flame", flame()),
        ("fire-smoke", smoke()),
        ("fire-embers", embers()),
    ]
}

/// UI state.
pub struct State {
    /// Emission-rate multiplier applied to every layer.
    pub intensity: f32,
    /// Curl-noise strength on the flame (how much it licks).
    pub turbulence: f32,
    /// Horizontal wind (x acceleration) bending the flame.
    pub wind: f32,
    /// Draw the smoke layer.
    pub smoke: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            turbulence: 2.6,
            wind: 0.0,
            smoke: true,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.add(egui::Slider::new(&mut state.intensity, 0.2..=2.5).text("intensity"));
    ui.add(egui::Slider::new(&mut state.turbulence, 0.0..=6.0).text("turbulence"));
    ui.add(egui::Slider::new(&mut state.wind, -3.0..=3.0).text("wind"));
    ui.checkbox(&mut state.smoke, "Smoke");
    ui.separator();
    ui.label("Additive HDR flame + curl-noise turbulence + rising smoke + embers.");
}

fn scaled(mut e: Emitter, mult: f32) -> Emitter {
    if let SpawnRate::PerSecond(r) = e.rate {
        e.rate = SpawnRate::PerSecond(r * mult);
    }
    e
}

/// Submit all three layers at the origin, applying the live controls.
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    if let Some(&id) = effect_ids.first() {
        // Flame: live turbulence + wind via a per-frame force override.
        items.push(
            ParticleItem::new(id)
                .with_emitter(scaled(flame_emitter(), state.intensity))
                .with_forces(vec![
                    ForceModifier::Accel([state.wind, 0.0, 1.6]),
                    ForceModifier::Drag(0.6),
                    ForceModifier::CurlNoise {
                        scale: 1.6,
                        strength: state.turbulence,
                        speed: 0.7,
                    },
                ]),
        );
    }
    if state.smoke {
        if let Some(&id) = effect_ids.get(1) {
            // Raise the smoke origin to the flame tip so it rises above the fire
            // instead of compositing over the core (alpha smoke draws in the OIT
            // pass, after the additive flame, so overlap always reads as "in
            // front"). Wind carries it as it climbs.
            items.push(
                ParticleItem::new(id)
                    .at([state.wind * 0.15, 0.0, 0.9])
                    .with_emitter(scaled(smoke_emitter(), state.intensity)),
            );
        }
    }
    if let Some(&id) = effect_ids.get(2) {
        items.push(ParticleItem::new(id).with_emitter(scaled(embers_emitter(), state.intensity)));
    }
    items
}
