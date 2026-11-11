//! Showcase 18: Soft particles.
//!
//! A low-lying smoke cloud sits on the ground plane. Without soft particles it
//! shows a hard clip line where each billboard cuts into the floor; with soft
//! particles on, the renderer's read-only-depth pass fades each billboard's
//! alpha as it nears the surface, so the smoke melts into the ground.
//!
//! Toggling "Soft fade" swaps between two registered variants of the same
//! effect (soft is baked into the pipeline, so it is a per-effect property, not
//! a per-frame uniform). The ground plane and a sun light are submitted by
//! `main.rs` for this mode.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, Gradient, ParticleBlend, ParticleItem,
    ParticleItems, SpawnRate, SpawnShape, VelocityDist,
};

fn smoke_emitter() -> Emitter {
    Emitter {
        rate: SpawnRate::PerSecond(220.0),
        lifetime: (1.6, 2.6),
        // A flat, wide volume straddling the floor (z = 0), so billboards
        // intersect the ground and the soft fade is visible at the contact line.
        spawn: SpawnShape::Sphere {
            radius: 2.2,
            volume: true,
        },
        velocity: VelocityDist::UniformCone {
            axis: [0.0, 0.0, 1.0],
            half_angle: 1.2,
            min_speed: 0.15,
            max_speed: 0.5,
        },
        colour: [0.7, 0.72, 0.78, 0.5],
        size: 1.6,
    }
}

fn base() -> EffectAsset {
    EffectAsset::new("soft-smoke")
        .with_capacity(6_000)
        .with_blend(ParticleBlend::Alpha)
        .with_emitter(smoke_emitter())
        .force(ForceModifier::Accel([0.0, 0.0, 0.15]))
        .force(ForceModifier::Drag(0.5))
        .force(ForceModifier::CurlNoise {
            scale: 0.5,
            strength: 0.6,
            speed: 0.3,
        })
        .with_gradient(
            Gradient::new()
                .with_colour(vec![(0.0, [0.8, 0.82, 0.88]), (1.0, [0.5, 0.52, 0.58])])
                .with_alpha(vec![(0.0, 0.0), (0.2, 0.7), (1.0, 0.0)])
                .with_size(vec![(0.0, 0.6), (1.0, 1.6)]),
        )
}

/// Two variants: `[soft, hard]`.
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![
        ("soft-smoke-soft", base().with_soft(0.7)),
        ("soft-smoke-hard", base()),
    ]
}

/// UI state.
pub struct State {
    /// Draw the soft-fading variant (else the hard-edged one).
    pub soft: bool,
}

impl Default for State {
    fn default() -> Self {
        Self { soft: true }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.checkbox(&mut state.soft, "Soft fade");
    ui.separator();
    ui.label("Smoke sits on the ground plane. Soft fade dissolves each billboard into the floor instead of showing a hard intersection line.");
}

/// Submit the selected variant at the origin (on the floor).
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    let id = if state.soft {
        effect_ids.first()
    } else {
        effect_ids.get(1)
    };
    if let Some(&id) = id {
        // The smoke casts a soft, alpha-dithered shadow onto the ground (see the
        // alpha-aware particle shadow pass).
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
    }
    items
}
